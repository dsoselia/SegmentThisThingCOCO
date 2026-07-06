#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline.config import load_config
from stt_pipeline.losses import multimask_segmentation_loss
from stt_pipeline.mae import FoveatedMAE
from stt_pipeline.modeling import build_foveator, build_model, load_segmentation_components
from stt_pipeline.runtime import (
    cleanup_distributed,
    configure_torch_runtime,
    get_autocast_dtype,
    get_device,
    init_distributed,
    seed_everything,
    unwrap_model,
)
from stt_pipeline.trainers import (
    FoveatedSegmentationTrainingModule,
    _build_loader,
    _load_mae_segmentation_initialization,
    _materialize_segmentation_batch,
    _maybe_wrap_ddp,
    _normalize_tokens,
    _set_lambda_trainability,
)
from stt_pipeline.transforms import build_model_inputs_batch, build_precomputed_crop_inputs_batch


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} is not finite: {value.detach().cpu()}")


def _check_fixed_lambda(module: torch.nn.Module) -> dict[str, float]:
    foveator = getattr(unwrap_model(module), "foveator", None)
    raw_lambda = getattr(foveator, "raw_lambda_scale", None)
    if raw_lambda is None:
        return {"lambda_present": 0.0}
    grad_abs = 0.0 if raw_lambda.grad is None else float(raw_lambda.grad.detach().abs().max().cpu())
    if raw_lambda.requires_grad:
        raise RuntimeError("Expected fixed logrect lambda, but raw_lambda_scale requires gradients")
    if grad_abs != 0.0:
        raise RuntimeError(f"Expected fixed logrect lambda grad to be zero, got {grad_abs}")
    lambda_scale = getattr(foveator, "lambda_scale", None)
    return {
        "lambda_present": 1.0,
        "lambda_scale": float(lambda_scale.detach().cpu()) if lambda_scale is not None else math.nan,
        "lambda_grad_abs": grad_abs,
        "lambda_learnable": float(raw_lambda.requires_grad),
    }


def _first_gradient_stats(module: torch.nn.Module) -> tuple[str, torch.Tensor]:
    for name, parameter in module.named_parameters():
        if "raw_lambda_scale" in name:
            continue
        if parameter.requires_grad and parameter.grad is not None:
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"Gradient for {name} is not finite")
            norm = grad.float().norm()
            if float(norm.cpu()) > 0.0:
                stats = torch.stack(
                    [
                        norm,
                        grad.float().sum(),
                        grad.float().abs().max(),
                    ]
                )
                return name, stats
    raise RuntimeError("No nonzero finite gradient found")


def _assert_ddp_synced(name: str, stats: torch.Tensor) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return {
            f"{name}_grad_norm": float(stats[0].detach().cpu()),
            f"{name}_grad_sum": float(stats[1].detach().cpu()),
            f"{name}_grad_max_abs": float(stats[2].detach().cpu()),
        }
    gathered = [torch.zeros_like(stats) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, stats)
    stacked = torch.stack(gathered)
    spread = (stacked.max(dim=0).values - stacked.min(dim=0).values).abs()
    if bool((spread > 1e-3).any().detach().cpu()):
        raise RuntimeError(f"DDP gradient stats differ across ranks for {name}: {stacked.detach().cpu().tolist()}")
    return {
        f"{name}_grad_norm": float(stacked[0, 0].detach().cpu()),
        f"{name}_grad_sum": float(stacked[0, 1].detach().cpu()),
        f"{name}_grad_max_abs": float(stacked[0, 2].detach().cpu()),
    }


def _run_mae_check(config, device: torch.device, ctx) -> dict[str, float]:
    if config.mae.worker_pre_crop and config.mae.views_per_image != 1:
        raise ValueError("mae.worker_pre_crop currently requires mae.views_per_image == 1")
    loader, sampler = _build_loader(
        config.mae.train_manifest,
        config.runtime,
        config.mae.margin,
        1,
        ctx,
        task="mae",
        mae_worker_pre_crop=config.mae.worker_pre_crop,
        mae_worker_pre_crop_size=config.model.pattern_size,
        mae_worker_pre_crop_jitter_radius=config.mae.worker_pre_crop_jitter_radius,
    )
    if sampler is not None:
        sampler.set_epoch(0)
    samples = next(iter(loader))
    foveator = build_foveator(config.model).to(device)
    segment_model = build_model(config.model.size, foveator)
    model = FoveatedMAE(
        image_encoder=segment_model.image_encoder,
        feature_dim=segment_model.mask_decoder.pos_enc.shape[-1],
        token_size=config.model.token_size,
        foveator=foveator,
    ).to(device)
    model = _maybe_wrap_ddp(model, device, ctx)
    _set_lambda_trainability(config, model, 0)
    active_foveator = unwrap_model(model).foveator
    images = []
    centers = []
    crop_bounds = []
    original_sizes = []
    for sample in samples:
        image = sample.image.to(device, non_blocking=config.runtime.pin_memory)
        center = sample.center.to(device, non_blocking=config.runtime.pin_memory)
        if config.mae.worker_pre_crop:
            if sample.crop_bounds is None or sample.original_image_size is None:
                raise RuntimeError("worker_pre_crop sample is missing crop metadata")
            crop_bounds.append(sample.crop_bounds.to(device, non_blocking=config.runtime.pin_memory))
            original_sizes.append(sample.original_image_size.to(device, non_blocking=config.runtime.pin_memory))
        images.append(image)
        centers.append(center)
    if config.mae.worker_pre_crop:
        tokens, valid_mask, _ = build_precomputed_crop_inputs_batch(
            images,
            torch.stack(original_sizes),
            torch.stack(crop_bounds),
            active_foveator,
        )
    else:
        tokens, valid_mask, _ = build_model_inputs_batch(images, torch.stack(centers), active_foveator)
    amp_dtype = get_autocast_dtype(config.runtime.amp_dtype)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        loss, _ = model(tokens.float(), valid_mask.to(device, non_blocking=True), config.mae.mask_ratio)
    _assert_finite("mae loss", loss)
    loss.backward()
    parameter_name, stats = _first_gradient_stats(model)
    payload = {
        "mae_loss": float(loss.detach().cpu()),
        "mae_batch": float(tokens.shape[0]),
        **_assert_ddp_synced("mae", stats),
        **_check_fixed_lambda(model),
    }
    payload["mae_gradient_parameter"] = parameter_name
    return payload


def _run_segmentation_check(config, device: torch.device, ctx) -> dict[str, float]:
    loader, sampler = _build_loader(
        config.segmentation.train_manifest,
        config.runtime,
        config.segmentation.margin,
        config.segmentation.max_segments_per_image,
        ctx,
        task="segmentation",
        model_config=config.model,
        prompt_noise_std=config.segmentation.prompt_noise_std,
        sample_multiple_segments_per_image=config.segmentation.sample_multiple_segments_per_image,
    )
    if sampler is not None:
        sampler.set_epoch(0)
    batch = next(iter(loader))
    foveator = build_foveator(config.model).to(device)
    segment_model = build_model(config.model.size, foveator).to(device)
    model = FoveatedSegmentationTrainingModule(segment_model, foveator).to(device)
    if config.segmentation.init_checkpoint:
        load_segmentation_components(model.segment_model, model.foveator, config.segmentation.init_checkpoint, strict=True)
    if config.segmentation.pretrained_mae_checkpoint:
        _load_mae_segmentation_initialization(model, config.segmentation.pretrained_mae_checkpoint)
    if config.segmentation.pretrained_encoder:
        state = torch.load(config.segmentation.pretrained_encoder, map_location="cpu", weights_only=False)
        encoder_state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.image_encoder.load_state_dict(encoder_state, strict=False)
    model = _maybe_wrap_ddp(model, device, ctx)
    _set_lambda_trainability(config, model, 0)
    active_foveator = unwrap_model(model).foveator
    image_tokens, valid_masks, target_masks, _ = _materialize_segmentation_batch(
        batch,
        active_foveator,
        device,
        non_blocking=config.runtime.pin_memory,
    )
    amp_dtype = get_autocast_dtype(config.runtime.amp_dtype)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        pred_masks, pred_iou = model(_normalize_tokens(image_tokens), valid_masks)
        loss, _ = multimask_segmentation_loss(
            pred_masks,
            pred_iou,
            target_masks,
            focal_weight=config.segmentation.focal_weight,
            dice_weight=config.segmentation.dice_weight,
            iou_weight=config.segmentation.iou_weight,
            focal_alpha=config.segmentation.focal_alpha,
            focal_gamma=config.segmentation.focal_gamma,
        )
    _assert_finite("segmentation loss", loss)
    loss.backward()
    parameter_name, stats = _first_gradient_stats(model)
    payload = {
        "segmentation_loss": float(loss.detach().cpu()),
        "segmentation_batch": float(image_tokens.shape[0]),
        **_assert_ddp_synced("segmentation", stats),
        **_check_fixed_lambda(model),
    }
    payload["segmentation_gradient_parameter"] = parameter_name
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("Check DDP gradient flow for SA-1B logrect pipelines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["mae", "seg", "all"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_torch_runtime(config.runtime)
    ctx = init_distributed(config.runtime)
    seed_everything(config.runtime.seed + ctx.rank)
    device = get_device(config.runtime.device, ctx)
    try:
        payload: dict[str, object] = {
            "rank": ctx.rank,
            "local_rank": ctx.local_rank,
            "world_size": ctx.world_size,
            "device": str(device),
            "config": args.config,
        }
        if args.task in ("mae", "all"):
            payload.update(_run_mae_check(config, device, ctx))
        if args.task in ("seg", "all"):
            payload.update(_run_segmentation_check(config, device, ctx))
        if ctx.rank == 0:
            print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
