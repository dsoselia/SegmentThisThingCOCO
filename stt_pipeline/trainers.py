from __future__ import annotations

from dataclasses import replace
import inspect
import math
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import ExperimentConfig
from .data import MAETrainingManifestDataset, Sample, SegmentationBatch, SegmentationTrainingManifestDataset, collate_samples, collate_segmentation_samples
from .evaluate import evaluate_checkpoint
from .losses import multimask_segmentation_loss
from .mae import FoveatedMAE
from .modeling import attach_foveator_if_learnable, build_foveator, build_model, load_checkpoint, save_checkpoint
from .runtime import (
    DistributedContext,
    append_jsonl,
    build_run_dir,
    cleanup_distributed,
    configure_torch_runtime,
    distributed_barrier,
    finish_wandb_run,
    get_autocast_dtype,
    get_device,
    get_open_file_descriptor_count,
    get_open_file_limit,
    gpu_memory_snapshot,
    infer_run_dir_from_checkpoint,
    init_wandb_run,
    init_distributed,
    save_run_metadata,
    seed_everything,
    seed_worker,
    set_rng_state,
    unwrap_model,
    update_run_status,
    wandb_log,
)
from .transforms import build_model_inputs, project_mask_to_foveation


def _scheduled_batch_size(step: int, schedule: dict[str, int], default: int) -> int:
    current = default
    for key, value in sorted(((int(k), v) for k, v in schedule.items()), key=lambda item: item[0]):
        if step >= key:
            current = value
    return current


def _loader_supports(name: str) -> bool:
    return name in inspect.signature(DataLoader).parameters


def _build_loader(
    manifest: str,
    runtime,
    margin: int,
    max_segments_per_image: int,
    ctx: DistributedContext,
    *,
    task: str,
    model_config=None,
    prompt_noise_std: float = 0.0,
    sample_multiple_segments_per_image: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    raw_segmentation_samples = bool(task == "segmentation" and getattr(model_config, "log_rect_learnable", False))
    if task == "segmentation":
        dataset = SegmentationTrainingManifestDataset(
            manifest_path=manifest,
            margin=margin,
            model_config=model_config,
            max_segments_per_image=max_segments_per_image,
            seed=runtime.seed + ctx.rank * 100_000,
            prompt_noise_std=prompt_noise_std,
            sample_multiple_segments_per_image=sample_multiple_segments_per_image,
            raw_samples=raw_segmentation_samples,
        )
        collate_fn = collate_samples if raw_segmentation_samples else collate_segmentation_samples
    else:
        dataset = MAETrainingManifestDataset(
            manifest_path=manifest,
            margin=margin,
            seed=runtime.seed + ctx.rank * 100_000,
        )
        collate_fn = collate_samples
    sampler = None
    shuffle = True
    if ctx.enabled:
        sampler = DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, drop_last=True)
        shuffle = False

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": runtime.micro_batch_size,
        "shuffle": shuffle,
        "num_workers": runtime.num_workers,
        "collate_fn": collate_fn,
        "drop_last": True,
        "pin_memory": runtime.pin_memory,
        "persistent_workers": runtime.persistent_workers and runtime.num_workers > 0,
        "worker_init_fn": seed_worker,
        "sampler": sampler,
        "timeout": runtime.dataloader_timeout_s,
    }
    if runtime.num_workers > 0:
        kwargs["prefetch_factor"] = runtime.prefetch_factor
        if runtime.multiprocessing_context:
            kwargs["multiprocessing_context"] = runtime.multiprocessing_context
    if runtime.pin_memory_device and runtime.pin_memory_device != "cuda" and _loader_supports("pin_memory_device"):
        kwargs["pin_memory_device"] = runtime.pin_memory_device
    if _loader_supports("in_order"):
        kwargs["in_order"] = runtime.dataloader_in_order
    return DataLoader(**kwargs), sampler


def _flatten_samples(samples: list[Sample | list[Sample]]) -> list[Sample]:
    flat_samples: list[Sample] = []
    for sample in samples:
        if isinstance(sample, list):
            flat_samples.extend(sample)
        else:
            flat_samples.append(sample)
    return flat_samples


def _build_learnable_segmentation_tensors(
    samples: list[Sample | list[Sample]],
    *,
    foveator,
    device: torch.device,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], int]:
    flat_samples = _flatten_samples(samples)
    token_batch = []
    valid_batch = []
    target_batch = []
    preprocessing = {
        "image_decode_time": 0.0,
        "mask_decode_time": 0.0,
        "prompt_sample_time": 0.0,
        "foveation_build_time": 0.0,
        "target_projection_time": 0.0,
        "host_to_device_time": 0.0,
        "valid_token_count": 0.0,
    }
    for sample in flat_samples:
        preprocessing["image_decode_time"] += float(sample.preprocessing.get("image_decode_time", 0.0))
        preprocessing["mask_decode_time"] += float(sample.preprocessing.get("mask_decode_time", 0.0))
        preprocessing["prompt_sample_time"] += float(sample.preprocessing.get("prompt_sample_time", 0.0))
        host_to_device_start = time.time()
        image = sample.image.to(device, non_blocking=pin_memory)
        mask = sample.mask.to(device, non_blocking=pin_memory)
        center = sample.center.to(device, non_blocking=pin_memory)
        preprocessing["host_to_device_time"] += time.time() - host_to_device_start

        foveation_start = time.time()
        tokens, valid_mask, _ = build_model_inputs(image, center, foveator)
        preprocessing["foveation_build_time"] += time.time() - foveation_start
        target_start = time.time()
        target, _ = project_mask_to_foveation(foveator, mask, center)
        preprocessing["target_projection_time"] += time.time() - target_start
        preprocessing["valid_token_count"] += float(valid_mask.sum().detach().cpu())
        token_batch.append(tokens)
        valid_batch.append(valid_mask)
        target_batch.append(target)

    return (
        torch.stack(token_batch).float(),
        torch.stack(valid_batch).bool(),
        torch.stack(target_batch).float(),
        preprocessing,
        len(flat_samples),
    )


def _build_mae_token_tensors(
    samples: list[Sample],
    *,
    foveator,
    device: torch.device,
    pin_memory: bool,
    views_per_image: int,
    jitter_radius: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], int]:
    token_batch = []
    valid_batch = []
    preprocessing = {
        "host_to_device_time": 0.0,
        "foveation_build_time": 0.0,
        "valid_token_count": 0.0,
    }
    for sample in samples:
        h2d_start = time.time()
        image = sample.image.to(device, non_blocking=pin_memory)
        base_center = sample.center.to(device, non_blocking=pin_memory)
        preprocessing["host_to_device_time"] += time.time() - h2d_start
        for _ in range(views_per_image):
            if jitter_radius > 0:
                jitter = torch.randint(-jitter_radius, jitter_radius + 1, (2,), device=device)
                center = (base_center + jitter).clamp(min=0)
            else:
                center = base_center
            foveation_start = time.time()
            tokens, valid_mask, _ = build_model_inputs(image, center, foveator)
            preprocessing["foveation_build_time"] += time.time() - foveation_start
            preprocessing["valid_token_count"] += float(valid_mask.sum().detach().cpu())
            token_batch.append(tokens)
            valid_batch.append(valid_mask)

    return (
        torch.stack(token_batch).float(),
        torch.stack(valid_batch).bool(),
        preprocessing,
        len(samples),
    )


def _normalize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tokens.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tokens.device).view(1, 1, 3, 1, 1)
    return (tokens / 255.0 - mean) / std


def _mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    if not metrics_list:
        return summary
    for key in metrics_list[0]:
        summary[key] = sum(item[key] for item in metrics_list) / len(metrics_list)
    return summary


def _make_run_dir(config: ExperimentConfig, prefix: str) -> Path:
    if config.runtime.resume_from:
        run_dir = infer_run_dir_from_checkpoint(config.runtime.resume_from)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        return run_dir
    return build_run_dir(config.runtime.output_dir, prefix)


def _update_status(
    run_dir: Path,
    config: ExperimentConfig,
    *,
    phase: str,
    state: str,
    step: int,
    ctx: DistributedContext,
    extra_lines: list[str] | None = None,
) -> None:
    if not ctx.is_main_process:
        return
    lines = [
        f"# STT Run Status",
        "",
        f"- phase: `{phase}`",
        f"- state: `{state}`",
        f"- step: `{step}`",
        f"- world_size: `{ctx.world_size}`",
        f"- output_dir: `{run_dir}`",
    ]
    if config.runtime.resume_from:
        lines.append(f"- resume_from: `{config.runtime.resume_from}`")
    if extra_lines:
        lines.extend(["", *extra_lines])
    update_run_status(run_dir, config.runtime.run_status_filename, lines)


def _mark_startup_stage(
    run_dir: Path,
    config: ExperimentConfig,
    ctx: DistributedContext,
    *,
    phase: str,
    step: int,
    stage: str,
    detail: str | None = None,
) -> None:
    extra = [
        f"- profile_name: `{config.runtime.profile_name}`",
        f"- startup_stage: `{stage}`",
        f"- torch_num_threads: `{torch.get_num_threads()}`",
        f"- torch_num_interop_threads: `{torch.get_num_interop_threads()}`",
        f"- sharing_strategy: `{torch.multiprocessing.get_sharing_strategy()}`",
        f"- multiprocessing_context: `{config.runtime.multiprocessing_context}`",
        f"- omp_num_threads: `{os.environ.get('OMP_NUM_THREADS')}`",
        f"- mkl_num_threads: `{os.environ.get('MKL_NUM_THREADS')}`",
        f"- rlimit_nofile_soft: `{get_open_file_limit()}`",
        f"- open_file_descriptors: `{get_open_file_descriptor_count()}`",
        f"- num_workers: `{config.runtime.num_workers}`",
        f"- micro_batch_size: `{config.runtime.micro_batch_size}`",
        f"- pin_memory_device: `{config.runtime.pin_memory_device}`",
        f"- dataloader_in_order: `{config.runtime.dataloader_in_order}`",
        f"- dataloader_timeout_s: `{config.runtime.dataloader_timeout_s}`",
    ]
    if detail:
        extra.append(f"- startup_detail: `{detail}`")
    _update_status(run_dir, config, phase=phase, state="starting", step=step, ctx=ctx, extra_lines=extra)


def _startup_context_lines(config: ExperimentConfig, device: torch.device, ctx: DistributedContext) -> list[str]:
    return [
        f"- profile_name: `{config.runtime.profile_name}`",
        f"- device: `{device}`",
        f"- hostname: `{socket.gethostname()}`",
        f"- rank: `{ctx.rank}`",
        f"- torch_num_threads: `{torch.get_num_threads()}`",
        f"- torch_num_interop_threads: `{torch.get_num_interop_threads()}`",
        f"- sharing_strategy: `{torch.multiprocessing.get_sharing_strategy()}`",
        f"- multiprocessing_context: `{config.runtime.multiprocessing_context}`",
        f"- omp_num_threads: `{os.environ.get('OMP_NUM_THREADS')}`",
        f"- mkl_num_threads: `{os.environ.get('MKL_NUM_THREADS')}`",
        f"- rlimit_nofile_soft: `{get_open_file_limit()}`",
        f"- open_file_descriptors: `{get_open_file_descriptor_count()}`",
        f"- num_workers: `{config.runtime.num_workers}`",
        f"- micro_batch_size: `{config.runtime.micro_batch_size}`",
        f"- pin_memory_device: `{config.runtime.pin_memory_device}`",
        f"- dataloader_in_order: `{config.runtime.dataloader_in_order}`",
        f"- dataloader_timeout_s: `{config.runtime.dataloader_timeout_s}`",
    ]


def _startup_detail(started_at: float, detail: str | None = None) -> str:
    elapsed = time.time() - started_at
    base = f"elapsed_s={elapsed:.3f}"
    if detail:
        return f"{base}; {detail}"
    return base


def _save_training_checkpoint(
    run_dir: Path,
    config: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    step: int,
    ctx: DistributedContext,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    if not ctx.is_main_process:
        return None
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        config=config,
        extra=extra,
        save_optimizer_state=config.runtime.save_optimizer_state,
        save_rng_state=config.runtime.save_rng_state,
    )
    (run_dir / "last_checkpoint.txt").write_text(str(checkpoint_path) + "\n")

    keep = max(0, int(config.runtime.checkpoint_keep_last))
    checkpoint_paths = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if keep > 0 and len(checkpoint_paths) > keep:
        for old_path in checkpoint_paths[:-keep]:
            old_path.unlink(missing_ok=True)
    return checkpoint_path


def _maybe_wrap_ddp(model: torch.nn.Module, device: torch.device, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
    )


def _load_resume_state(
    config: ExperimentConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ctx: DistributedContext,
) -> int:
    if not config.runtime.resume_from:
        return 0
    state = load_checkpoint(
        model,
        config.runtime.resume_from,
        optimizer=optimizer,
        scaler=scaler,
        strict=True,
        restore_training_state=True,
    )
    set_rng_state(state.get("rng_state"))
    start_step = int(state.get("step", -1)) + 1
    distributed_barrier(ctx)
    return start_step


def _build_iterator(loader: DataLoader, sampler: DistributedSampler | None, epoch: int):
    if sampler is not None:
        sampler.set_epoch(epoch)
    return iter(loader)


def _advance_iterator(loader: DataLoader, sampler: DistributedSampler | None, iterator, epoch: int):
    try:
        batch = next(iterator)
        return batch, iterator, epoch, False
    except StopIteration:
        epoch += 1
        iterator = _build_iterator(loader, sampler, epoch)
        batch = next(iterator)
        return batch, iterator, epoch, True


def _mark_batch_failure(
    run_dir: Path,
    config: ExperimentConfig,
    ctx: DistributedContext,
    *,
    phase: str,
    step: int,
    stage: str,
    exc: Exception,
 ) -> dict[str, Any]:
    _update_status(
        run_dir,
        config,
        phase=phase,
        state="failed",
        step=step,
        ctx=ctx,
        extra_lines=[
            f"- startup_stage: `{stage}`",
            f"- exception_type: `{type(exc).__name__}`",
            f"- exception: `{str(exc)}`",
            f"- open_file_descriptors: `{get_open_file_descriptor_count()}`",
        ],
    )


def _mark_running_status(
    run_dir: Path,
    config: ExperimentConfig,
    ctx: DistributedContext,
    *,
    phase: str,
    step: int,
    epoch: int,
    loop_stage: str = "training",
    detail: str | None = None,
) -> None:
    extra_lines = [
        f"- profile_name: `{config.runtime.profile_name}`",
        f"- epoch: `{epoch}`",
        f"- loop_stage: `{loop_stage}`",
        f"- open_file_descriptors: `{get_open_file_descriptor_count()}`",
    ]
    if detail:
        extra_lines.append(f"- loop_detail: `{detail}`")
    _update_status(run_dir, config, phase=phase, state="running", step=step, ctx=ctx, extra_lines=extra_lines)


def _record_training_metrics(
    *,
    metrics_path: Path,
    step: int,
    start_step: int,
    target_batch: int,
    accum_steps: int,
    elapsed: float,
    batch_metrics: list[dict[str, float]],
    data_time: float,
    compute_time: float,
    foveator,
    device: torch.device,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    is_warmup_step = step == start_step
    summary = {
        "step": step,
        "is_warmup_step": is_warmup_step,
        "post_startup_step_index": None if is_warmup_step else (step - start_step - 1),
        "seconds_per_step": elapsed,
        "samples_per_second": target_batch / max(elapsed, 1e-6),
        "effective_batch": target_batch,
        "accum_steps": accum_steps,
        "data_time": data_time,
        "compute_time": compute_time,
        "tokens_per_second": (target_batch * foveator.get_num_tokens()) / max(elapsed, 1e-6),
    }
    summary.update(_mean_metrics(batch_metrics))
    if extra_metrics:
        summary.update(extra_metrics)
    if hasattr(foveator, "get_learned_warp_metrics"):
        summary.update(foveator.get_learned_warp_metrics())
    summary["open_file_descriptors"] = get_open_file_descriptor_count() or 0
    summary.update(gpu_memory_snapshot(device))
    append_jsonl(metrics_path, summary)
    return summary


def _run_periodic_eval(
    *,
    config: ExperimentConfig,
    checkpoint_path: Path,
    run_dir: Path,
    step: int,
    ctx: DistributedContext,
) -> dict[str, Any] | None:
    summary = evaluate_checkpoint(
        config,
        checkpoint_path=str(checkpoint_path),
        run_dir=run_dir / "eval",
        summary_name=f"summary_step_{step:07d}.json",
        enable_wandb=False,
    )
    if ctx.is_main_process:
        return summary
    return None


@torch.no_grad()
def _run_mae_validation(
    *,
    config: ExperimentConfig,
    trainer: torch.nn.Module,
    foveator,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    iterator,
    epoch: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    ctx: DistributedContext,
    step: int,
) -> tuple[dict[str, Any] | None, Any, int]:
    was_training = trainer.training
    foveator_was_training = getattr(foveator, "training", False)
    trainer.eval()
    if hasattr(foveator, "eval"):
        foveator.eval()

    local_loss_sum = 0.0
    local_mask_fraction_sum = 0.0
    local_examples = 0
    local_token_views = 0
    validation_start = time.time()
    max_examples = max(1, int(config.mae.val_max_examples))
    views_per_image = max(1, int(config.mae.val_views_per_image))
    jitter_radius = max(0, int(config.mae.val_jitter_radius))
    rng_devices = []
    if device.type == "cuda":
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(config.runtime.seed + step * 1_000_003 + ctx.rank)
        if device.type == "cuda":
            torch.cuda.manual_seed(config.runtime.seed + step * 1_000_003 + ctx.rank)
        while local_examples < max_examples:
            samples, iterator, epoch, _ = _advance_iterator(loader, sampler, iterator, epoch)
            tokens, valid_mask, _, example_count = _build_mae_token_tensors(
                samples,
                foveator=foveator,
                device=device,
                pin_memory=config.runtime.pin_memory,
                views_per_image=views_per_image,
                jitter_radius=jitter_radius,
            )
            token_views = int(tokens.shape[0])
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss, metrics = trainer(tokens, valid_mask, config.mae.mask_ratio)
            if torch.isfinite(loss):
                local_loss_sum += float(loss.detach().cpu()) * token_views
                local_mask_fraction_sum += float(metrics["mask_fraction"]) * token_views
                local_token_views += token_views
            local_examples += example_count

    elapsed = time.time() - validation_start
    totals = torch.tensor(
        [local_loss_sum, local_mask_fraction_sum, float(local_examples), float(local_token_views), elapsed],
        device=device,
        dtype=torch.float64,
    )
    if ctx.enabled:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    if was_training:
        trainer.train()
    if foveator_was_training and hasattr(foveator, "train"):
        foveator.train()

    if not ctx.is_main_process:
        return None, iterator, epoch

    token_views_total = max(float(totals[3].item()), 1.0)
    summary = {
        "step": step,
        "loss": float(totals[0].item() / token_views_total),
        "mask_fraction": float(totals[1].item() / token_views_total),
        "examples": int(totals[2].item()),
        "token_views": int(totals[3].item()),
        "seconds": float(totals[4].item()),
        "views_per_image": views_per_image,
        "jitter_radius": jitter_radius,
    }
    return summary, iterator, epoch


@torch.no_grad()
def run_segmentation_preflight(config: ExperimentConfig) -> dict[str, Any]:
    startup_started_at = time.time()
    configure_torch_runtime(config.runtime)
    seed_everything(config.runtime.seed)
    device = get_device(config.runtime.device)
    loader, _ = _build_loader(
        config.segmentation.train_manifest,
        config.runtime,
        config.segmentation.margin,
        config.segmentation.max_segments_per_image,
        DistributedContext(enabled=False),
        task="segmentation",
        model_config=config.model,
        prompt_noise_std=config.segmentation.prompt_noise_std,
    )
    foveator = build_foveator(config.model).to(device)
    model = build_model(config.model.size, foveator).to(device).eval()
    model = attach_foveator_if_learnable(model, foveator)
    if config.segmentation.init_checkpoint:
        load_checkpoint(model, config.segmentation.init_checkpoint, strict=True)
    iterator = iter(loader)
    batch = next(iterator)
    if config.model.log_rect_learnable:
        tokens, valid_masks, target, preprocessing, _ = _build_learnable_segmentation_tensors(
            batch,
            foveator=foveator,
            device=device,
            pin_memory=config.runtime.pin_memory,
        )
        center = _flatten_samples(batch)[0].center.tolist()
    else:
        tokens = batch.tokens.to(device, non_blocking=config.runtime.pin_memory)
        valid_masks = batch.valid_masks.to(device, non_blocking=config.runtime.pin_memory)
        target = batch.target_masks.to(device, non_blocking=config.runtime.pin_memory)
        preprocessing = batch.preprocessing
        center = batch.centers[0].tolist()
    pred_masks, pred_iou = model(_normalize_tokens(tokens), valid_masks)
    return {
        "manifest": config.segmentation.train_manifest,
        "dataset_len": len(loader.dataset),
        "batch_len": int(tokens.shape[0]),
        "center": center,
        "tokens_shape": list(tokens.shape[1:]),
        "target_shape": list(target.shape[1:]),
        "pred_masks_shape": list(pred_masks.shape),
        "pred_iou_shape": list(pred_iou.shape),
        "device": str(device),
        "tokenizer_type": config.model.tokenizer_type,
        "image_decode_time": preprocessing.get("image_decode_time"),
        "mask_decode_time": preprocessing.get("mask_decode_time"),
        "prompt_sample_time": preprocessing.get("prompt_sample_time"),
        "foveation_build_time": preprocessing.get("foveation_build_time"),
        "target_projection_time": preprocessing.get("target_projection_time"),
        "startup_elapsed_s": round(time.time() - startup_started_at, 3),
    }


def run_mae_pretraining(config: ExperimentConfig) -> Path:
    ctx = init_distributed(config.runtime)
    wandb_run = None
    try:
        startup_started_at = time.time()
        configure_torch_runtime(config.runtime)
        seed_everything(config.runtime.seed + ctx.rank)
        device = get_device(config.runtime.device, ctx)
        amp_dtype = get_autocast_dtype(config.runtime.amp_dtype)
        run_dir = _make_run_dir(config, "mae")
        if ctx.is_main_process:
            save_run_metadata(run_dir, config, {"phase": "mae", "profile_name": config.runtime.profile_name})
            wandb_run, wandb_status = init_wandb_run(
                run_dir,
                config,
                extra={"phase": "mae", "profile_name": config.runtime.profile_name},
            )
            _update_status(run_dir, config, phase="mae", state="starting", step=0, ctx=ctx)
            _mark_startup_stage(
                run_dir, config, ctx, phase="mae", step=0, stage="metadata_saved", detail=_startup_detail(startup_started_at)
            )
            _update_status(
                run_dir,
                config,
                phase="mae",
                state="starting",
                step=0,
                ctx=ctx,
                extra_lines=[* _startup_context_lines(config, device, ctx), f"- wandb_status: `{wandb_status}`"],
            )

        loader, sampler = _build_loader(config.mae.train_manifest, config.runtime, config.mae.margin, 1, ctx, task="mae")
        val_loader = None
        val_sampler = None
        val_iterator = None
        val_epoch = 0
        mae_val_enabled = bool(config.mae.val_manifest and config.mae.val_every and config.mae.val_every > 0)
        if mae_val_enabled:
            val_loader, val_sampler = _build_loader(config.mae.val_manifest, config.runtime, config.mae.margin, 1, ctx, task="mae")
            val_iterator = _build_iterator(val_loader, val_sampler, val_epoch)
        if ctx.is_main_process:
            loader_detail = f"dataset_len={len(loader.dataset)}"
            if val_loader is not None:
                loader_detail += f"; val_dataset_len={len(val_loader.dataset)}; val_every={config.mae.val_every}"
            _mark_startup_stage(
                run_dir,
                config,
                ctx,
                phase="mae",
                step=0,
                stage="loader_built",
                detail=_startup_detail(startup_started_at, loader_detail),
            )
        foveator = build_foveator(config.model).to(device)
        segment_model = build_model(config.model.size, foveator)
        trainer = FoveatedMAE(
            image_encoder=segment_model.image_encoder,
            feature_dim=segment_model.mask_decoder.pos_enc.shape[-1],
            token_size=config.model.token_size,
        ).to(device)
        trainer = _maybe_wrap_ddp(trainer, device, ctx)
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=config.mae.lr, weight_decay=config.mae.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
        epoch = 0
        iterator = _build_iterator(loader, sampler, epoch)
        world_batch = config.runtime.micro_batch_size * ctx.world_size * config.mae.views_per_image
        start_step = _load_resume_state(config, trainer, optimizer, scaler, ctx)
        if ctx.is_main_process:
            _update_status(run_dir, config, phase="mae", state="running", step=start_step, ctx=ctx)

        for step in range(start_step, config.runtime.num_steps):
            target_batch = _scheduled_batch_size(step, config.mae.target_batch_schedule, config.mae.effective_batch_size)
            accum_steps = max(1, math.ceil(target_batch / world_batch))
            optimizer.zero_grad(set_to_none=True)
            start = time.time()
            data_time = 0.0
            compute_time = 0.0
            running = []
            batch_fetch_total = 0.0
            host_to_device_total = 0.0
            foveation_build_total = 0.0
            for _ in range(accum_steps):
                data_start = time.time()
                try:
                    samples, iterator, epoch, rolled_epoch = _advance_iterator(loader, sampler, iterator, epoch)
                except Exception as exc:
                    _mark_batch_failure(run_dir, config, ctx, phase="mae", step=step, stage="next_batch", exc=exc)
                    raise
                if rolled_epoch and ctx.is_main_process:
                    _mark_running_status(
                        run_dir,
                        config,
                        ctx,
                        phase="mae",
                        step=step,
                        epoch=epoch,
                        loop_stage="epoch_rollover",
                        detail=f"dataset_len={len(loader.dataset)}",
                    )
                batch_fetch_time = time.time() - data_start
                data_time += batch_fetch_time
                batch_fetch_total += batch_fetch_time
                if step == start_step and ctx.is_main_process:
                    _mark_startup_stage(
                        run_dir,
                        config,
                        ctx,
                        phase="mae",
                        step=step,
                        stage="first_batch_fetched",
                        detail=_startup_detail(startup_started_at, f"batch_len={len(samples)}"),
                    )

                compute_start = time.time()
                tokens, valid_mask, preprocessing, _ = _build_mae_token_tensors(
                    samples,
                    foveator=foveator,
                    device=device,
                    pin_memory=config.runtime.pin_memory,
                    views_per_image=config.mae.views_per_image,
                    jitter_radius=8,
                )
                host_to_device_total += preprocessing["host_to_device_time"]
                foveation_build_total += preprocessing["foveation_build_time"]
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    loss, metrics = trainer(tokens, valid_mask, config.mae.mask_ratio)
                    loss = loss / accum_steps
                if step == start_step and ctx.is_main_process:
                    _mark_startup_stage(
                        run_dir, config, ctx, phase="mae", step=step, stage="first_forward_done", detail=_startup_detail(startup_started_at)
                    )
                scaler.scale(loss).backward()
                running.append(metrics)
                compute_time += time.time() - compute_start

            scaler.step(optimizer)
            scaler.update()
            if step == start_step and ctx.is_main_process:
                _mark_startup_stage(
                    run_dir,
                    config,
                    ctx,
                    phase="mae",
                    step=step,
                    stage="first_optimizer_step_done",
                    detail=_startup_detail(startup_started_at),
                )
            elapsed = time.time() - start
            if ctx.is_main_process:
                _mark_running_status(run_dir, config, ctx, phase="mae", step=step, epoch=epoch)
            if ctx.is_main_process and step % config.runtime.log_every == 0:
                summary = _record_training_metrics(
                    metrics_path=run_dir / "metrics.jsonl",
                    step=step,
                    start_step=start_step,
                    target_batch=target_batch,
                    accum_steps=accum_steps,
                    elapsed=elapsed,
                    batch_metrics=running,
                    data_time=data_time,
                    compute_time=compute_time,
                    foveator=foveator,
                    device=device,
                    extra_metrics={
                        "batch_fetch_time": batch_fetch_total,
                        "image_decode_time": 0.0,
                        "mask_decode_time": 0.0,
                        "prompt_sample_time": 0.0,
                        "foveation_build_time": foveation_build_total,
                        "target_projection_time": 0.0,
                        "host_to_device_time": host_to_device_total,
                        "stack_batch_time": 0.0,
                    },
                )
                wandb_log(wandb_run, {f"train/{k}": v for k, v in summary.items() if k != "step"}, step=step)
            if mae_val_enabled and val_loader is not None and val_iterator is not None and (
                step % int(config.mae.val_every) == 0 or step == (config.runtime.num_steps - 1)
            ):
                val_summary, val_iterator, val_epoch = _run_mae_validation(
                    config=config,
                    trainer=trainer,
                    foveator=foveator,
                    loader=val_loader,
                    sampler=val_sampler,
                    iterator=val_iterator,
                    epoch=val_epoch,
                    device=device,
                    amp_dtype=amp_dtype,
                    ctx=ctx,
                    step=step,
                )
                if val_summary is not None:
                    append_jsonl(run_dir / "mae_val_metrics.jsonl", val_summary)
                    wandb_log(wandb_run, {f"val/{k}": v for k, v in val_summary.items() if k != "step"}, step=step)
                    _update_status(
                        run_dir,
                        config,
                        phase="mae",
                        state="validated",
                        step=step,
                        ctx=ctx,
                        extra_lines=[
                            f"- val_loss: `{val_summary['loss']:.6f}`",
                            f"- val_examples: `{val_summary['examples']}`",
                            f"- val_token_views: `{val_summary['token_views']}`",
                        ],
                    )
            if step % config.runtime.save_every == 0 or step == (config.runtime.num_steps - 1):
                checkpoint_path = _save_training_checkpoint(
                    run_dir,
                    config,
                    trainer,
                    optimizer,
                    scaler,
                    step=step,
                    ctx=ctx,
                    extra={"phase": "mae"},
                )
                distributed_barrier(ctx)
                if checkpoint_path is not None:
                    _update_status(run_dir, config, phase="mae", state="checkpointed", step=step, ctx=ctx)

        if ctx.is_main_process:
            save_checkpoint(
                run_dir / "final_encoder.pt",
                unwrap_model(trainer).image_encoder,
                step=config.runtime.num_steps,
                config=config,
                save_optimizer_state=False,
                save_rng_state=False,
            )
            _update_status(run_dir, config, phase="mae", state="completed", step=config.runtime.num_steps, ctx=ctx)
            if wandb_run is not None:
                wandb_run.summary["phase"] = "mae"
                wandb_run.summary["completed_steps"] = config.runtime.num_steps
        distributed_barrier(ctx)
        return run_dir
    finally:
        finish_wandb_run(wandb_run)
        cleanup_distributed(ctx)


def run_segmentation_training(config: ExperimentConfig) -> Path:
    ctx = init_distributed(config.runtime)
    wandb_run = None
    try:
        startup_started_at = time.time()
        configure_torch_runtime(config.runtime)
        seed_everything(config.runtime.seed + ctx.rank)
        device = get_device(config.runtime.device, ctx)
        amp_dtype = get_autocast_dtype(config.runtime.amp_dtype)
        run_dir = _make_run_dir(config, "seg")
        if ctx.is_main_process:
            save_run_metadata(run_dir, config, {"phase": "segmentation", "profile_name": config.runtime.profile_name})
            wandb_run, wandb_status = init_wandb_run(
                run_dir,
                config,
                extra={"phase": "segmentation", "profile_name": config.runtime.profile_name},
            )
            _update_status(run_dir, config, phase="segmentation", state="starting", step=0, ctx=ctx)
            _mark_startup_stage(
                run_dir, config, ctx, phase="segmentation", step=0, stage="metadata_saved", detail=_startup_detail(startup_started_at)
            )
            _update_status(
                run_dir,
                config,
                phase="segmentation",
                state="starting",
                step=0,
                ctx=ctx,
                extra_lines=[*_startup_context_lines(config, device, ctx), f"- wandb_status: `{wandb_status}`"],
            )

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
        if ctx.is_main_process:
            _mark_startup_stage(
                run_dir,
                config,
                ctx,
                phase="segmentation",
                step=0,
                stage="loader_built",
                detail=_startup_detail(startup_started_at, f"dataset_len={len(loader.dataset)}"),
            )
        foveator = build_foveator(config.model).to(device)
        model = build_model(config.model.size, foveator).to(device)
        model = attach_foveator_if_learnable(model, foveator)
        if config.segmentation.init_checkpoint and not config.runtime.resume_from:
            load_checkpoint(model, config.segmentation.init_checkpoint, strict=True)
        if config.segmentation.pretrained_encoder and not config.runtime.resume_from:
            state = torch.load(config.segmentation.pretrained_encoder, map_location="cpu", weights_only=False)
            encoder_state = state["model"] if isinstance(state, dict) and "model" in state else state
            model.image_encoder.load_state_dict(encoder_state, strict=False)
        model = _maybe_wrap_ddp(model, device, ctx)
        if config.model.log_rect_learnable:
            warp_param_ids = {id(param) for param in foveator.parameters() if param.requires_grad}
            model_params = [param for param in model.parameters() if param.requires_grad and id(param) not in warp_param_ids]
            warp_params = [param for param in foveator.parameters() if param.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": model_params, "lr": config.segmentation.lr, "weight_decay": config.segmentation.weight_decay},
                    {"params": warp_params, "lr": config.segmentation.log_rect_warp_lr, "weight_decay": 0.0},
                ]
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.segmentation.lr, weight_decay=config.segmentation.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
        epoch = 0
        iterator = _build_iterator(loader, sampler, epoch)
        segments_per_image = config.segmentation.max_segments_per_image if config.segmentation.sample_multiple_segments_per_image else 1
        world_batch = config.runtime.micro_batch_size * ctx.world_size * segments_per_image
        start_step = _load_resume_state(config, model, optimizer, scaler, ctx)
        if ctx.is_main_process:
            _update_status(run_dir, config, phase="segmentation", state="running", step=start_step, ctx=ctx)

        for step in range(start_step, config.runtime.num_steps):
            target_batch = _scheduled_batch_size(
                step,
                config.segmentation.target_batch_schedule,
                config.segmentation.effective_batch_size,
            )
            accum_steps = max(1, math.ceil(target_batch / world_batch))
            optimizer.zero_grad(set_to_none=True)
            batch_metrics = []
            start = time.time()
            data_time = 0.0
            compute_time = 0.0
            batch_fetch_total = 0.0
            image_decode_total = 0.0
            mask_decode_total = 0.0
            prompt_sample_total = 0.0
            foveation_build_total = 0.0
            target_projection_total = 0.0
            host_to_device_total = 0.0
            valid_token_total = 0.0

            for _ in range(accum_steps):
                data_start = time.time()
                try:
                    batch, iterator, epoch, rolled_epoch = _advance_iterator(loader, sampler, iterator, epoch)
                except Exception as exc:
                    _mark_batch_failure(run_dir, config, ctx, phase="segmentation", step=step, stage="next_batch", exc=exc)
                    raise
                if rolled_epoch and ctx.is_main_process:
                    _mark_running_status(
                        run_dir,
                        config,
                        ctx,
                        phase="segmentation",
                        step=step,
                        epoch=epoch,
                        loop_stage="epoch_rollover",
                        detail=f"dataset_len={len(loader.dataset)}",
                    )
                batch_fetch_time = time.time() - data_start
                data_time += batch_fetch_time
                batch_fetch_total += batch_fetch_time
                if step == start_step and ctx.is_main_process:
                    fetched_batch_len = len(_flatten_samples(batch)) if config.model.log_rect_learnable else batch.tokens.shape[0]
                    _mark_startup_stage(
                        run_dir,
                        config,
                        ctx,
                        phase="segmentation",
                        step=step,
                        stage="first_batch_fetched",
                        detail=_startup_detail(startup_started_at, f"batch_len={fetched_batch_len}"),
                    )

                compute_start = time.time()
                if config.model.log_rect_learnable:
                    image_tokens, valid_masks, target_masks, preprocessing, actual_segments = _build_learnable_segmentation_tensors(
                        batch,
                        foveator=foveator,
                        device=device,
                        pin_memory=config.runtime.pin_memory,
                    )
                    host_to_device_total += float(preprocessing.get("host_to_device_time", 0.0))
                    image_decode_total += float(preprocessing.get("image_decode_time", 0.0))
                    mask_decode_total += float(preprocessing.get("mask_decode_time", 0.0))
                    prompt_sample_total += float(preprocessing.get("prompt_sample_time", 0.0))
                    foveation_build_total += float(preprocessing.get("foveation_build_time", 0.0))
                    target_projection_total += float(preprocessing.get("target_projection_time", 0.0))
                    valid_token_total += float(preprocessing.get("valid_token_count", 0.0))
                else:
                    host_to_device_start = time.time()
                    image_tokens = batch.tokens.to(device, non_blocking=config.runtime.pin_memory)
                    valid_masks = batch.valid_masks.to(device, non_blocking=config.runtime.pin_memory)
                    target_masks = batch.target_masks.to(device, non_blocking=config.runtime.pin_memory)
                    host_to_device_time = time.time() - host_to_device_start
                    host_to_device_total += host_to_device_time
                    image_decode_total += float(batch.preprocessing.get("image_decode_time", 0.0))
                    mask_decode_total += float(batch.preprocessing.get("mask_decode_time", 0.0))
                    prompt_sample_total += float(batch.preprocessing.get("prompt_sample_time", 0.0))
                    foveation_build_total += float(batch.preprocessing.get("foveation_build_time", 0.0))
                    target_projection_total += float(batch.preprocessing.get("target_projection_time", 0.0))
                    valid_token_total += float(batch.preprocessing.get("valid_token_count", 0.0))

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    pred_masks, pred_iou = model(_normalize_tokens(image_tokens), valid_masks)
                    loss, metrics = multimask_segmentation_loss(
                        pred_masks,
                        pred_iou,
                        target_masks,
                        focal_weight=config.segmentation.focal_weight,
                        dice_weight=config.segmentation.dice_weight,
                        iou_weight=config.segmentation.iou_weight,
                        focal_alpha=config.segmentation.focal_alpha,
                        focal_gamma=config.segmentation.focal_gamma,
                    )
                    loss = loss / accum_steps
                if step == start_step and ctx.is_main_process:
                    _mark_startup_stage(
                        run_dir,
                        config,
                        ctx,
                        phase="segmentation",
                        step=step,
                        stage="first_forward_done",
                        detail=_startup_detail(startup_started_at),
                    )
                scaler.scale(loss).backward()
                batch_metrics.append(metrics)
                compute_time += time.time() - compute_start

            scaler.step(optimizer)
            scaler.update()
            if step == start_step and ctx.is_main_process:
                _mark_startup_stage(
                    run_dir,
                    config,
                    ctx,
                    phase="segmentation",
                    step=step,
                    stage="first_optimizer_step_done",
                    detail=_startup_detail(startup_started_at),
                )
            elapsed = time.time() - start
            if ctx.is_main_process:
                _mark_running_status(run_dir, config, ctx, phase="segmentation", step=step, epoch=epoch)
            if ctx.is_main_process and step % config.runtime.log_every == 0:
                summary = _record_training_metrics(
                    metrics_path=run_dir / "metrics.jsonl",
                    step=step,
                    start_step=start_step,
                    target_batch=target_batch,
                    accum_steps=accum_steps,
                    elapsed=elapsed,
                    batch_metrics=batch_metrics,
                    data_time=data_time,
                    compute_time=compute_time,
                    foveator=foveator,
                    device=device,
                    extra_metrics={
                        "batch_fetch_time": batch_fetch_total,
                        "image_decode_time": image_decode_total,
                        "mask_decode_time": mask_decode_total,
                        "prompt_sample_time": prompt_sample_total,
                        "foveation_build_time": foveation_build_total,
                        "target_projection_time": target_projection_total,
                        "host_to_device_time": host_to_device_total,
                        "stack_batch_time": 0.0,
                        "valid_token_count": valid_token_total,
                    },
                )
                wandb_log(wandb_run, {f"train/{k}": v for k, v in summary.items() if k != "step"}, step=step)

            should_checkpoint = step % config.runtime.save_every == 0 or step == (config.runtime.num_steps - 1)
            checkpoint_path = None
            if should_checkpoint:
                checkpoint_path = _save_training_checkpoint(
                    run_dir,
                    config,
                    model,
                    optimizer,
                    scaler,
                    step=step,
                    ctx=ctx,
                    extra={"phase": "segmentation"},
                )
                distributed_barrier(ctx)

            should_eval = config.runtime.eval_every is not None and step > 0 and step % config.runtime.eval_every == 0
            if should_eval:
                eval_checkpoint = checkpoint_path
                if eval_checkpoint is None:
                    eval_checkpoint = run_dir / "checkpoints" / f"checkpoint_step_{step:07d}.pt"
                if ctx.is_main_process and not eval_checkpoint.exists():
                    eval_checkpoint = _save_training_checkpoint(
                        run_dir,
                        config,
                        model,
                        optimizer,
                        scaler,
                        step=step,
                        ctx=ctx,
                        extra={"phase": "segmentation", "trigger": "eval"},
                    )
                distributed_barrier(ctx)
                if eval_checkpoint.exists():
                    summary = _run_periodic_eval(config=config, checkpoint_path=eval_checkpoint, run_dir=run_dir, step=step, ctx=ctx)
                    _update_status(
                        run_dir,
                        config,
                        phase="segmentation",
                        state="evaluated",
                        step=step,
                        ctx=ctx,
                        extra_lines=[f"- eval_global_miou: `{summary.get('global_miou')}`"] if summary is not None else None,
                    )
                    if summary is not None:
                        wandb_log(wandb_run, {f"eval/{k}": v for k, v in summary.items()}, step=step)
                distributed_barrier(ctx)

        if ctx.is_main_process:
            save_checkpoint(
                run_dir / "final_model.pt",
                model,
                optimizer=optimizer,
                scaler=scaler,
                step=config.runtime.num_steps,
                config=config,
                save_optimizer_state=config.runtime.save_optimizer_state,
                save_rng_state=config.runtime.save_rng_state,
            )
            _update_status(run_dir, config, phase="segmentation", state="completed", step=config.runtime.num_steps, ctx=ctx)
            if wandb_run is not None:
                wandb_run.summary["phase"] = "segmentation"
                wandb_run.summary["completed_steps"] = config.runtime.num_steps
        distributed_barrier(ctx)
        return run_dir
    finally:
        finish_wandb_run(wandb_run)
        cleanup_distributed(ctx)


@torch.no_grad()
def run_benchmark(config: ExperimentConfig) -> dict:
    configure_torch_runtime(config.runtime)
    device = get_device(config.runtime.device)
    foveator = build_foveator(config.model).to(device)
    model = build_model(config.model.size, foveator).to(device).eval()
    model = attach_foveator_if_learnable(model, foveator)
    if config.benchmark.checkpoint:
        load_checkpoint(model, config.benchmark.checkpoint, strict=True)
    image = torch.randint(0, 255, (config.benchmark.image_size, config.benchmark.image_size, 3), dtype=torch.uint8, device=device)
    center = torch.tensor([config.benchmark.image_size // 2, config.benchmark.image_size // 2], device=device)
    tokens, valid_mask, _ = build_model_inputs(image, center, foveator)
    inputs = _normalize_tokens(tokens.unsqueeze(0).float())
    valid_mask = valid_mask.unsqueeze(0)
    for _ in range(config.benchmark.warmup_steps):
        model(inputs, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(config.benchmark.measure_steps):
        model(inputs, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return {
        "mean_latency_ms": 1000.0 * elapsed / config.benchmark.measure_steps,
        "image_size": config.benchmark.image_size,
        "model_size": config.model.size,
        "num_tokens": foveator.get_num_tokens(),
        "tokenizer_type": config.model.tokenizer_type,
        "device": str(device),
    }


def run_segmentation_train_start_smoke(config: ExperimentConfig) -> Path:
    smoke_config = replace(config)
    smoke_config.runtime = replace(config.runtime, num_steps=1, save_every=1, log_every=1, eval_every=None)
    return run_segmentation_training(smoke_config)
