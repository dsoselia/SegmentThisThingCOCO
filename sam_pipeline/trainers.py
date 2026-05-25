from __future__ import annotations

import inspect
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import ExperimentConfig
from .data import (
    MAETrainingManifestDataset,
    SegmentationTrainingManifestDataset,
    collate_mae_samples,
    collate_segmentation_samples,
)
from .evaluate import evaluate_checkpoint
from .losses import multimask_segmentation_loss
from .modeling import SamBackboneMAE, SamSegmentationModel, build_sam_model, load_backbone_into_sam, load_sam_checkpoint, normalize_image_batch
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
    infer_run_dir_from_checkpoint,
    init_distributed,
    init_wandb_run,
    load_training_checkpoint,
    save_checkpoint,
    save_run_metadata,
    seed_everything,
    seed_worker,
    set_rng_state,
    unwrap_model,
    update_run_status,
    wandb_log,
    write_json,
)


def _scheduled_batch_size(step: int, schedule: dict[str, int], default: int) -> int:
    current = default
    for key, value in sorted(((int(k), v) for k, v in schedule.items()), key=lambda item: item[0]):
        if step >= key:
            current = value
    return current


def _loader_supports(name: str) -> bool:
    return name in inspect.signature(DataLoader).parameters


def _build_loader(dataset, runtime, ctx: DistributedContext, collate_fn):
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
    if runtime.pin_memory and runtime.pin_memory_device and _loader_supports("pin_memory_device"):
        kwargs["pin_memory_device"] = runtime.pin_memory_device
    if runtime.num_workers > 0:
        kwargs["prefetch_factor"] = runtime.prefetch_factor
        if runtime.multiprocessing_context:
            kwargs["multiprocessing_context"] = runtime.multiprocessing_context
    if _loader_supports("in_order"):
        kwargs["in_order"] = runtime.dataloader_in_order
    return DataLoader(**kwargs), sampler


def _build_eval_loader(dataset, runtime, ctx: DistributedContext, collate_fn):
    if ctx.enabled:
        dataset = torch.utils.data.Subset(dataset, range(ctx.rank, len(dataset), ctx.world_size))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": runtime.micro_batch_size,
        "shuffle": False,
        "num_workers": runtime.num_workers,
        "collate_fn": collate_fn,
        "drop_last": False,
        "pin_memory": runtime.pin_memory,
        "persistent_workers": runtime.persistent_workers and runtime.num_workers > 0,
        "worker_init_fn": seed_worker,
        "timeout": runtime.dataloader_timeout_s,
    }
    if runtime.pin_memory and runtime.pin_memory_device and _loader_supports("pin_memory_device"):
        kwargs["pin_memory_device"] = runtime.pin_memory_device
    if runtime.num_workers > 0:
        kwargs["prefetch_factor"] = runtime.prefetch_factor
        if runtime.multiprocessing_context:
            kwargs["multiprocessing_context"] = runtime.multiprocessing_context
    if _loader_supports("in_order"):
        kwargs["in_order"] = runtime.dataloader_in_order
    return DataLoader(**kwargs)


def _make_run_dir(config: ExperimentConfig, prefix: str) -> Path:
    if config.runtime.resume_from:
        run_dir = infer_run_dir_from_checkpoint(config.runtime.resume_from)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        return run_dir
    return build_run_dir(config.runtime.output_dir, prefix)


def _maybe_wrap_ddp(model: torch.nn.Module, device: torch.device, ctx: DistributedContext) -> torch.nn.Module:
    if not ctx.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
    )


def _build_iterator(loader: DataLoader, sampler: DistributedSampler | None, epoch: int):
    if sampler is not None:
        sampler.set_epoch(epoch)
    return iter(loader)


def _advance_iterator(loader: DataLoader, sampler: DistributedSampler | None, iterator, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        iterator = _build_iterator(loader, sampler, epoch)
        return next(iterator), iterator, epoch


def _set_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        lr = base_lr
    else:
        lr = base_lr * min(1.0, float(step + 1) / float(warmup_steps))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _update_status(run_dir: Path, config: ExperimentConfig, *, phase: str, state: str, step: int, ctx: DistributedContext, extra_lines: list[str] | None = None) -> None:
    if not ctx.is_main_process:
        return
    lines = [
        f"# SAM Run Status",
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


def _startup_status(run_dir: Path, config: ExperimentConfig, *, phase: str, ctx: DistributedContext, device: torch.device, detail: str) -> None:
    _update_status(
        run_dir,
        config,
        phase=phase,
        state="starting",
        step=0,
        ctx=ctx,
        extra_lines=[
            f"- rank: `{ctx.rank}`",
            f"- local_rank: `{ctx.local_rank}`",
            f"- device: `{device}`",
            f"- detail: `{detail}`",
        ],
    )


def _record_training_metrics(
    metrics_path: Path,
    *,
    step: int,
    elapsed: float,
    target_batch: int,
    accum_steps: int,
    lr: float,
    batch_metrics: list[dict[str, float]],
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    summary = {}
    for key in batch_metrics[0]:
        summary[key] = sum(item[key] for item in batch_metrics) / len(batch_metrics)
    summary.update(
        {
            "step": float(step),
            "lr": float(lr),
            "target_batch": float(target_batch),
            "accum_steps": float(accum_steps),
            "seconds_per_step": float(elapsed),
            "samples_per_second": float(target_batch / max(elapsed, 1e-6)),
        }
    )
    if extra_metrics:
        summary.update(extra_metrics)
    append_jsonl(metrics_path, summary)
    return summary


def _flatten_eval_summary_for_wandb(summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    global_miou = summary.get("global_miou")
    if global_miou is not None:
        payload["eval/global_miou"] = float(global_miou)
    num_examples = summary.get("num_examples")
    if num_examples is not None:
        payload["eval/num_examples"] = float(num_examples)
    manifests = summary.get("manifests", {})
    for manifest_name, manifest_summary in manifests.items():
        manifest_miou = manifest_summary.get("global_miou")
        if manifest_miou is not None:
            payload[f"eval/{manifest_name}/global_miou"] = float(manifest_miou)
        manifest_examples = manifest_summary.get("num_examples")
        if manifest_examples is not None:
            payload[f"eval/{manifest_name}/num_examples"] = float(manifest_examples)
    return payload


@torch.no_grad()
def _evaluate_mae_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    mask_ratio: float,
    amp_dtype: torch.dtype,
    device: torch.device,
    ctx: DistributedContext,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    local_loss_sum = 0.0
    local_masked_fraction_sum = 0.0
    local_count = 0
    for batch in loader:
        images = batch.image.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            loss, metrics = model(images, mask_ratio)
        batch_count = int(images.shape[0])
        local_loss_sum += float(loss.detach().cpu()) * batch_count
        local_masked_fraction_sum += float(metrics["masked_fraction"]) * batch_count
        local_count += batch_count
    totals = torch.tensor([local_loss_sum, local_masked_fraction_sum, float(local_count)], dtype=torch.float64, device=device)
    if ctx.enabled:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    count = max(float(totals[2].item()), 1.0)
    return {
        "val/loss": float(totals[0].item() / count),
        "val/masked_fraction": float(totals[1].item() / count),
        "val/num_examples": float(totals[2].item()),
    }


def _load_resume_state(config: ExperimentConfig, model: torch.nn.Module, optimizer, scaler, ctx: DistributedContext) -> int:
    if not config.runtime.resume_from:
        return 0
    state = load_training_checkpoint(config.runtime.resume_from, model, optimizer=optimizer, scaler=scaler, strict=True)
    set_rng_state(state.get("rng_state"))
    distributed_barrier(ctx)
    return int(state.get("step", -1)) + 1


def _save_periodic_checkpoint(run_dir: Path, config: ExperimentConfig, model, optimizer, scaler, *, step: int, ctx: DistributedContext, extra: dict[str, Any] | None = None):
    if not ctx.is_main_process:
        distributed_barrier(ctx)
        return None
    checkpoint_path = run_dir / "checkpoints" / f"checkpoint_step_{step:07d}.pt"
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
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_step_*.pt"))
    if keep > 0 and len(checkpoints) > keep:
        for old_path in checkpoints[:-keep]:
            old_path.unlink(missing_ok=True)
    distributed_barrier(ctx)
    return checkpoint_path


def run_mae_pretraining(config: ExperimentConfig) -> Path:
    ctx = init_distributed(config.runtime)
    wandb_run = None
    try:
        configure_torch_runtime(config.runtime)
        seed_everything(config.runtime.seed + ctx.rank)
        device = get_device(config.runtime.device, ctx)
        amp_dtype = get_autocast_dtype(config.runtime.amp_dtype)
        run_dir = _make_run_dir(config, "mae")
        if ctx.is_main_process:
            save_run_metadata(run_dir, config, {"phase": "mae"})
            wandb_run, _ = init_wandb_run(run_dir, config, extra={"phase": "mae"})
            _startup_status(run_dir, config, phase="mae", ctx=ctx, device=device, detail="metadata_saved")
        dataset = MAETrainingManifestDataset(config.mae.train_manifest, config.model.image_size)
        loader, sampler = _build_loader(dataset, config.runtime, ctx, collate_mae_samples)
        val_loader = None
        if config.mae.val_manifest:
            val_dataset = MAETrainingManifestDataset(
                config.mae.val_manifest,
                config.model.image_size,
                max_examples=config.mae.val_max_examples,
            )
            val_loader = _build_eval_loader(val_dataset, config.runtime, ctx, collate_mae_samples)
        sam_model = build_sam_model(config.model.size, image_size=config.model.image_size, patch_size=config.model.patch_size).to(device)
        model = SamBackboneMAE(sam_model).to(device)
        model = _maybe_wrap_ddp(model, device, ctx)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.mae.lr, weight_decay=config.mae.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
        epoch = 0
        iterator = _build_iterator(loader, sampler, epoch)
        world_batch = config.runtime.micro_batch_size * ctx.world_size
        if ctx.is_main_process:
            _startup_status(run_dir, config, phase="mae", ctx=ctx, device=device, detail="model_ready")
        start_step = _load_resume_state(config, model, optimizer, scaler, ctx)
        if ctx.is_main_process:
            _startup_status(run_dir, config, phase="mae", ctx=ctx, device=device, detail=f"resume_loaded start_step={start_step}")
        _update_status(run_dir, config, phase="mae", state="running", step=start_step, ctx=ctx)

        for step in range(start_step, config.runtime.num_steps):
            lr = _set_lr(optimizer, config.mae.lr, step, config.mae.warmup_steps)
            target_batch = _scheduled_batch_size(step, config.mae.target_batch_schedule, config.mae.effective_batch_size)
            accum_steps = max(1, math.ceil(target_batch / world_batch))
            optimizer.zero_grad(set_to_none=True)
            step_start = time.time()
            batch_metrics = []
            if step == start_step and ctx.is_main_process:
                _update_status(run_dir, config, phase="mae", state="running", step=step, ctx=ctx, extra_lines=[f"- detail: `entered_first_step accum_steps={accum_steps}`"])
            for _ in range(accum_steps):
                batch, iterator, epoch = _advance_iterator(loader, sampler, iterator, epoch)
                images = batch.image.to(device, non_blocking=config.runtime.pin_memory)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    loss, metrics = model(images, config.mae.mask_ratio)
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
                batch_metrics.append(metrics)
            scaler.step(optimizer)
            scaler.update()
            if ctx.is_main_process and step % config.runtime.log_every == 0:
                summary = _record_training_metrics(
                    run_dir / "metrics.jsonl",
                    step=step,
                    elapsed=time.time() - step_start,
                    target_batch=target_batch,
                    accum_steps=accum_steps,
                    lr=lr,
                    batch_metrics=batch_metrics,
                )
                wandb_log(wandb_run, {f"train/{k}": v for k, v in summary.items() if k != "step"}, step=step)
            if val_loader is not None and config.runtime.eval_every is not None and (
                step % config.runtime.eval_every == 0 or step == config.runtime.num_steps - 1
            ):
                val_start = time.time()
                val_summary = _evaluate_mae_loss(
                    model,
                    val_loader,
                    mask_ratio=config.mae.mask_ratio,
                    amp_dtype=amp_dtype,
                    device=device,
                    ctx=ctx,
                )
                val_summary.update(
                    {
                        "step": float(step),
                        "val/seconds": float(time.time() - val_start),
                        "val/mask_ratio": float(config.mae.mask_ratio),
                    }
                )
                if ctx.is_main_process:
                    write_json(run_dir / "eval" / f"mae_val_step_{step:07d}.json", val_summary)
                    append_jsonl(run_dir / "metrics.jsonl", val_summary)
                    wandb_log(wandb_run, {k: v for k, v in val_summary.items() if k != "step"}, step=step)
                distributed_barrier(ctx)
            if step % config.runtime.save_every == 0 or step == config.runtime.num_steps - 1:
                _save_periodic_checkpoint(run_dir, config, model, optimizer, scaler, step=step, ctx=ctx, extra={"phase": "mae"})

        if ctx.is_main_process:
            save_checkpoint(
                run_dir / "final_backbone.pt",
                unwrap_model(model).backbone,
                step=config.runtime.num_steps,
                config=config,
                save_optimizer_state=False,
                save_rng_state=False,
            )
            _update_status(run_dir, config, phase="mae", state="completed", step=config.runtime.num_steps, ctx=ctx)
        distributed_barrier(ctx)
        return run_dir
    finally:
        finish_wandb_run(wandb_run)
        cleanup_distributed(ctx)


def run_segmentation_training(config: ExperimentConfig) -> Path:
    ctx = init_distributed(config.runtime)
    wandb_run = None
    try:
        configure_torch_runtime(config.runtime)
        seed_everything(config.runtime.seed + ctx.rank)
        device = get_device(config.runtime.device, ctx)
        amp_dtype = get_autocast_dtype(config.runtime.amp_dtype)
        run_dir = _make_run_dir(config, "seg")
        if ctx.is_main_process:
            save_run_metadata(run_dir, config, {"phase": "segmentation"})
            wandb_run, _ = init_wandb_run(run_dir, config, extra={"phase": "segmentation"})
            _startup_status(run_dir, config, phase="segmentation", ctx=ctx, device=device, detail="metadata_saved")
        dataset = SegmentationTrainingManifestDataset(
            config.segmentation.train_manifest,
            config.model.image_size,
            max_segments_per_image=config.segmentation.max_segments_per_image,
            seed=config.runtime.seed + ctx.rank * 100_000,
            prompt_noise_std=config.segmentation.prompt_noise_std,
            sample_multiple_segments_per_image=config.segmentation.sample_multiple_segments_per_image,
        )
        loader, sampler = _build_loader(dataset, config.runtime, ctx, collate_segmentation_samples)
        model = SamSegmentationModel(
            build_sam_model(config.model.size, image_size=config.model.image_size, patch_size=config.model.patch_size)
        ).to(device)
        if config.segmentation.init_checkpoint and not config.runtime.resume_from:
            load_training_checkpoint(config.segmentation.init_checkpoint, model, strict=True)
        elif config.segmentation.pretrained_backbone and not config.runtime.resume_from:
            load_backbone_into_sam(model, config.segmentation.pretrained_backbone)
        model = _maybe_wrap_ddp(model, device, ctx)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.segmentation.lr, weight_decay=config.segmentation.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
        epoch = 0
        iterator = _build_iterator(loader, sampler, epoch)
        segments_per_image = config.segmentation.max_segments_per_image if config.segmentation.sample_multiple_segments_per_image else 1
        world_batch = config.runtime.micro_batch_size * ctx.world_size * segments_per_image
        start_step = _load_resume_state(config, model, optimizer, scaler, ctx)
        _update_status(run_dir, config, phase="segmentation", state="running", step=start_step, ctx=ctx)

        for step in range(start_step, config.runtime.num_steps):
            lr = _set_lr(optimizer, config.segmentation.lr, step, config.segmentation.warmup_steps)
            target_batch = _scheduled_batch_size(step, config.segmentation.target_batch_schedule, config.segmentation.effective_batch_size)
            accum_steps = max(1, math.ceil(target_batch / world_batch))
            optimizer.zero_grad(set_to_none=True)
            step_start = time.time()
            batch_metrics = []
            if step == start_step and ctx.is_main_process:
                _update_status(run_dir, config, phase="segmentation", state="running", step=step, ctx=ctx, extra_lines=[f"- detail: `entered_first_step accum_steps={accum_steps}`"])
            for _ in range(accum_steps):
                batch, iterator, epoch = _advance_iterator(loader, sampler, iterator, epoch)
                images = batch.image.to(device, non_blocking=config.runtime.pin_memory)
                point_coords = batch.point_coords.view(-1, 1, 2).to(device, non_blocking=config.runtime.pin_memory)
                point_labels = batch.point_labels.to(device, non_blocking=config.runtime.pin_memory)
                target_masks = batch.target_mask.to(device, non_blocking=config.runtime.pin_memory)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    pred_masks, pred_iou = model(
                        images,
                        point_coords,
                        point_labels,
                    )
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
                scaler.scale(loss).backward()
                batch_metrics.append(metrics)
            scaler.step(optimizer)
            scaler.update()
            if ctx.is_main_process and step % config.runtime.log_every == 0:
                summary = _record_training_metrics(
                    run_dir / "metrics.jsonl",
                    step=step,
                    elapsed=time.time() - step_start,
                    target_batch=target_batch,
                    accum_steps=accum_steps,
                    lr=lr,
                    batch_metrics=batch_metrics,
                )
                wandb_log(wandb_run, {f"train/{k}": v for k, v in summary.items() if k != "step"}, step=step)
            if step % config.runtime.save_every == 0 or step == config.runtime.num_steps - 1:
                checkpoint_path = _save_periodic_checkpoint(
                    run_dir,
                    config,
                    model,
                    optimizer,
                    scaler,
                    step=step,
                    ctx=ctx,
                    extra={"phase": "segmentation"},
                )
                if checkpoint_path is None:
                    checkpoint_path = run_dir / "checkpoints" / f"checkpoint_step_{step:07d}.pt"
                if config.runtime.eval_every is not None and step % config.runtime.eval_every == 0:
                    eval_dir = run_dir / "eval" / f"step_{step:07d}"
                    eval_summary = evaluate_checkpoint(
                        config,
                        checkpoint_path=str(checkpoint_path),
                        run_dir=eval_dir,
                        summary_name=f"summary_step_{step:07d}.json",
                        enable_wandb=False,
                    )
                    if ctx.is_main_process and eval_summary:
                        wandb_log(wandb_run, _flatten_eval_summary_for_wandb(eval_summary), step=step)
                    distributed_barrier(ctx)

        if ctx.is_main_process:
            _update_status(run_dir, config, phase="segmentation", state="completed", step=config.runtime.num_steps, ctx=ctx)
        distributed_barrier(ctx)
        return run_dir
    finally:
        finish_wandb_run(wandb_run)
        cleanup_distributed(ctx)


@torch.no_grad()
def run_benchmark(config: ExperimentConfig) -> dict[str, Any]:
    device = get_device(config.runtime.device)
    model = build_sam_model(config.model.size, image_size=config.model.image_size, patch_size=config.model.patch_size).to(device).eval()
    if config.benchmark.checkpoint:
        load_sam_checkpoint(model, config.benchmark.checkpoint)
    dummy = torch.zeros((1, 3, config.model.image_size, config.model.image_size), device=device)
    point_coords = torch.tensor([[[config.model.image_size / 2, config.model.image_size / 2]]], device=device)
    point_labels = torch.ones((1, 1), device=device, dtype=torch.int64)
    for _ in range(config.benchmark.warmup_steps):
        image_embeddings = model.image_encoder(normalize_image_batch(dummy, model))
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)
        model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(config.benchmark.measure_steps):
        image_embeddings = model.image_encoder(normalize_image_batch(dummy, model))
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)
        model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "measure_steps": config.benchmark.measure_steps,
        "seconds_total": elapsed,
        "seconds_per_step": elapsed / config.benchmark.measure_steps,
    }


def run_preflight(config: ExperimentConfig) -> dict[str, Any]:
    dataset = SegmentationTrainingManifestDataset(
        config.segmentation.train_manifest,
        config.model.image_size,
        max_segments_per_image=config.segmentation.max_segments_per_image,
        sample_multiple_segments_per_image=config.segmentation.sample_multiple_segments_per_image,
    )
    sample = dataset[0]
    if isinstance(sample, list):
        sample_count = len(sample)
        first = sample[0]
    else:
        sample_count = 1
        first = sample
    return {
        "dataset_len": len(dataset),
        "sample_count_first_item": sample_count,
        "image_shape": list(first.image.shape),
        "point_shape": list(first.point_coords.shape),
        "target_shape": list(first.target_mask.shape),
    }
