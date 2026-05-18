from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .config import ExperimentConfig
from .data import EvalManifestDataset, resolve_eval_manifests
from .modeling import attach_foveator_if_learnable, build_foveator, build_model, load_checkpoint
from .runtime import build_run_dir, finish_wandb_run, get_device, init_wandb_run, save_run_metadata, wandb_log, write_json
from .transforms import build_model_inputs, maybe_resize_small_image, reconstruct_logits_to_image


def _normalize_tokens(tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    return (tokens / 255.0 - mean) / std


def _dist_context() -> tuple[bool, int, int]:
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


@torch.no_grad()
def _evaluate_manifest(config: ExperimentConfig, model: torch.nn.Module, foveator, manifest_path: str, device: torch.device) -> dict[str, Any] | None:
    dataset = EvalManifestDataset(manifest_path, config.evaluation.max_examples)
    dist_enabled, rank, world_size = _dist_context()
    dataset_names = sorted({entry["dataset_name"] for entry in dataset.entries})
    local_sums = {name: 0.0 for name in dataset_names}
    local_counts = {name: 0 for name in dataset_names}

    for index in range(rank, len(dataset), world_size):
        sample = dataset[index]
        image = sample.image
        if config.evaluation.upsample_small_images:
            image = maybe_resize_small_image(image, foveator.get_pattern_bounds_size())
            scale_x = image.shape[1] / sample.image.shape[1]
            scale_y = image.shape[0] / sample.image.shape[0]
            center = torch.tensor(
                [int(round(sample.center[0].item() * scale_x)), int(round(sample.center[1].item() * scale_y))],
                dtype=torch.int64,
            )
            mask = torch.nn.functional.interpolate(
                sample.mask.float().unsqueeze(0).unsqueeze(0),
                size=image.shape[:2],
                mode="nearest",
            ).squeeze(0).squeeze(0) > 0.5
        else:
            center = sample.center
            mask = sample.mask

        image = image.to(device)
        center = center.to(device)
        tokens, valid_mask, crop_bounds = build_model_inputs(image, center, foveator)
        norm = _normalize_tokens(tokens.unsqueeze(0).float(), device)
        pred_masks, pred_iou = model(norm, valid_mask.unsqueeze(0))
        best_idx = pred_iou.squeeze(0).argmax()
        recon = reconstruct_logits_to_image(
            foveator,
            pred_masks.squeeze(0)[best_idx].cpu(),
            crop_bounds.cpu(),
            tuple(image.shape[:2]),
        )
        pred = recon.sigmoid() > config.evaluation.threshold
        inter = (pred & mask.cpu().bool()).sum().item()
        union = (pred | mask.cpu().bool()).sum().item()
        iou = 0.0 if union == 0 else inter / union
        local_sums[sample.dataset_name] += iou
        local_counts[sample.dataset_name] += 1

    if dist_enabled:
        sum_tensor = torch.tensor([local_sums[name] for name in dataset_names], dtype=torch.float64, device=device)
        count_tensor = torch.tensor([local_counts[name] for name in dataset_names], dtype=torch.float64, device=device)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        global_sums = {name: float(sum_tensor[idx].item()) for idx, name in enumerate(dataset_names)}
        global_counts = {name: int(round(count_tensor[idx].item())) for idx, name in enumerate(dataset_names)}
    else:
        global_sums = local_sums
        global_counts = local_counts

    if dist_enabled and rank != 0:
        return None

    summary: dict[str, Any] = {
        "manifest_path": manifest_path,
        "per_dataset_miou": {
            name: (global_sums[name] / global_counts[name]) for name in dataset_names if global_counts[name] > 0
        },
        "num_examples": sum(global_counts.values()),
    }
    if summary["per_dataset_miou"]:
        summary["global_miou"] = sum(summary["per_dataset_miou"].values()) / len(summary["per_dataset_miou"])
    else:
        summary["global_miou"] = None
    return summary


@torch.no_grad()
def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | None = None,
    *,
    run_dir: str | Path | None = None,
    summary_name: str = "summary.json",
    enable_wandb: bool = True,
) -> dict[str, Any]:
    out_dir = build_run_dir(config.runtime.output_dir, "eval") if run_dir is None else Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_enabled, rank, _ = _dist_context()
    if not dist_enabled or rank == 0:
        save_run_metadata(out_dir, config, {"checkpoint_path": checkpoint_path})
    wandb_run = None
    if enable_wandb and (not dist_enabled or rank == 0):
        wandb_run, _ = init_wandb_run(out_dir, config, extra={"phase": "evaluation", "checkpoint_path": checkpoint_path})

    try:
        device = get_device(config.runtime.device)
        foveator = build_foveator(config.model).to(device)
        model = build_model(config.model.size, foveator)
        model = attach_foveator_if_learnable(model, foveator)
        if checkpoint_path is not None:
            load_checkpoint(model, checkpoint_path, strict=True)
        model = model.to(device).eval()

        manifest_summaries = {}
        skipped_manifests = {}
        for manifest_name, manifest_path in resolve_eval_manifests(config.evaluation):
            if not Path(manifest_path).exists():
                if not dist_enabled or rank == 0:
                    skipped_manifests[manifest_name] = manifest_path
                continue
            summary = _evaluate_manifest(config, model, foveator, manifest_path, device)
            if summary is not None:
                manifest_summaries[manifest_name] = summary

        if dist_enabled and rank != 0:
            return {}

        summary: dict[str, Any] = {
            "checkpoint_path": checkpoint_path,
            "manifests": manifest_summaries,
            "skipped_manifests": skipped_manifests,
        }
        available = [item["global_miou"] for item in manifest_summaries.values() if item["global_miou"] is not None]
        summary["global_miou"] = sum(available) / len(available) if available else None
        summary["num_examples"] = sum(item["num_examples"] for item in manifest_summaries.values())
        write_json(out_dir / summary_name, summary)
        wandb_log(wandb_run, {f"eval/{k}": v for k, v in summary.items() if not isinstance(v, dict)}, step=0)
        if wandb_run is not None:
            wandb_run.summary["phase"] = "evaluation"
            wandb_run.summary["global_miou"] = summary["global_miou"]
        return summary
    finally:
        finish_wandb_run(wandb_run)
