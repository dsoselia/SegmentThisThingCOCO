from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .config import ExperimentConfig
from .data import EvalManifestDataset, prepare_image_mask_point, resolve_eval_manifests
from .modeling import build_sam_model, load_backbone_into_sam, load_sam_checkpoint, normalize_image_batch
from .runtime import DistributedContext
from .runtime import build_run_dir, finish_wandb_run, get_device, init_wandb_run, save_run_metadata, wandb_log, write_json


def _dist_context() -> tuple[bool, int, int, int]:
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), int(os.environ.get("LOCAL_RANK", dist.get_rank())), dist.get_world_size()
    return False, 0, 0, 1


@torch.no_grad()
def _evaluate_manifest(config: ExperimentConfig, model: torch.nn.Module, manifest_path: str, device: torch.device) -> dict[str, Any] | None:
    dataset = EvalManifestDataset(manifest_path, config.evaluation.max_examples)
    dist_enabled, rank, _, world_size = _dist_context()
    dataset_names = sorted({entry["dataset_name"] for entry in dataset.entries})
    local_sums = {name: 0.0 for name in dataset_names}
    local_counts = {name: 0 for name in dataset_names}

    for index in range(rank, len(dataset), world_size):
        sample = dataset[index]
        image, point_coords, _, input_size = prepare_image_mask_point(
            sample.image,
            sample.mask.float(),
            sample.center,
            image_size=config.model.image_size,
            upsample_small_images=config.evaluation.upsample_small_images,
        )
        image = image.unsqueeze(0).to(device)
        point_coords = point_coords.view(1, 1, 2).to(device)
        point_labels = torch.ones((1, 1), device=device, dtype=torch.int64)
        image_embeddings = model.image_encoder(normalize_image_batch(image, model))
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, pred_iou = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        best_idx = pred_iou.squeeze(0).argmax()
        masks = model.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=sample.original_size,
        )
        pred = masks.squeeze(0)[best_idx] > config.evaluation.threshold
        target = sample.mask.bool()
        inter = (pred.cpu() & target).sum().item()
        union = (pred.cpu() | target).sum().item()
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
    summary["global_miou"] = (
        sum(summary["per_dataset_miou"].values()) / len(summary["per_dataset_miou"])
        if summary["per_dataset_miou"]
        else None
    )
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
    dist_enabled, rank, local_rank, _ = _dist_context()
    if not dist_enabled or rank == 0:
        save_run_metadata(out_dir, config, {"checkpoint_path": checkpoint_path})
    wandb_run = None
    if enable_wandb and (not dist_enabled or rank == 0):
        wandb_run, _ = init_wandb_run(out_dir, config, extra={"phase": "evaluation", "checkpoint_path": checkpoint_path})
    try:
        device = get_device(
            config.runtime.device,
            DistributedContext(enabled=dist_enabled, rank=rank, local_rank=local_rank, world_size=1 if not dist_enabled else int(os.environ.get("WORLD_SIZE", "1"))),
        )
        model = build_sam_model(config.model.size, image_size=config.model.image_size, patch_size=config.model.patch_size).to(device).eval()
        if checkpoint_path is not None:
            load_sam_checkpoint(model, checkpoint_path)
        elif config.segmentation.pretrained_backbone:
            load_backbone_into_sam(model, config.segmentation.pretrained_backbone)
        manifest_summaries = {}
        skipped_manifests = {}
        for manifest_name, manifest_path in resolve_eval_manifests(config.evaluation):
            if not Path(manifest_path).exists():
                if not dist_enabled or rank == 0:
                    skipped_manifests[manifest_name] = manifest_path
                continue
            summary = _evaluate_manifest(config, model, manifest_path, device)
            if summary is not None:
                manifest_summaries[manifest_name] = summary
        if dist_enabled and rank != 0:
            return {}
        summary = {
            "checkpoint_path": checkpoint_path,
            "manifests": manifest_summaries,
            "skipped_manifests": skipped_manifests,
        }
        available = [item["global_miou"] for item in manifest_summaries.values() if item["global_miou"] is not None]
        summary["global_miou"] = sum(available) / len(available) if available else None
        summary["num_examples"] = sum(item["num_examples"] for item in manifest_summaries.values())
        write_json(out_dir / summary_name, summary)
        wandb_log(wandb_run, {f"eval/{k}": v for k, v in summary.items() if not isinstance(v, dict)}, step=0)
        return summary
    finally:
        finish_wandb_run(wandb_run)
