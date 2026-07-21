#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _optional_int(value: str | None) -> int | None:
    return None if value in {None, ""} else int(value)


def _optional_path(value: str | None) -> str | None:
    return None if value in {None, ""} else value


def _batch_schedule(
    explicit_schedule: str | None,
    *,
    effective_batch: int,
    num_steps: int,
    batch_double_every: str | None,
) -> dict[str, int]:
    if explicit_schedule:
        schedule = json.loads(explicit_schedule)
        if not isinstance(schedule, dict) or not schedule:
            raise ValueError("target batch schedule must be a non-empty JSON object")
        return {str(int(step)): int(batch) for step, batch in schedule.items()}
    if batch_double_every:
        interval = int(batch_double_every)
        if interval <= 0:
            return {"0": effective_batch}
        schedule: dict[str, int] = {}
        batch = effective_batch
        for step in range(0, num_steps, interval):
            schedule[str(step)] = batch
            batch *= 2
        return schedule
    return {"0": effective_batch}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.resume_from and args.fork_from:
        raise ValueError("resume_from and fork_from are mutually exclusive")
    initialization_sources = [
        value
        for value in (args.pretrained_encoder, args.pretrained_mae_checkpoint, args.init_checkpoint)
        if value
    ]
    if len(initialization_sources) > 1:
        raise ValueError("segmentation initialization options are mutually exclusive")
    config = json.loads(args.base.read_text())
    root = args.experiment_root
    milestones = list(range(args.milestone_every, args.num_steps + 1, args.milestone_every)) if args.milestone_every > 0 else []
    if args.num_steps not in milestones:
        milestones.append(args.num_steps)
    run_name = args.wandb_run_name or f"beacon-ddp-{args.profile}"
    base_tags = [
        tag for tag in config["runtime"].get("wandb_tags", []) if tag.lower() not in {"a100", "h100", "h200"}
    ]

    config["runtime"].update(
        {
            "output_dir": str(root / "runs" / args.profile),
            "profile_name": args.profile,
            "distributed": True,
            "micro_batch_size": args.micro_batch,
            "num_workers": args.workers,
            "prefetch_factor": args.prefetch,
            "dataloader_in_order": args.in_order,
            "num_steps": args.num_steps,
            "save_every": args.save_every,
            "milestone_steps": milestones,
            "log_every": args.log_every,
            "eval_every": _optional_int(args.val_every) if args.task == "seg" else None,
            "resume_from": _optional_path(args.resume_from),
            "fork_from": _optional_path(args.fork_from),
            "checkpoint_keep_last": args.checkpoint_keep_last,
            "wandb_project": args.wandb_project or config["runtime"].get("wandb_project", "SegmentThisThingSA1BLogRect"),
            "wandb_run_name": run_name,
            "wandb_mode": args.wandb_mode or config["runtime"].get("wandb_mode", "offline"),
            "wandb_tags": sorted(
                set(
                    base_tags
                    + [
                        args.task,
                        "ddp",
                        "sa1b",
                        args.gpus_tag,
                        "fixed-lambda",
                        "beacon",
                        args.gpu_type,
                        f"ranks-{args.nproc}",
                        f"mb-per-rank-{args.micro_batch}",
                        f"workers-per-rank-{args.workers}",
                        f"prefetch-{args.prefetch}",
                        f"in-order-{int(args.in_order)}",
                        f"worker-precrop-{int(args.worker_pre_crop)}",
                        f"steps-{args.num_steps}",
                    ]
                )
            ),
        }
    )

    schedule = _batch_schedule(
        args.target_batch_schedule,
        effective_batch=args.effective_batch,
        num_steps=args.num_steps,
        batch_double_every=args.batch_double_every,
    )
    staged = args.staged_manifest_root
    if args.task == "mae":
        mae = config["mae"]
        mae["train_manifest"] = str(staged / "sa1b_subset_mae_train_images.jsonl")
        mae["val_manifest"] = str(staged / "sa1b_subset_mae_val_images.jsonl")
        mae["val_every"] = _optional_int(args.val_every)
        if args.val_max_examples:
            mae["val_max_examples"] = int(args.val_max_examples)
        mae["effective_batch_size"] = args.effective_batch
        mae["target_batch_schedule"] = schedule
        if args.warmup_steps:
            mae["warmup_steps"] = int(args.warmup_steps)
        if args.views_per_image:
            mae["views_per_image"] = int(args.views_per_image)
        mae["worker_pre_crop"] = args.worker_pre_crop
    else:
        segmentation = config["segmentation"]
        segmentation["train_manifest"] = str(staged / "sa1b_subset_train.jsonl")
        segmentation["effective_batch_size"] = args.effective_batch
        segmentation["target_batch_schedule"] = schedule
        if args.warmup_steps:
            segmentation["warmup_steps"] = int(args.warmup_steps)
        if args.max_segments_per_image:
            segmentation["max_segments_per_image"] = int(args.max_segments_per_image)
        if initialization_sources:
            segmentation.update(
                {
                    "pretrained_encoder": _optional_path(args.pretrained_encoder),
                    "pretrained_mae_checkpoint": _optional_path(args.pretrained_mae_checkpoint),
                    "init_checkpoint": _optional_path(args.init_checkpoint),
                }
            )
        config["evaluation"]["named_eval_manifests"]["sa1b_subset_val"] = str(staged / "sa1b_subset_val.jsonl")
        if args.val_max_examples:
            config["evaluation"]["max_examples"] = int(args.val_max_examples)

    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a task-specific SA-1B DDP config from a stable base config.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("mae", "seg"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--staged-manifest-root", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefetch", type=int, required=True)
    parser.add_argument("--in-order", type=_parse_bool, required=True)
    parser.add_argument("--worker-pre-crop", type=_parse_bool, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--milestone-every", type=int, required=True)
    parser.add_argument("--log-every", type=int, required=True)
    parser.add_argument("--checkpoint-keep-last", type=int, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--gpus-tag", required=True)
    parser.add_argument("--gpu-type", default="h100")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--fork-from", default="")
    parser.add_argument("--val-every", default="")
    parser.add_argument("--val-max-examples", default="")
    parser.add_argument("--target-batch-schedule", default="")
    parser.add_argument("--batch-double-every", default="")
    parser.add_argument("--warmup-steps", default="")
    parser.add_argument("--views-per-image", default="")
    parser.add_argument("--max-segments-per-image", default="")
    parser.add_argument("--pretrained-encoder", default="")
    parser.add_argument("--pretrained-mae-checkpoint", default="")
    parser.add_argument("--init-checkpoint", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = build_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
