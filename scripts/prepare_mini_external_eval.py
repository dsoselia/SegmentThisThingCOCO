#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

EXPECTED_COUNTS = {
    "ADE20K": 200,
    "Cityscapes": 200,
    "EgoHOS": 200,
    "VISOR": 200,
    "ZeroWaste-f": 200,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _convert_manifest(pack: Path, out_dir: Path, *, strict_mini_counts: bool) -> dict[str, Any]:
    source = pack / "manifest.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    out_jsonl = out_dir / "mini_external_eval_200x5.stt_eval.jsonl"
    out_csv = out_dir / "mini_external_eval_200x5.converted.csv"
    counts: Counter[str] = Counter()
    missing: list[str] = []
    empty_masks: list[str] = []
    rows: list[dict[str, Any]] = []

    with source.open() as handle, out_jsonl.open("w") as jout:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = (pack / row["image_relpath"]).resolve()
            mask_path = (pack / row["mask_relpath"]).resolve()
            if not image_path.exists():
                missing.append(str(image_path))
            if not mask_path.exists():
                missing.append(str(mask_path))
            else:
                mask = np.array(Image.open(mask_path).convert("L")) > 0
                if int(mask.sum()) <= 0:
                    empty_masks.append(str(mask_path))
            dataset = str(row["dataset"])
            prompt_x = int(row["prompt_x"])
            prompt_y = int(row["prompt_y"])
            counts[dataset] += 1
            converted = {
                "image_path": str(image_path),
                "dataset_name": dataset,
                "source_id": row.get("id"),
                "segments": [
                    {
                        "mask_path": str(mask_path),
                        "center": [prompt_x, prompt_y],
                        "prompt_x": prompt_x,
                        "prompt_y": prompt_y,
                        "source_id": row.get("id"),
                        "area_pixels": int(row.get("area_pixels", 0)),
                        "bbox_xyxy": row.get("bbox_xyxy"),
                    }
                ],
            }
            rows.append(
                {
                    "dataset_name": dataset,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "prompt_x": prompt_x,
                    "prompt_y": prompt_y,
                    "source_id": row.get("id"),
                }
            )
            jout.write(json.dumps(converted, sort_keys=True) + "\n")

    if rows:
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "source_manifest": str(source),
        "converted_manifest": str(out_jsonl),
        "converted_csv": str(out_csv),
        "total_records": len(rows),
        "counts": dict(sorted(counts.items())),
        "missing_files": missing[:20],
        "missing_count": len(missing),
        "empty_masks": empty_masks[:20],
        "empty_mask_count": len(empty_masks),
        "prompt_policy": "uses explicit prompt_x/prompt_y from mini_external_eval_200x5 as segment.center",
    }
    if strict_mini_counts:
        if len(rows) != 1000:
            raise RuntimeError(f"Expected 1000 records, got {len(rows)}")
        if dict(sorted(counts.items())) != EXPECTED_COUNTS:
            raise RuntimeError(f"Unexpected dataset counts: {dict(sorted(counts.items()))}")
    if missing:
        raise RuntimeError(f"Missing files: {missing[:5]}")
    if empty_masks:
        raise RuntimeError(f"Empty masks: {empty_masks[:5]}")
    _write_json(out_dir / "conversion_summary.json", summary)
    return summary


def _prepare_config(
    *,
    base_config: Path,
    out_dir: Path,
    manifest: Path,
    checkpoint: Path,
    run_name: str,
    profile_prefix: str,
    mode: str,
    max_examples: int | None,
    wandb_project: str,
    wandb_tags: list[str],
) -> Path:
    config = _read_json(base_config)
    config["runtime"].update(
        {
            "output_dir": str(out_dir / ("eval_runs_smoke" if mode == "smoke" else "eval_runs_full")),
            "profile_name": f"{profile_prefix}_{mode}",
            "device": "cuda",
            "distributed": False,
            "micro_batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "eval_every": None,
            "resume_from": None,
            "fork_from": None,
            "wandb_enabled": mode == "full",
            "wandb_project": wandb_project,
            "wandb_mode": "offline",
            "wandb_dir": str(out_dir / "wandb"),
            "wandb_run_name": run_name if mode == "full" else f"{run_name}-smoke",
            "wandb_tags": wandb_tags,
        }
    )
    config["evaluation"] = {
        "named_eval_manifests": {"mini_external_eval_200x5": str(manifest)},
        "max_examples": max_examples,
        "threshold": 0.5,
        "upsample_small_images": True,
    }
    config["assumptions"] = [
        "External evaluation only; writes under isolated artifact directory and does not alter active training runs.",
        f"Evaluates checkpoint {checkpoint}.",
        "Uses explicit prompt_x/prompt_y from mini_external_eval_200x5 as the eval center.",
        "Full-image mIoU, threshold 0.5, upsample_small_images=true.",
    ]
    path = out_dir / f"external_eval_mini_200x5_{mode}_config.json"
    _write_json(path, config)
    return path


def main() -> None:
    parser = argparse.ArgumentParser("Prepare mini_external_eval_200x5 STT eval artifacts")
    parser.add_argument("--pack", required=True, type=Path, help="mini_external_eval_200x5 pack directory")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile-prefix", default="mini_external_eval_200x5")
    parser.add_argument("--source-run", default=None)
    parser.add_argument("--selected-step", type=int, default=None)
    parser.add_argument("--selection-policy", default="manual checkpoint selection")
    parser.add_argument("--internal-miou", type=float, default=None)
    parser.add_argument("--no-copy-checkpoint", action="store_true")
    parser.add_argument("--no-strict-mini-counts", action="store_true")
    parser.add_argument("--wandb-project", default="SegmentThisThingSA1BExternalEval")
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=["stt", "sa1b", "external-eval", "mini-200x5", "logrect", "fixed-lambda", "h100", "offline"],
    )
    args = parser.parse_args()

    pack = args.pack.resolve()
    out_dir = args.artifact_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    copied_checkpoint = checkpoint if args.no_copy_checkpoint else out_dir / checkpoint.name
    if copied_checkpoint != checkpoint:
        shutil.copy2(checkpoint, copied_checkpoint)

    conversion = _convert_manifest(pack, out_dir, strict_mini_counts=not args.no_strict_mini_counts)
    manifest = Path(conversion["converted_manifest"])
    smoke_config = _prepare_config(
        base_config=args.base_config,
        out_dir=out_dir,
        manifest=manifest,
        checkpoint=copied_checkpoint,
        run_name=args.run_name,
        profile_prefix=args.profile_prefix,
        mode="smoke",
        max_examples=5,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
    )
    full_config = _prepare_config(
        base_config=args.base_config,
        out_dir=out_dir,
        manifest=manifest,
        checkpoint=copied_checkpoint,
        run_name=args.run_name,
        profile_prefix=args.profile_prefix,
        mode="full",
        max_examples=None,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
    )
    selection = {
        "artifact_dir": str(out_dir),
        "selected_step": args.selected_step,
        "source_checkpoint": str(checkpoint),
        "copied_checkpoint": str(copied_checkpoint),
        "selection_policy": args.selection_policy,
        "internal_sa1b_val_miou": args.internal_miou,
        "source_run": args.source_run,
    }
    _write_json(out_dir / "selection.json", selection)
    print(
        json.dumps(
            {
                "artifact_dir": str(out_dir),
                "checkpoint": str(copied_checkpoint),
                "smoke_config": str(smoke_config),
                "full_config": str(full_config),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
