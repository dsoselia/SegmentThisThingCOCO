#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_entry(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    image_path = json_path.with_suffix(".jpg")
    if not image_path.exists():
        return {}
    segments = []
    for ann in payload.get("annotations", []):
        if "segmentation" not in ann:
            continue
        segments.append(
            {
                "rle": ann["segmentation"],
                "point_coords": ann.get("point_coords"),
                "area": ann.get("area"),
                "bbox": ann.get("bbox"),
                "predicted_iou": ann.get("predicted_iou"),
                "stability_score": ann.get("stability_score"),
            }
        )
    if not segments:
        return {}
    return {
        "image_path": str(image_path.resolve()),
        "dataset_name": "sa1b_subset",
        "segments": segments,
    }


def _discover_json_files(root: Path) -> list[Path]:
    flat_files = sorted(root.glob("sa_*.json"))
    sharded_files = sorted(root.glob("sa_*/*.json"))
    return sorted({*flat_files, *sharded_files})


def main() -> None:
    parser = argparse.ArgumentParser("Create SA-1B subset train/val manifests from extracted shard directories.")
    parser.add_argument("--root", required=True, help="Root directory containing extracted shard subdirectories.")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--max-images", type=int, default=None, help="Maximum train images. Omit or use <=0 for all.")
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    root = Path(args.root)
    json_files = _discover_json_files(root)
    rng = random.Random(args.seed)
    rng.shuffle(json_files)

    if args.max_images is None or args.max_images <= 0:
        selected = json_files
    else:
        selected = json_files[: args.max_images + args.val_images]
    val_count = min(args.val_images, max(0, len(selected) - 1))
    val_set = set(selected[:val_count])

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_out).parent.mkdir(parents=True, exist_ok=True)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    train_tmp = train_out.with_suffix(train_out.suffix + ".tmp")
    val_tmp = val_out.with_suffix(val_out.suffix + ".tmp")

    train_count = 0
    val_count_actual = 0
    with train_tmp.open("w") as train_handle, val_tmp.open("w") as val_handle:
        for path in selected:
            entry = build_entry(path)
            if not entry:
                continue
            if path in val_set:
                val_handle.write(json.dumps(entry) + "\n")
                val_count_actual += 1
            else:
                train_handle.write(json.dumps(entry) + "\n")
                train_count += 1

    train_tmp.replace(train_out)
    val_tmp.replace(val_out)

    print(
        json.dumps(
            {
                "discovered_json": len(json_files),
                "selected_json": len(selected),
                "train_images": train_count,
                "val_images": val_count_actual,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
