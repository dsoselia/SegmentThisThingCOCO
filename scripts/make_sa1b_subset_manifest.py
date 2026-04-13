#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_entry(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text())
    image_path = json_path.with_suffix(".jpg")
    segments = []
    for ann in payload.get("annotations", []):
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


def main() -> None:
    parser = argparse.ArgumentParser("Create SA-1B subset train/val manifests from extracted shard directories.")
    parser.add_argument("--root", required=True, help="Root directory containing extracted shard subdirectories.")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--max-images", type=int, default=6000)
    parser.add_argument("--val-images", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    root = Path(args.root)
    json_files = sorted(root.glob("sa_*/*.json"))
    rng = random.Random(args.seed)
    rng.shuffle(json_files)

    selected = json_files[: args.max_images + args.val_images]
    val_set = set(selected[: args.val_images])
    train_set = selected[args.val_images :]

    train_entries = []
    val_entries = []
    for path in selected:
        entry = build_entry(path)
        if not entry:
            continue
        if path in val_set:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.train_out).write_text("".join(json.dumps(x) + "\n" for x in train_entries))
    Path(args.val_out).write_text("".join(json.dumps(x) + "\n" for x in val_entries))

    print(json.dumps({"train_images": len(train_entries), "val_images": len(val_entries)}, indent=2))


if __name__ == "__main__":
    main()
