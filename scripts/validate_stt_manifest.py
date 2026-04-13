#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline.config import ModelConfig
from stt_pipeline.data import (
    MAETrainingManifestDataset,
    SegmentationTrainingManifestDataset,
    read_jsonl,
)


def _check_image_paths(entries: list[dict], sample_size: int, rng: random.Random) -> None:
    sample = entries if len(entries) <= sample_size else rng.sample(entries, sample_size)
    for entry in sample:
        image_path = Path(entry["image_path"])
        if not image_path.is_absolute():
            raise ValueError(f"Manifest image path is not absolute: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image path does not exist: {image_path}")


def _check_segmentation_entries(entries: list[dict], sample_size: int, rng: random.Random) -> None:
    sample = entries if len(entries) <= sample_size else rng.sample(entries, sample_size)
    for entry in sample:
        segments = entry.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"Segmentation manifest entry has no segments: {entry['image_path']}")


def main() -> None:
    parser = argparse.ArgumentParser("Validate STT JSONL manifests against the current data pipeline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kind", choices=["mae", "seg"], required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--margin", type=int, default=256)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    entries = read_jsonl(args.manifest)
    if not entries:
        raise ValueError("Manifest is empty.")

    _check_image_paths(entries, args.sample_size, rng)

    if args.kind == "mae":
        dataset = MAETrainingManifestDataset(args.manifest, margin=args.margin, seed=args.seed)
        sample = dataset[0]
        summary = {
            "kind": "mae",
            "entries": len(entries),
            "dataset_name": entries[0].get("dataset_name"),
            "sample_image_shape": list(sample.image.shape),
            "sample_center": sample.center.tolist(),
        }
    else:
        _check_segmentation_entries(entries, args.sample_size, rng)
        dataset = SegmentationTrainingManifestDataset(
            args.manifest,
            margin=args.margin,
            model_config=ModelConfig(),
            seed=args.seed,
        )
        sample = dataset[0]
        if isinstance(sample, list):
            sample = sample[0]
        summary = {
            "kind": "seg",
            "entries": len(entries),
            "dataset_name": entries[0].get("dataset_name"),
            "sample_image_shape": list(sample.image.shape),
            "sample_mask_shape": list(sample.mask.shape),
            "sample_center": sample.center.tolist(),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
