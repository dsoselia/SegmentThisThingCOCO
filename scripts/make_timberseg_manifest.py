#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from make_instance_manifest import build_manifest_from_coco


def _resolve_default_annotations(root: Path, split: str) -> Path:
    candidates = [
        root / "annotations" / f"{split}.json",
        root / "annotations" / f"instances_{split}.json",
        root / f"{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find TimberSeg annotations for split '{split}' under {root}")


def _resolve_default_images_dir(root: Path, split: str) -> Path:
    candidates = [
        root / "images" / split,
        root / split / "images",
        root / split,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find TimberSeg images for split '{split}' under {root}")


def main() -> None:
    parser = argparse.ArgumentParser("Create an STT eval manifest for TimberSeg.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--min-area", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(args.root)
    annotations_path = Path(args.annotations) if args.annotations else _resolve_default_annotations(root, args.split)
    images_dir = Path(args.images_dir) if args.images_dir else _resolve_default_images_dir(root, args.split)

    build_manifest_from_coco(
        annotations_path=annotations_path,
        images_dir=images_dir,
        output_path=Path(args.output),
        dataset_name="timberseg",
        split_name=args.split,
        min_area=args.min_area,
    )


if __name__ == "__main__":
    main()
