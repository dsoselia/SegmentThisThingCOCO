#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        and "_mask" not in path.stem.lower()
    )


def build_image_manifest(images_dir: Path, output_path: Path, dataset_name: str, split_name: str | None) -> None:
    images = _iter_images(images_dir)
    if not images:
        raise ValueError(f"No image files found in {images_dir}")

    dataset_value = dataset_name if split_name is None else f"{dataset_name}:{split_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for image_path in images:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path.resolve()),
                        "dataset_name": dataset_value,
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser("Create an image-only JSONL manifest for STT MAE pretraining.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", default=None)
    args = parser.parse_args()

    build_image_manifest(
        images_dir=Path(args.images_dir),
        output_path=Path(args.output),
        dataset_name=args.dataset_name,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()
