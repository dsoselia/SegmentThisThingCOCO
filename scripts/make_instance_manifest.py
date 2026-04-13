#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_coco_annotations(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_segmentation(annotation: dict[str, Any]) -> dict[str, Any] | None:
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        return {"polygon": segmentation}
    if isinstance(segmentation, dict):
        return {"rle": segmentation}
    return None


def build_manifest_from_coco(
    *,
    annotations_path: Path,
    images_dir: Path,
    output_path: Path,
    dataset_name: str,
    split_name: str | None,
    min_area: float,
) -> None:
    payload = _load_coco_annotations(annotations_path)
    image_entries = {item["id"]: item for item in payload["images"]}
    grouped = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        if annotation.get("area", 0.0) < min_area:
            continue
        segmentation = _normalize_segmentation(annotation)
        if segmentation is None:
            continue
        grouped[annotation["image_id"]].append(segmentation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for image_id in sorted(grouped):
            segments = grouped[image_id]
            image_entry = image_entries[image_id]
            image_path = images_dir / image_entry["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image referenced by annotations: {image_path}")
            record = {
                "image_path": str(image_path.resolve()),
                "dataset_name": dataset_name if split_name is None else f"{dataset_name}:{split_name}",
                "segments": segments,
            }
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser("Convert COCO-style instance annotations into STT JSONL manifests.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--min-area", type=float, default=1.0)
    args = parser.parse_args()

    build_manifest_from_coco(
        annotations_path=Path(args.annotations),
        images_dir=Path(args.images_dir),
        output_path=Path(args.output),
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        min_area=args.min_area,
    )


if __name__ == "__main__":
    main()
