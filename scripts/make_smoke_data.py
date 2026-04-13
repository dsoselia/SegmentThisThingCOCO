#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_example(root: Path, name: str, offset: int) -> dict:
    height = width = 1536
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((xx / max(width - 1, 1)) * 255).astype(np.uint8)
    image[..., 1] = ((yy / max(height - 1, 1)) * 255).astype(np.uint8)
    image[..., 2] = 64

    circle = ((xx - (768 + offset)) ** 2 + (yy - 768) ** 2) < 200**2
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[circle] = 255
    image[circle] = np.array([255, 220, 40], dtype=np.uint8)

    image_path = root / f"{name}.png"
    mask_path = root / f"{name}_mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    return {
        "image_path": str(image_path.resolve()),
        "dataset_name": "smoke",
        "segments": [{"mask_path": str(mask_path.resolve())}],
    }


def main() -> None:
    root = Path("smoke_data")
    manifests = Path("manifests")
    root.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    train_entries = [_write_example(root, "train_0", 0), _write_example(root, "train_1", -180)]
    eval_entries = [_write_example(root, "eval_0", 120), _write_example(root, "eval_1", -120)]

    (manifests / "train_smoke.jsonl").write_text("".join(json.dumps(x) + "\n" for x in train_entries))
    (manifests / "eval_smoke.jsonl").write_text("".join(json.dumps(x) + "\n" for x in eval_entries))


if __name__ == "__main__":
    main()
