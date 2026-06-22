"""
Visualize LogRect2 reconstruction for different lambda_scale (c) values.

Exponent is fixed at the paper default (4.0); only the lambda scaler varies.
Produces a 3-panel composite per image: Original | c=0.6 | c=1.0.

Usage:
  python scripts/visualize_logrect_lambda.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from segment_this_thing.foveation import LogRectilinearFoveator

COCO_VAL = "/fs/cml-datasets/coco/images/val2017"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "lambda_comparison")
LAMBDAS = [0.6, 1.0]          # c values to compare
EXPONENT = 4.0                # fixed paper default
PATTERN_SIZE = 1280
AXIS_BINS = 13
TOKEN_SIZE = 16
N_IMAGES = 10


def compute_bin_widths(c: float) -> list[int]:
    crop_half = PATTERN_SIZE / 2.0
    buffer_half = (AXIS_BINS * TOKEN_SIZE) / 2.0
    lam = c * crop_half / (math.e - 1.0)
    edges = []
    for i in range(AXIS_BINS + 1):
        du = i * TOKEN_SIZE - buffer_half
        ad = abs(du)
        if ad < 1e-9:
            dx = 0.0
        else:
            exp_term = lam * (math.exp((ad / buffer_half) ** EXPONENT) - 1.0)
            dx = max(ad, exp_term) * (1.0 if du > 0 else -1.0)
        edges.append(int(math.floor(crop_half + dx)))
    edges[0] = 0
    edges[-1] = PATTERN_SIZE
    return [edges[i + 1] - edges[i] for i in range(AXIS_BINS)]


def center_crop_1280(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if min(img.size) < PATTERN_SIZE:
        img = TF.resize(img, PATTERN_SIZE)
    img = TF.center_crop(img, [PATTERN_SIZE, PATTERN_SIZE])
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)  # uint8 CHW


def add_label(img_np: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, PATTERN_SIZE, 60], fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Lambda (c) values and bin widths (exponent={EXPONENT}, outer -> center):")
    for c in LAMBDAS:
        tag = "  (paper default)" if c == 1.0 else ""
        print(f"  c={c}: {compute_bin_widths(c)}{tag}")
    print()

    foveators = {
        c: LogRectilinearFoveator(
            token_size=TOKEN_SIZE,
            pattern_size=PATTERN_SIZE,
            axis_bins=AXIS_BINS,
            exponent=EXPONENT,
            lambda_scale=c,
        )
        for c in LAMBDAS
    }

    val_images = sorted(os.listdir(COCO_VAL))[:N_IMAGES]
    for fname in val_images:
        stem = os.path.splitext(fname)[0]
        print(f"Processing {fname} ...", end=" ", flush=True)
        img_chw = center_crop_1280(os.path.join(COCO_VAL, fname))

        panels = [add_label(img_chw.permute(1, 2, 0).numpy(), "Original")]
        for c in LAMBDAS:
            fov = foveators[c]
            tokens = fov.extract_foveated_image(img_chw)
            recon = fov.generate_foveated_visualization(tokens)
            recon_np = recon.permute(1, 2, 0).clamp(0, 255).byte().numpy()
            label = f"c={c}" + ("  (paper)" if c == 1.0 else "")
            panels.append(add_label(recon_np, label))

        out_path = os.path.join(OUT_DIR, f"img_{stem}.png")
        Image.fromarray(np.concatenate(panels, axis=1)).save(out_path)
        print(f"saved -> {out_path}")

    print(f"\nDone. {len(val_images)} composites written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
