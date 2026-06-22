"""
Visualize LogRect2 reconstruction quality at different exponent values.

For each of 10 COCO val images, produces a 4-panel composite PNG:
  Original | exp=2.0 | exp=4.0 (current) | exp=8.0

Usage:
  python scripts/visualize_logrect_exponents.py
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
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "exponent_comparison")
EXPONENTS = [2.0, 4.0, 8.0]
PATTERN_SIZE = 1280
AXIS_BINS = 13
TOKEN_SIZE = 16
N_IMAGES = 10


def compute_bin_widths(exponent: float) -> list[int]:
    crop_half = PATTERN_SIZE / 2.0
    buffer_half = (AXIS_BINS * TOKEN_SIZE) / 2.0
    lam = crop_half / (math.e - 1.0)
    edges = []
    for i in range(AXIS_BINS + 1):
        du = i * TOKEN_SIZE - buffer_half
        ad = abs(du)
        if ad < 1e-9:
            dx = 0.0
        else:
            exp_term = lam * (math.exp((ad / buffer_half) ** exponent) - 1.0)
            dx = max(ad, exp_term) * (1.0 if du > 0 else -1.0)
        edges.append(int(math.floor(crop_half + dx)))
    edges[0] = 0
    edges[-1] = PATTERN_SIZE
    return [edges[i + 1] - edges[i] for i in range(AXIS_BINS)]


def center_crop_1280(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    if short < PATTERN_SIZE:
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

    print(f"Exponents and bin widths (outer → center):")
    for exp in EXPONENTS:
        widths = compute_bin_widths(exp)
        marker = "  ← current" if exp == 4.0 else ""
        print(f"  exp={exp:.1f}: {widths}{marker}")
    print()

    foveators = {
        exp: LogRectilinearFoveator(
            token_size=TOKEN_SIZE,
            pattern_size=PATTERN_SIZE,
            axis_bins=AXIS_BINS,
            exponent=exp,
        )
        for exp in EXPONENTS
    }

    val_images = sorted(os.listdir(COCO_VAL))[:N_IMAGES]

    for fname in val_images:
        path = os.path.join(COCO_VAL, fname)
        stem = os.path.splitext(fname)[0]
        print(f"Processing {fname} ...", end=" ", flush=True)

        img_chw = center_crop_1280(path)  # uint8 [3, H, W]

        panels = []
        original_np = img_chw.permute(1, 2, 0).numpy()
        panels.append(add_label(original_np, "Original"))

        for exp in EXPONENTS:
            fov = foveators[exp]
            tokens = fov.extract_foveated_image(img_chw)          # [169, 3, 16, 16] float
            recon = fov.generate_foveated_visualization(tokens)    # [3, 1280, 1280] float
            recon_np = recon.permute(1, 2, 0).clamp(0, 255).byte().numpy()
            label = f"exp={exp:.1f}" + ("  (current)" if exp == 4.0 else "")
            panels.append(add_label(recon_np, label))

        composite = np.concatenate(panels, axis=1)  # [1280, 4*1280, 3]
        out_path = os.path.join(OUT_DIR, f"img_{stem}.png")
        Image.fromarray(composite).save(out_path)
        print(f"saved → {out_path}")

    print(f"\nDone. {len(val_images)} composites written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
