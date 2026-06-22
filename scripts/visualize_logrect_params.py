"""
Visualize LogRect2 reconstruction for different tokenizer parameter changes.

For a handful of COCO val images this produces, per image, three composite PNGs
that each sweep ONE parameter while holding the others at the paper defaults:

  * exponent      : 2.0 | 4.0 (default) | 6.0 | 8.0      -> falloff sharpness
  * lambda_scale  : 0.6 | 1.0 (default) | 1.4 | 1.8      -> foveal-band size  (configs c06..c18)
  * axis_bins     : 9 | 11 | 13 (default) | 15           -> token-grid resolution

Each panel is the reconstructed (foveated then de-foveated) crop so the spatial
effect of the parameter is directly visible. Panel titles also report the
resulting per-axis bin widths (outer -> center).

Usage:
  python scripts/visualize_logrect_params.py
  python scripts/visualize_logrect_params.py --n-images 4 --out figs/param_comparison
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from segment_this_thing.foveation import LogRectilinearFoveator

COCO_VAL = "/fs/cml-datasets/coco/images/val2017"

# Paper / config defaults (see configs/stt_b_coco_log_rect2_*).
PATTERN_SIZE = 1280
TOKEN_SIZE = 16
DEF_AXIS_BINS = 13
DEF_EXPONENT = 4.0
DEF_LAMBDA = 1.0

# Parameter sweeps. Each entry: (param_name, default_value, [values...]).
SWEEPS = [
    ("exponent", DEF_EXPONENT, [2.0, 4.0, 6.0, 8.0]),
    ("lambda_scale", DEF_LAMBDA, [0.6, 1.0, 1.4, 1.8]),
    ("axis_bins", DEF_AXIS_BINS, [9, 11, 13, 15]),
]


def make_foveator(**overrides) -> LogRectilinearFoveator:
    kwargs = dict(
        token_size=TOKEN_SIZE,
        pattern_size=PATTERN_SIZE,
        axis_bins=DEF_AXIS_BINS,
        exponent=DEF_EXPONENT,
        lambda_scale=DEF_LAMBDA,
    )
    kwargs.update(overrides)
    return LogRectilinearFoveator(**kwargs)


def bin_widths(fov: LogRectilinearFoveator) -> list[int]:
    edges = fov._build_axis_edges()
    return [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]


def center_crop(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if min(img.size) < PATTERN_SIZE:
        img = TF.resize(img, PATTERN_SIZE)
    img = TF.center_crop(img, [PATTERN_SIZE, PATTERN_SIZE])
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)  # uint8 CHW


def add_label(img_np: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        sfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = sfont = ImageFont.load_default()
    bar = 96 if subtitle else 60
    draw.rectangle([0, 0, img_np.shape[1], bar], fill=(0, 0, 0))
    draw.text((10, 8), title, fill=(255, 255, 255), font=font)
    if subtitle:
        draw.text((10, 58), subtitle, fill=(180, 220, 255), font=sfont)
    return np.array(pil)


def reconstruct(fov: LogRectilinearFoveator, img_chw: torch.Tensor) -> np.ndarray:
    tokens = fov.extract_foveated_image(img_chw)
    recon = fov.generate_foveated_visualization(tokens)
    return recon.permute(1, 2, 0).clamp(0, 255).byte().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "param_comparison"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    val_images = sorted(os.listdir(COCO_VAL))[: args.n_images]

    for fname in val_images:
        stem = os.path.splitext(fname)[0]
        img_chw = center_crop(os.path.join(COCO_VAL, fname))
        original = add_label(img_chw.permute(1, 2, 0).numpy(), "Original")

        for param, default, values in SWEEPS:
            panels = [original]
            for val in values:
                fov = make_foveator(**{param: val})
                recon_np = reconstruct(fov, img_chw)
                widths = bin_widths(fov)
                tag = "  (default)" if val == default else ""
                vstr = f"{val:g}" if isinstance(val, float) else str(val)
                title = f"{param}={vstr}{tag}"
                sub = f"bins: {widths}"
                panels.append(add_label(recon_np, title, sub))

            composite = np.concatenate(panels, axis=1)
            out_path = os.path.join(args.out, f"{param}_img_{stem}.png")
            Image.fromarray(composite).save(out_path)
            print(f"saved -> {out_path}")

    print(f"\nDone. Composites written to {args.out}/")


if __name__ == "__main__":
    main()
