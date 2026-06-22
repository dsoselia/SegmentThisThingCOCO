#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segment_this_thing.foveation import LogRectilinearFoveator
from stt_pipeline.config import load_config
from stt_pipeline.modeling import build_foveator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    frozen = build_foveator(config.model).to(device)
    if not isinstance(frozen, LogRectilinearFoveator):
        raise TypeError("Preflight requires the LogRectilinearFoveator.")
    frozen.set_lambda_learnable(False)
    initial = float(frozen.lambda_scale.detach().cpu())
    if not math.isclose(initial, config.model.log_rect_lambda_scale, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Unexpected frozen lambda: {initial}")

    learnable = build_foveator(config.model).to(device)
    learnable.set_lambda_learnable(True)
    size = learnable.get_pattern_bounds_size()
    rows = torch.arange(size, device=device, dtype=torch.float32).view(1, size, 1)
    cols = torch.arange(size, device=device, dtype=torch.float32).view(1, 1, size)
    image = torch.cat(
        [
            (rows + cols).remainder(256),
            (2 * rows + cols).remainder(256),
            (rows + 3 * cols).remainder(256),
        ],
        dim=0,
    ).byte()
    tokens = learnable.extract_foveated_image(image)
    weights = torch.linspace(0.0, 1.0, tokens.numel(), device=device).reshape_as(tokens)
    (tokens * weights).mean().backward()
    gradient = learnable.raw_lambda_scale.grad
    if gradient is None or not torch.isfinite(gradient) or gradient.abs() == 0:
        raise RuntimeError(f"Invalid lambda gradient: {gradient}")
    edges = learnable._build_axis_edges_tensor()
    if not torch.all(edges.diff() > 0):
        raise RuntimeError("LogRect axis edges are not strictly increasing.")
    print(json.dumps({
        "lambda_scale": initial,
        "lambda_gradient": float(gradient.detach().cpu()),
        "num_tokens": learnable.get_num_tokens(),
        "min_axis_width": float(edges.diff().amin().detach().cpu()),
        "device": str(device),
        "status": "ok"
    }, indent=2))


if __name__ == "__main__":
    main()
