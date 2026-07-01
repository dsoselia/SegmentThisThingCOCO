#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline.config import load_config
from stt_pipeline.data import (
    MAETrainingManifestDataset,
    SegmentationTrainingManifestDataset,
)
from stt_pipeline.modeling import build_foveator
from stt_pipeline.transforms import build_model_inputs, project_mask_to_foveation


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().cpu()
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _synthetic_samples(config, device: torch.device):
    foveator = build_foveator(config.model).to(device)
    pattern = foveator.get_pattern_bounds_size()
    image = torch.zeros((pattern, pattern, 3), dtype=torch.uint8, device=device)
    image[..., 0] = torch.arange(pattern, device=device, dtype=torch.uint8).view(1, -1)
    image[..., 1] = torch.arange(pattern, device=device, dtype=torch.uint8).view(-1, 1)
    image[..., 2] = 127
    mask = torch.zeros((pattern, pattern), dtype=torch.bool, device=device)
    low = pattern // 4
    high = 3 * pattern // 4
    mask[low:high, low:high] = True
    center = torch.tensor([pattern // 2, pattern // 2], dtype=torch.int64, device=device)
    return [(image, mask, center, "synthetic")]


def _manifest_samples(config, max_samples: int, device: torch.device):
    samples = []
    if config.segmentation.train_manifest:
        dataset = SegmentationTrainingManifestDataset(
            config.segmentation.train_manifest,
            margin=config.segmentation.margin,
            model_config=config.model,
            max_segments_per_image=config.segmentation.max_segments_per_image,
            seed=config.runtime.seed,
            prompt_noise_std=config.segmentation.prompt_noise_std,
            sample_multiple_segments_per_image=False,
        )
        for index in range(min(max_samples, len(dataset))):
            sample = dataset[index]
            if isinstance(sample, list):
                sample = sample[0]
            samples.append(
                (
                    sample.image.to(device),
                    sample.mask.to(device),
                    sample.center.to(device),
                    sample.image_path,
                )
            )
    elif config.mae.train_manifest:
        dataset = MAETrainingManifestDataset(
            config.mae.train_manifest,
            margin=config.mae.margin,
            seed=config.runtime.seed,
        )
        for index in range(min(max_samples, len(dataset))):
            sample = dataset[index]
            mask = torch.zeros(sample.image.shape[:2], dtype=torch.bool)
            samples.append(
                (
                    sample.image.to(device),
                    mask.to(device),
                    sample.center.to(device),
                    sample.image_path,
                )
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LogRect token and mask target numeric ranges.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    foveator = build_foveator(config.model).to(device)
    foveator.set_lambda_learnable(False) if hasattr(foveator, "set_lambda_learnable") else None

    samples = _synthetic_samples(config, device) if args.synthetic else _manifest_samples(config, args.max_samples, device)
    if not samples:
        samples = _synthetic_samples(config, device)

    token_stats = []
    target_stats = []
    for image, mask, center, name in samples:
        tokens, valid_mask, _ = build_model_inputs(image, center, foveator)
        target, _ = project_mask_to_foveation(foveator, mask, center)
        if not torch.isfinite(tokens.float()).all():
            raise RuntimeError(f"Non-finite image tokens for {name}")
        if not torch.isfinite(target.float()).all():
            raise RuntimeError(f"Non-finite mask target for {name}")
        token_stat = _stats(tokens)
        target_stat = _stats(target)
        token_stats.append(token_stat)
        target_stats.append(target_stat)
        if token_stat["min"] < -args.tolerance or token_stat["max"] > 255.0 + args.tolerance:
            raise RuntimeError(f"Image token range out of bounds for {name}: {token_stat}")
        if target_stat["min"] < -args.tolerance or target_stat["max"] > 1.0 + args.tolerance:
            raise RuntimeError(f"Mask target range out of bounds for {name}: {target_stat}")
        if valid_mask.shape[0] != foveator.get_num_tokens():
            raise RuntimeError(f"Valid mask shape mismatch for {name}: {tuple(valid_mask.shape)}")

    lower, upper, area = foveator.get_bin_coordinates()
    summary = {
        "config": args.config,
        "samples_checked": len(samples),
        "lambda_scale": float(foveator.lambda_scale.detach().cpu()) if hasattr(foveator, "lambda_scale") else None,
        "lambda_learnable": bool(getattr(foveator, "raw_lambda_scale", torch.tensor(0)).requires_grad)
        if hasattr(foveator, "raw_lambda_scale")
        else None,
        "num_tokens": foveator.get_num_tokens(),
        "pattern_size": foveator.get_pattern_bounds_size(),
        "bin_area": _stats(area),
        "bin_width": _stats((upper - lower).float()),
        "tokens": {
            "min": min(item["min"] for item in token_stats),
            "max": max(item["max"] for item in token_stats),
            "mean": sum(item["mean"] for item in token_stats) / len(token_stats),
        },
        "targets": {
            "min": min(item["min"] for item in target_stats),
            "max": max(item["max"] for item in target_stats),
            "mean": sum(item["mean"] for item in target_stats) / len(target_stats),
        },
    }
    expected_lambda_scale = float(config.model.log_rect_lambda_scale)
    if not math.isclose(summary["lambda_scale"], expected_lambda_scale, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError(f"Expected lambda_scale={expected_lambda_scale}, got {summary['lambda_scale']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
