#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segment_this_thing.utils import get_centered_crop
from stt_pipeline import load_config
from stt_pipeline.data import MAETrainingManifestDataset
from stt_pipeline.mae import FoveatedMAE
from stt_pipeline.modeling import build_foveator, build_model, load_checkpoint
from stt_pipeline.transforms import build_model_inputs


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _valid_crop_mask(crop_bounds: torch.Tensor, image_size: tuple[int, int], crop_size: int) -> torch.Tensor:
    lower = crop_bounds[0].cpu()
    upper = crop_bounds[1].cpu()
    width, height = image_size
    x0 = max(lower[0].item(), 0)
    y0 = max(lower[1].item(), 0)
    x1 = min(upper[0].item(), width)
    y1 = min(upper[1].item(), height)

    crop_x0 = x0 - lower[0].item()
    crop_y0 = y0 - lower[1].item()
    crop_x1 = crop_x0 + (x1 - x0)
    crop_y1 = crop_y0 + (y1 - y0)

    mask = torch.zeros((crop_size, crop_size), dtype=torch.bool)
    if crop_x1 > crop_x0 and crop_y1 > crop_y0:
        mask[crop_y0:crop_y1, crop_x0:crop_x1] = True
    return mask


def _feature_dim_for_model_size(size: str) -> int:
    if size not in {"b", "l", "h"}:
        raise ValueError(f"Unsupported model size: {size}")
    return 256


def _build_mask(num_tokens: int, valid_mask: torch.Tensor, mask_ratio: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    token_mask = torch.rand((num_tokens,), generator=generator) < mask_ratio
    return token_mask & valid_mask.cpu()


def _summarize_sse_sae(sse: float, sae: float, count: int) -> dict[str, float]:
    if count <= 0:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "psnr": float("nan"),
        }
    mse = sse / count
    mae = sae / count
    return {
        "mse": mse,
        "mae": mae,
        "psnr": _psnr_from_mse(mse),
    }


def _build_summary(
    *,
    config,
    args,
    manifest_path: str,
    total_examples: int,
    processed_examples: int,
    token_all_sse: float,
    token_all_sae: float,
    token_all_count: int,
    token_masked_sse: float,
    token_masked_sae: float,
    token_masked_count: int,
    crop_full_sse: float,
    crop_full_sae: float,
    crop_full_count: int,
    crop_valid_sse: float,
    crop_valid_sae: float,
    crop_valid_count: int,
    valid_token_fraction_sum: float,
    masked_token_fraction_sum: float,
    completed: bool,
) -> dict:
    processed_safe = max(processed_examples, 1)
    return {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "manifest": str(manifest_path),
        "examples": total_examples,
        "processed_examples": processed_examples,
        "completed": completed,
        "mask_ratio": args.mask_ratio if args.mask_ratio is not None else config.mae.mask_ratio,
        "seed": args.seed,
        "model": asdict(config.model),
        "token_metrics_valid_all": _summarize_sse_sae(token_all_sse, token_all_sae, token_all_count),
        "token_metrics_valid_masked": _summarize_sse_sae(token_masked_sse, token_masked_sae, token_masked_count),
        "crop_metrics_full_1280": _summarize_sse_sae(crop_full_sse, crop_full_sae, crop_full_count),
        "crop_metrics_valid_region_only": _summarize_sse_sae(crop_valid_sse, crop_valid_sae, crop_valid_count),
        "mean_valid_token_fraction": valid_token_fraction_sum / processed_safe,
        "mean_masked_token_fraction": masked_token_fraction_sum / processed_safe,
    }


def _write_summary(path: Path | None, summary: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2))
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MAE reconstruction metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = args.manifest or config.mae.train_manifest
    if manifest_path is None:
        raise ValueError("A manifest path is required either via --manifest or config.mae.train_manifest.")
    mask_ratio = config.mae.mask_ratio if args.mask_ratio is None else args.mask_ratio
    output_path = Path(args.output) if args.output is not None else None

    device = torch.device(args.device)
    foveator = build_foveator(config.model)
    model = build_model(config.model.size, foveator)
    trainer = FoveatedMAE(
        image_encoder=model.image_encoder,
        feature_dim=_feature_dim_for_model_size(config.model.size),
        token_size=config.model.token_size,
    ).to(device)
    load_checkpoint(trainer, args.checkpoint, strict=True)
    trainer.eval()

    dataset = MAETrainingManifestDataset(manifest_path, margin=config.mae.margin, seed=config.runtime.seed)
    total_examples = len(dataset) if args.max_examples is None else min(args.max_examples, len(dataset))

    token_all_sse = 0.0
    token_all_sae = 0.0
    token_all_count = 0

    token_masked_sse = 0.0
    token_masked_sae = 0.0
    token_masked_count = 0

    crop_full_sse = 0.0
    crop_full_sae = 0.0
    crop_full_count = 0

    crop_valid_sse = 0.0
    crop_valid_sae = 0.0
    crop_valid_count = 0

    valid_token_fraction_sum = 0.0
    masked_token_fraction_sum = 0.0

    with torch.no_grad():
        for batch_start in range(0, total_examples, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_examples)
            samples = [dataset[index] for index in range(batch_start, batch_end)]

            target_token_batch = []
            valid_mask_batch = []
            token_mask_batch = []
            crop_bounds_batch = []
            for offset, sample in enumerate(samples):
                image = sample.image.to(device)
                center = sample.center.to(device)
                tokens, valid_mask, crop_bounds = build_model_inputs(image, center, foveator)
                target_token_batch.append(tokens.float().to(device) / 255.0)
                valid_mask_batch.append(valid_mask.to(device))
                token_mask_batch.append(
                    _build_mask(foveator.get_num_tokens(), valid_mask, mask_ratio, args.seed + batch_start + offset).to(device)
                )
                crop_bounds_batch.append(crop_bounds.cpu())

            target_tokens = torch.stack(target_token_batch)
            valid_masks = torch.stack(valid_mask_batch)
            token_masks = torch.stack(token_mask_batch)

            masked_tokens = target_tokens.clone()
            num_masked = int(token_masks.sum().item())
            if num_masked > 0:
                masked_tokens[token_masks] = trainer.mask_token.unsqueeze(0).expand(num_masked, -1, -1, -1)

            image_features, _ = trainer.image_encoder(masked_tokens, valid_masks)
            recon_batch = trainer.decoder(image_features).view(
                len(samples), foveator.get_num_tokens(), 3, config.model.token_size, config.model.token_size
            )

            for offset, sample in enumerate(samples):
                recon_tokens = recon_batch[offset].float().cpu()
                target_tokens_cpu = target_tokens[offset].float().cpu()
                valid_mask_cpu = valid_masks[offset].cpu()
                token_mask_cpu = token_masks[offset].cpu()

                token_error = recon_tokens - target_tokens_cpu
                token_abs = token_error.abs()
                token_sq = token_error.square()

                valid_selector = valid_mask_cpu.view(-1, 1, 1, 1)
                masked_selector = token_mask_cpu.view(-1, 1, 1, 1)

                token_all_sse += float(token_sq[valid_selector.expand_as(token_sq)].sum().item())
                token_all_sae += float(token_abs[valid_selector.expand_as(token_abs)].sum().item())
                token_all_count += int(valid_mask_cpu.sum().item()) * token_sq.shape[1] * token_sq.shape[2] * token_sq.shape[3]

                token_masked_sse += float(token_sq[masked_selector.expand_as(token_sq)].sum().item())
                token_masked_sae += float(token_abs[masked_selector.expand_as(token_abs)].sum().item())
                token_masked_count += int(token_mask_cpu.sum().item()) * token_sq.shape[1] * token_sq.shape[2] * token_sq.shape[3]

                recon_crop = foveator.generate_foveated_visualization(recon_tokens).float()
                target_crop = get_centered_crop(sample.image, crop_bounds_batch[offset]).permute(2, 0, 1).float() / 255.0
                crop_error = recon_crop - target_crop
                crop_abs = crop_error.abs()
                crop_sq = crop_error.square()

                crop_full_sse += float(crop_sq.sum().item())
                crop_full_sae += float(crop_abs.sum().item())
                crop_full_count += crop_sq.numel()

                valid_crop = _valid_crop_mask(
                    crop_bounds_batch[offset],
                    (sample.image.shape[1], sample.image.shape[0]),
                    foveator.get_pattern_bounds_size(),
                )
                valid_crop_selector = valid_crop.unsqueeze(0).expand_as(crop_sq)
                crop_valid_sse += float(crop_sq[valid_crop_selector].sum().item())
                crop_valid_sae += float(crop_abs[valid_crop_selector].sum().item())
                crop_valid_count += int(valid_crop.sum().item()) * crop_sq.shape[0]

                valid_token_fraction_sum += float(valid_mask_cpu.float().mean().item())
                masked_token_fraction_sum += float(token_mask_cpu.float().mean().item())

            summary = _build_summary(
                config=config,
                args=args,
                manifest_path=str(manifest_path),
                total_examples=total_examples,
                processed_examples=batch_end,
                token_all_sse=token_all_sse,
                token_all_sae=token_all_sae,
                token_all_count=token_all_count,
                token_masked_sse=token_masked_sse,
                token_masked_sae=token_masked_sae,
                token_masked_count=token_masked_count,
                crop_full_sse=crop_full_sse,
                crop_full_sae=crop_full_sae,
                crop_full_count=crop_full_count,
                crop_valid_sse=crop_valid_sse,
                crop_valid_sae=crop_valid_sae,
                crop_valid_count=crop_valid_count,
                valid_token_fraction_sum=valid_token_fraction_sum,
                masked_token_fraction_sum=masked_token_fraction_sum,
                completed=(batch_end == total_examples),
            )
            _write_summary(output_path, summary)
            if batch_end % args.log_every == 0 or batch_end == total_examples:
                print(json.dumps({"processed": batch_end, "total_examples": total_examples}), flush=True)

    summary = _build_summary(
        config=config,
        args=args,
        manifest_path=str(manifest_path),
        total_examples=total_examples,
        processed_examples=total_examples,
        token_all_sse=token_all_sse,
        token_all_sae=token_all_sae,
        token_all_count=token_all_count,
        token_masked_sse=token_masked_sse,
        token_masked_sae=token_masked_sae,
        token_masked_count=token_masked_count,
        crop_full_sse=crop_full_sse,
        crop_full_sae=crop_full_sae,
        crop_full_count=crop_full_count,
        crop_valid_sse=crop_valid_sse,
        crop_valid_sae=crop_valid_sae,
        crop_valid_count=crop_valid_count,
        valid_token_fraction_sum=valid_token_fraction_sum,
        masked_token_fraction_sum=masked_token_fraction_sum,
        completed=True,
    )
    _write_summary(output_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
