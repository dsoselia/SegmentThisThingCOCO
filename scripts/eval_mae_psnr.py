#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from sam_pipeline.data import MAETrainingManifestDataset, resize_and_pad_image
from sam_pipeline.modeling import SamBackboneMAE, build_sam_model
from sam_pipeline.runtime import load_training_checkpoint


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int = 3) -> torch.Tensor:
    batch, num_patches, _ = patches.shape
    grid = int(num_patches**0.5)
    x = patches.reshape(batch, grid, grid, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, grid * patch_size, grid * patch_size)
    return x


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate SAM MAE PSNR")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-size", type=int, default=208)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = build_sam_model("vit_b", image_size=args.image_size, patch_size=16).to(device)
    model = SamBackboneMAE(sam).to(device)
    load_training_checkpoint(args.checkpoint, model, strict=True)
    model.eval()

    dataset = MAETrainingManifestDataset(args.manifest, args.image_size)
    limit = len(dataset) if args.max_examples <= 0 else min(len(dataset), args.max_examples)

    psnr_sum = 0.0
    count = 0
    with torch.no_grad():
        for idx in range(limit):
            sample = dataset[idx]
            image = sample.image.unsqueeze(0).to(device)
            target_image = image / 255.0

            normalized = (
                image - model.pixel_mean.view(1, -1, 1, 1).to(device)
            ) / model.pixel_std.view(1, -1, 1, 1).to(device)
            x = model.backbone.patch_embed(normalized)
            batch, grid_h, grid_w, embed_dim = x.shape
            mask = model._random_mask(batch, grid_h, grid_w, args.mask_ratio, x.device)
            if model.backbone.pos_embed is not None:
                x = x + model.backbone.pos_embed
            x = torch.where(mask.unsqueeze(-1), model.mask_token.expand(batch, grid_h, grid_w, embed_dim), x)
            for block in model.backbone.blocks:
                x = block(x)

            pred_patches = model.reconstruction_head(x.reshape(batch, grid_h * grid_w, embed_dim))
            target_patches = model.patchify(target_image)
            mask_flat = mask.reshape(batch, grid_h * grid_w, 1)
            recon_patches = torch.where(mask_flat, pred_patches, target_patches)
            recon = unpatchify(recon_patches, model.patch_size).clamp(0.0, 1.0)

            _, input_size, _ = resize_and_pad_image(sample.image, args.image_size)
            in_h, in_w = input_size
            recon_crop = recon[:, :, :in_h, :in_w]
            target_crop = target_image[:, :, :in_h, :in_w]
            mse = torch.mean((recon_crop - target_crop) ** 2).item()
            psnr = 100.0 if mse <= 0 else (-10.0 * math.log10(mse))

            psnr_sum += psnr
            count += 1
            if args.log_every > 0 and count % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "processed_examples": count,
                            "running_mean_psnr_db": psnr_sum / count,
                        }
                    ),
                    flush=True,
                )

    result = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "num_examples": count,
        "mask_ratio": args.mask_ratio,
        "metric": "inbounds_crop_psnr_db",
        "space": "resized_pre_patchify_inbounds_crop",
        "mean_psnr_db": psnr_sum / max(count, 1),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
