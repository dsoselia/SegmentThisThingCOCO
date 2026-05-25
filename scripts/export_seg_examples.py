#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam_pipeline.config import load_config
from sam_pipeline.data import EvalManifestDataset, prepare_image_mask_point, resolve_eval_manifests
from sam_pipeline.modeling import build_sam_model, load_sam_checkpoint, normalize_image_batch


def _to_uint8_image(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.permute(1, 2, 0).cpu().numpy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _overlay_mask(base: np.ndarray, mask: torch.Tensor, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    canvas = base.astype(np.float32).copy()
    mask_np = mask.cpu().numpy().astype(bool)
    overlay = np.zeros_like(canvas)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    canvas[mask_np] = (1.0 - alpha) * canvas[mask_np] + alpha * overlay[mask_np]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _draw_prompt(image: np.ndarray, center_xy: torch.Tensor, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    x, y = float(center_xy[0].item()), float(center_xy[1].item())
    r = max(4, int(round(min(pil.size) * 0.01)))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(2, r // 2))
    draw.line((x - 2 * r, y, x + 2 * r, y), fill=color, width=max(2, r // 2))
    draw.line((x, y - 2 * r, x, y + 2 * r), fill=color, width=max(2, r // 2))
    return np.array(pil)


def _captioned_panel(image: np.ndarray, text: str, height: int = 28) -> Image.Image:
    panel = Image.new("RGB", (image.shape[1], image.shape[0] + height), color=(255, 255, 255))
    panel.paste(Image.fromarray(image), (0, height))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), text, fill=(0, 0, 0))
    return panel


def _render_composite(
    image: torch.Tensor,
    center: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    score: float,
    sample_index: int,
    image_path: str,
    subset_name: str,
) -> Image.Image:
    base = _to_uint8_image(image)
    prompt_only = _draw_prompt(base, center)
    pred_overlay = _draw_prompt(_overlay_mask(base, pred_mask, (0, 200, 0)), center)
    gt_overlay = _draw_prompt(_overlay_mask(base, gt_mask, (0, 120, 255)), center)

    panels = [
        _captioned_panel(base, "original"),
        _captioned_panel(prompt_only, "prompt"),
        _captioned_panel(pred_overlay, "prediction"),
        _captioned_panel(gt_overlay, "ground_truth"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + 30
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    title = f"{subset_name} | idx={sample_index} | iou={score:.4f} | {Path(image_path).name}"
    ImageDraw.Draw(canvas).text((8, 6), title, fill=(0, 0, 0))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 30))
        x += panel.width
    return canvas


def _run_inference(model: torch.nn.Module, config, sample, device: torch.device) -> tuple[torch.Tensor, float]:
    image, point_coords, _, input_size = prepare_image_mask_point(
        sample.image,
        sample.mask.float(),
        sample.center,
        image_size=config.model.image_size,
        upsample_small_images=config.evaluation.upsample_small_images,
    )
    image = image.unsqueeze(0).to(device)
    point_coords = point_coords.view(1, 1, 2).to(device)
    point_labels = torch.ones((1, 1), device=device, dtype=torch.int64)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_embeddings = model.image_encoder(normalize_image_batch(image, model))
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, pred_iou = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        best_idx = pred_iou.squeeze(0).argmax()
        masks = model.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=sample.original_size,
        )
        pred_mask = masks.squeeze(0)[best_idx] > config.evaluation.threshold

    target = sample.mask.bool()
    inter = (pred_mask.cpu() & target).sum().item()
    union = (pred_mask.cpu() | target).sum().item()
    iou = 0.0 if union == 0 else inter / union
    return pred_mask.cpu(), iou


def main() -> None:
    parser = argparse.ArgumentParser("Export qualitative segmentation examples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--num-best", type=int, default=20)
    parser.add_argument("--num-random", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    config = load_config(args.config)
    config.runtime.distributed = False
    config.evaluation.max_examples = args.max_examples

    device = torch.device(config.runtime.device)
    torch.set_float32_matmul_precision(config.runtime.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark

    manifests = resolve_eval_manifests(config.evaluation)
    if not manifests:
        raise ValueError("No evaluation manifests configured.")
    manifest_name, manifest_path = manifests[0]
    dataset = EvalManifestDataset(manifest_path, max_examples=args.max_examples)

    model = build_sam_model(
        config.model.size,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
    ).to(device).eval()
    load_sam_checkpoint(model, args.checkpoint)

    scored: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        _, iou = _run_inference(model, config, sample, device)
        scored.append(
            {
                "index": index,
                "iou": iou,
                "image_path": sample.image_path,
                "dataset_name": sample.dataset_name,
            }
        )
        if (index + 1) % 100 == 0 or (index + 1) == len(dataset):
            print(json.dumps({"processed_examples": index + 1, "running_mean_iou": sum(item["iou"] for item in scored) / len(scored)}), flush=True)

    ranked = sorted(scored, key=lambda item: item["iou"], reverse=True)
    best = ranked[: min(args.num_best, len(ranked))]
    remaining = [item for item in scored if item["index"] not in {entry["index"] for entry in best}]
    rng = random.Random(args.seed)
    random_examples = rng.sample(remaining, k=min(args.num_random, len(remaining)))
    selections = [("best", item) for item in best] + [("random", item) for item in random_examples]

    out_dir = Path(args.output_dir)
    (out_dir / "best").mkdir(parents=True, exist_ok=True)
    (out_dir / "random").mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    for subset_name, item in selections:
        sample = dataset[item["index"]]
        pred_mask, iou = _run_inference(model, config, sample, device)
        composite = _render_composite(
            image=sample.image,
            center=sample.center,
            pred_mask=pred_mask,
            gt_mask=sample.mask.bool(),
            score=iou,
            sample_index=item["index"],
            image_path=sample.image_path,
            subset_name=subset_name,
        )
        filename = f"{item['index']:04d}_iou_{iou:.4f}.png"
        composite.save(out_dir / subset_name / filename)
        metadata_rows.append(
            {
                "subset": subset_name,
                "index": item["index"],
                "iou": iou,
                "image_path": sample.image_path,
                "dataset_name": sample.dataset_name,
                "output_file": str((out_dir / subset_name / filename).resolve()),
            }
        )

    summary = {
        "manifest_name": manifest_name,
        "manifest_path": str(Path(manifest_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "max_examples": len(dataset),
        "num_best": len(best),
        "num_random": len(random_examples),
        "output_dir": str(out_dir.resolve()),
        "mean_iou": sum(item["iou"] for item in scored) / len(scored) if scored else None,
        "best_mean_iou": sum(item["iou"] for item in best) / len(best) if best else None,
        "selection_seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "selection.json").write_text(json.dumps(metadata_rows, indent=2))
    with (out_dir / "selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else ["subset", "index", "iou", "image_path", "dataset_name", "output_file"])
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
