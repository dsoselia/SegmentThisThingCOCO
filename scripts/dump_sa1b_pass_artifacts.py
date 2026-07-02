from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from segment_this_thing.utils import get_centered_crop, get_crop_bounds
from stt_pipeline.config import load_config
from stt_pipeline.data import collate_samples, collate_segmentation_samples
from stt_pipeline.modeling import build_foveator
from stt_pipeline.runtime import configure_torch_runtime, get_device, seed_everything
from stt_pipeline.trainers import _build_loader, _materialize_segmentation_batch
from stt_pipeline.transforms import (
    build_model_inputs_batch,
    build_precomputed_crop_inputs_batch,
    get_centered_mask_crop,
    project_masks_to_foveation_batch,
)


_RUNTIME_CONFIGURED = False


def _configure_once(runtime_config) -> None:
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return
    configure_torch_runtime(runtime_config)
    _RUNTIME_CONFIGURED = True


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _stats(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    array = image.detach().cpu().round().clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(array).save(path)


def _save_gray(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().cpu().float().clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(path)


def _save_preview(path: Path, image: torch.Tensor, max_side: int = 768) -> None:
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / float(max(height, width)))
    if scale < 1.0:
        image = torch.nn.functional.interpolate(
            image.permute(2, 0, 1).unsqueeze(0).float(),
            size=(int(round(height * scale)), int(round(width * scale))),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
    _save_rgb(path, image)


def _pack_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4:
        raise ValueError(f"Expected token tensor (N, C, H, W), got {tuple(tokens.shape)}")
    axis_bins = int(round(tokens.shape[0] ** 0.5))
    if axis_bins * axis_bins != tokens.shape[0]:
        raise ValueError(f"Token count is not square: {tokens.shape[0]}")
    return (
        tokens.detach()
        .cpu()
        .unflatten(0, (axis_bins, axis_bins))
        .permute(2, 0, 3, 1, 4)
        .flatten(3, 4)
        .flatten(1, 2)
    )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n")


def dump_mae(config_path: Path, output_dir: Path, max_examples: int, device_name: str, worker_pre_crop: bool) -> dict[str, Any]:
    config = load_config(config_path)
    config.runtime.device = device_name
    config.runtime.distributed = False
    config.runtime.micro_batch_size = max_examples
    config.runtime.num_workers = 2
    config.runtime.prefetch_factor = 2
    config.runtime.persistent_workers = False
    config.mae.worker_pre_crop = worker_pre_crop
    config.mae.views_per_image = 1
    if config.model.pattern_size is None:
        raise ValueError("MAE artifact dump requires model.pattern_size")

    _configure_once(config.runtime)
    seed_everything(config.runtime.seed)
    device = get_device(config.runtime.device)
    foveator = build_foveator(config.model).to(device).eval()
    loader, _ = _build_loader(
        config.mae.train_manifest,
        config.runtime,
        config.mae.margin,
        1,
        type("Context", (), {"enabled": False, "rank": 0, "world_size": 1})(),
        task="mae",
        mae_worker_pre_crop=config.mae.worker_pre_crop,
        mae_worker_pre_crop_size=config.model.pattern_size,
        mae_worker_pre_crop_jitter_radius=config.mae.worker_pre_crop_jitter_radius,
    )
    samples = next(iter(loader))
    samples = samples[:max_examples]

    images = []
    centers = []
    crop_bounds_list = []
    original_sizes = []
    crops_for_save = []
    for sample in samples:
        image = sample.image.to(device, non_blocking=False)
        center = sample.center.to(device, non_blocking=False)
        images.append(image)
        centers.append(center)
        if config.mae.worker_pre_crop:
            crop_bounds = sample.crop_bounds.to(device)
            original_size = sample.original_image_size.to(device)
            crop = image
        else:
            crop_bounds = get_crop_bounds(center, foveator.get_pattern_bounds_size()).to(device)
            original_size = torch.tensor(image.shape[1::-1], device=device)
            crop = get_centered_crop(image, crop_bounds)
        crop_bounds_list.append(crop_bounds)
        original_sizes.append(original_size)
        crops_for_save.append(crop.detach().cpu())

    with torch.no_grad():
        if config.mae.worker_pre_crop:
            tokens, valid_masks, crop_bounds = build_precomputed_crop_inputs_batch(
                images,
                torch.stack(original_sizes),
                torch.stack(crop_bounds_list),
                foveator,
            )
        else:
            tokens, valid_masks, crop_bounds = build_model_inputs_batch(images, torch.stack(centers), foveator)

    out = output_dir / "mae"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "worker_pre_crop": config.mae.worker_pre_crop,
            "samples": samples,
            "tokens": tokens.detach().cpu(),
            "valid_masks": valid_masks.detach().cpu(),
            "crop_bounds": crop_bounds.detach().cpu(),
            "crops": crops_for_save,
        },
        out / "mae_tensors.pt",
    )

    examples = []
    for idx, sample in enumerate(samples):
        sample_dir = out / f"sample_{idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _save_preview(sample_dir / "00_worker_image_preview.png", sample.image)
        _save_rgb(sample_dir / "01_crop_1280.png", crops_for_save[idx])
        packed = _pack_tokens(tokens[idx])
        _save_rgb(sample_dir / "02_tokens_packed_208.png", packed)
        recon = foveator.generate_foveated_visualization(tokens[idx].detach().cpu())
        _save_rgb(sample_dir / "03_reconstruction_1280.png", recon)
        examples.append(
            {
                "image_path": sample.image_path,
                "dataset_name": sample.dataset_name,
                "center": sample.center,
                "crop_bounds": crop_bounds[idx],
                "original_image_size": sample.original_image_size,
                "preprocessing": sample.preprocessing,
                "worker_image": _stats(sample.image),
                "crop": _stats(crops_for_save[idx]),
                "tokens": _stats(tokens[idx]),
                "valid_token_count": int(valid_masks[idx].sum().item()),
            }
        )

    manifest = {
        "task": "mae",
        "config": str(config_path),
        "worker_pre_crop": config.mae.worker_pre_crop,
        "num_examples": len(samples),
        "device": str(device),
        "lambda_scale": float(foveator.lambda_scale.detach().cpu()),
        "examples": examples,
    }
    _write_manifest(out / "manifest.json", manifest)
    return manifest


def dump_segmentation(config_path: Path, output_dir: Path, max_examples: int, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    config.runtime.device = device_name
    config.runtime.distributed = False
    config.runtime.micro_batch_size = max(1, max_examples // max(1, config.segmentation.max_segments_per_image))
    config.runtime.num_workers = 2
    config.runtime.prefetch_factor = 2
    config.runtime.persistent_workers = False

    _configure_once(config.runtime)
    seed_everything(config.runtime.seed)
    device = get_device(config.runtime.device)
    foveator = build_foveator(config.model).to(device).eval()
    loader, _ = _build_loader(
        config.segmentation.train_manifest,
        config.runtime,
        config.segmentation.margin,
        config.segmentation.max_segments_per_image,
        type("Context", (), {"enabled": False, "rank": 0, "world_size": 1})(),
        task="segmentation",
        model_config=config.model,
        prompt_noise_std=config.segmentation.prompt_noise_std,
        sample_multiple_segments_per_image=config.segmentation.sample_multiple_segments_per_image,
    )
    batch = next(iter(loader))
    if len(batch.images) > max_examples:
        batch = collate_segmentation_samples(
            [
                type(
                    "SampleLike",
                    (),
                    {
                        "image": batch.images[i],
                        "mask": batch.masks[i],
                        "center": batch.centers[i],
                        "dataset_name": batch.dataset_names[i],
                        "image_path": batch.image_paths[i],
                        "preprocessing": {},
                    },
                )()
                for i in range(max_examples)
            ]
        )

    image_tokens, valid_masks, target_masks, timings = _materialize_segmentation_batch(
        batch,
        foveator,
        device,
        non_blocking=False,
    )
    target_single, mask_crop_bounds = project_masks_to_foveation_batch(
        foveator,
        [mask.to(device) for mask in batch.masks],
        batch.centers.to(device),
    )
    if not torch.allclose(target_masks.detach().cpu(), target_single.detach().cpu()):
        raise RuntimeError("Materialized segmentation targets do not match direct projection")

    out = output_dir / "segmentation"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "batch": batch,
            "image_tokens": image_tokens.detach().cpu(),
            "valid_masks": valid_masks.detach().cpu(),
            "target_masks": target_masks.detach().cpu(),
            "crop_bounds": mask_crop_bounds.detach().cpu(),
            "timings": timings,
        },
        out / "segmentation_tensors.pt",
    )

    examples = []
    for idx, (image, mask, center) in enumerate(zip(batch.images, batch.masks, batch.centers)):
        sample_dir = out / f"sample_{idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        crop_bounds = get_crop_bounds(center.to(device), foveator.get_pattern_bounds_size()).to(device)
        crop = get_centered_crop(image.to(device), crop_bounds).detach().cpu()
        mask_crop = get_centered_mask_crop(mask.to(device).float(), crop_bounds).detach().cpu()
        _save_preview(sample_dir / "00_worker_image_preview.png", image)
        _save_gray(sample_dir / "01_mask_original_preview.png", mask.float())
        _save_rgb(sample_dir / "02_crop_1280.png", crop)
        _save_gray(sample_dir / "03_mask_crop_1280.png", mask_crop)
        _save_rgb(sample_dir / "04_tokens_packed_208.png", _pack_tokens(image_tokens[idx]))
        _save_rgb(sample_dir / "05_reconstruction_1280.png", foveator.generate_foveated_visualization(image_tokens[idx].detach().cpu()))
        target_packed = _pack_tokens(target_masks[idx]).squeeze(0)
        _save_gray(sample_dir / "06_target_tokens_packed_208.png", target_packed)
        target_recon = foveator.generate_foveated_visualization(target_masks[idx].detach().cpu()).squeeze(0)
        _save_gray(sample_dir / "07_target_reconstruction_1280.png", target_recon)
        examples.append(
            {
                "image_path": batch.image_paths[idx],
                "dataset_name": batch.dataset_names[idx],
                "center": center,
                "crop_bounds": crop_bounds,
                "image": _stats(image),
                "mask": _stats(mask),
                "crop": _stats(crop),
                "tokens": _stats(image_tokens[idx]),
                "target": _stats(target_masks[idx]),
                "valid_token_count": int(valid_masks[idx].sum().item()),
            }
        )

    manifest = {
        "task": "segmentation",
        "config": str(config_path),
        "num_examples": len(batch.images),
        "device": str(device),
        "lambda_scale": float(foveator.lambda_scale.detach().cpu()),
        "materialize_timings": timings,
        "examples": examples,
    }
    _write_manifest(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mae-config", type=Path, required=True)
    parser.add_argument("--seg-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--mae-worker-pre-crop", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mae_manifest = dump_mae(args.mae_config, args.output_dir, args.max_examples, args.device, args.mae_worker_pre_crop)
    seg_manifest = dump_segmentation(args.seg_config, args.output_dir, args.max_examples, args.device)
    _write_manifest(
        args.output_dir / "manifest.json",
        {
            "mae": mae_manifest,
            "segmentation": seg_manifest,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
