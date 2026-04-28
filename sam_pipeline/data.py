from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import get_worker_info

from segment_anything.utils.transforms import ResizeLongestSide


def _optional_import_pycocotools():
    try:
        from pycocotools import mask as mask_utils  # type: ignore

        return mask_utils
    except Exception:
        return None


@dataclass
class MAESample:
    image: torch.Tensor
    dataset_name: str
    image_path: str


@dataclass
class SegmentationSample:
    image: torch.Tensor
    point_coords: torch.Tensor
    point_labels: torch.Tensor
    target_mask: torch.Tensor
    dataset_name: str
    image_path: str


@dataclass
class EvalSample:
    image: torch.Tensor
    mask: torch.Tensor
    center: torch.Tensor
    dataset_name: str
    image_path: str
    original_size: tuple[int, int]


def _compute_jsonl_offsets(path: str | Path) -> list[int]:
    offsets: list[int] = []
    with Path(path).open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


class IndexedJsonl:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets = _compute_jsonl_offsets(path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open() as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_manifest_path(manifest_path: str | Path, entry_path: str | Path) -> str:
    entry = Path(entry_path)
    if entry.is_absolute():
        return str(entry)
    return str((Path(manifest_path).resolve().parent / entry).resolve())


def load_image(path: str | Path) -> torch.Tensor:
    array = np.array(Image.open(path).convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float()


def _decode_polygon_mask(polygons: list[list[float]], image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for polygon in polygons:
        xy = list(zip(polygon[0::2], polygon[1::2]))
        drawer.polygon(xy, fill=255)
    return torch.from_numpy((np.array(canvas, copy=True) > 0).astype(np.float32))


def _decode_rle_mask(rle, image_size: tuple[int, int]) -> torch.Tensor:
    mask_utils = _optional_import_pycocotools()
    if mask_utils is None:
        raise RuntimeError("pycocotools is required for RLE masks.")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return torch.from_numpy(decoded.astype(np.float32))


def load_mask(mask_entry: dict[str, Any], image_size: tuple[int, int]) -> torch.Tensor:
    if "mask_path" in mask_entry:
        return torch.from_numpy((np.array(Image.open(mask_entry["mask_path"]).convert("L"), copy=True) > 0).astype(np.float32))
    if "polygon" in mask_entry:
        return _decode_polygon_mask(mask_entry["polygon"], image_size)
    if "rle" in mask_entry:
        return _decode_rle_mask(mask_entry["rle"], image_size)
    raise ValueError(f"Unsupported mask entry keys: {sorted(mask_entry.keys())}")


def pad_to_square(image: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = image.shape
    pad_h = size - height
    pad_w = size - width
    return F.pad(image, (0, pad_w, 0, pad_h))


def resize_and_pad_image(image: torch.Tensor, image_size: int) -> tuple[torch.Tensor, tuple[int, int], ResizeLongestSide]:
    resizer = ResizeLongestSide(image_size)
    image_np = image.permute(1, 2, 0).byte().numpy()
    resized_np = resizer.apply_image(image_np)
    resized = torch.from_numpy(resized_np).permute(2, 0, 1).float()
    input_size = (resized.shape[1], resized.shape[2])
    return pad_to_square(resized, image_size), input_size, resizer


def maybe_upsample_small_image_and_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    center: torch.Tensor,
    *,
    image_size: int,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not enabled:
        return image, mask, center
    height, width = image.shape[1], image.shape[2]
    longest = max(height, width)
    if longest >= image_size:
        return image, mask, center
    scale = float(image_size) / float(longest)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    upsampled_image = F.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    upsampled_mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(new_height, new_width),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    upsampled_center = center.float() * scale
    upsampled_center[0] = upsampled_center[0].clamp(0, new_width - 1)
    upsampled_center[1] = upsampled_center[1].clamp(0, new_height - 1)
    return upsampled_image, upsampled_mask, upsampled_center


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=size, mode="nearest")
    return resized.squeeze(0).squeeze(0)


def transform_prompt(center: torch.Tensor, original_size: tuple[int, int], image_size: int) -> torch.Tensor:
    resizer = ResizeLongestSide(image_size)
    coords = center.view(1, 1, 2).float()
    transformed = resizer.apply_coords_torch(coords, original_size)
    return transformed.view(2)


def prepare_image_mask_point(
    image: torch.Tensor,
    mask: torch.Tensor,
    center: torch.Tensor,
    *,
    image_size: int,
    upsample_small_images: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    image, mask, center = maybe_upsample_small_image_and_mask(
        image,
        mask,
        center,
        image_size=image_size,
        enabled=upsample_small_images,
    )
    padded_image, input_size, resizer = resize_and_pad_image(image, image_size)
    transformed_center = resizer.apply_coords_torch(center.view(1, 1, 2).float(), (image.shape[1], image.shape[2])).view(2)
    resized_mask = resize_mask(mask.float(), input_size)
    padded_mask = pad_to_square(resized_mask.unsqueeze(0), image_size).squeeze(0)
    low_res_mask = F.interpolate(
        padded_mask.unsqueeze(0).unsqueeze(0),
        size=(image_size // 4, image_size // 4),
        mode="nearest",
    ).squeeze(0)
    return padded_image, transformed_center, low_res_mask, input_size


def _sample_point(mask: torch.Tensor, rng: random.Random) -> torch.Tensor:
    ys, xs = (mask > 0).nonzero(as_tuple=True)
    if len(xs) == 0:
        raise ValueError("Cannot sample from empty mask.")
    idx = rng.randrange(len(xs))
    return torch.tensor([xs[idx].item(), ys[idx].item()], dtype=torch.float32)


def _furthest_point_with_fallback(mask: torch.Tensor) -> torch.Tensor:
    mask_np = mask.numpy().astype(np.uint8)
    try:
        import cv2  # type: ignore

        distance = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        return torch.tensor([int(x), int(y)], dtype=torch.float32)
    except Exception:
        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore

            distance = distance_transform_edt(mask_np)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            return torch.tensor([int(x), int(y)], dtype=torch.float32)
        except Exception:
            ys, xs = (mask > 0).nonzero(as_tuple=True)
            mid = len(xs) // 2
            return torch.tensor([xs[mid].item(), ys[mid].item()], dtype=torch.float32)


class MAETrainingManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str | Path, image_size: int):
        self.manifest = IndexedJsonl(manifest_path)
        self.manifest_path = str(manifest_path)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> MAESample:
        entry = self.manifest[index]
        image_path = resolve_manifest_path(self.manifest_path, entry["image_path"])
        image = load_image(image_path)
        padded_image, _, _ = resize_and_pad_image(image, self.image_size)
        return MAESample(
            image=padded_image,
            dataset_name=entry.get("dataset_name", "coco2017"),
            image_path=image_path,
        )


class SegmentationTrainingManifestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int,
        max_segments_per_image: int = 16,
        seed: int = 1337,
        prompt_noise_std: float = 0.0,
        sample_multiple_segments_per_image: bool = True,
    ):
        self.manifest = IndexedJsonl(manifest_path)
        self.manifest_path = str(manifest_path)
        self.image_size = image_size
        self.max_segments_per_image = max_segments_per_image
        self.seed = seed
        self.prompt_noise_std = prompt_noise_std
        self.sample_multiple_segments_per_image = sample_multiple_segments_per_image

    def __len__(self) -> int:
        return len(self.manifest)

    def _select_segments(self, segments: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
        if not self.sample_multiple_segments_per_image:
            return [segments[rng.randrange(len(segments))]]
        count = min(self.max_segments_per_image, len(segments))
        if count == len(segments):
            return list(segments)
        return [segments[idx] for idx in rng.sample(range(len(segments)), count)]

    def __getitem__(self, index: int) -> SegmentationSample | list[SegmentationSample]:
        worker = get_worker_info()
        worker_seed = 0 if worker is None else worker.seed
        rng = random.Random(worker_seed + self.seed + index * 9973)

        entry = self.manifest[index]
        image_path = resolve_manifest_path(self.manifest_path, entry["image_path"])
        image = load_image(image_path)
        image_size = (image.shape[2], image.shape[1])
        selected_segments = self._select_segments(entry["segments"], rng)
        samples: list[SegmentationSample] = []
        for segment in selected_segments:
            segment = dict(segment)
            if "mask_path" in segment:
                segment["mask_path"] = resolve_manifest_path(self.manifest_path, segment["mask_path"])
            mask = load_mask(segment, image_size)
            center = _sample_point(mask, rng)
            if self.prompt_noise_std > 0.0:
                center += torch.tensor([rng.gauss(0.0, self.prompt_noise_std), rng.gauss(0.0, self.prompt_noise_std)])
            center[0] = center[0].clamp(0, image.shape[2] - 1)
            center[1] = center[1].clamp(0, image.shape[1] - 1)
            padded_image, transformed_center, low_res_mask, _ = prepare_image_mask_point(
                image,
                mask,
                center,
                image_size=self.image_size,
            )
            samples.append(
                SegmentationSample(
                    image=padded_image,
                    point_coords=transformed_center,
                    point_labels=torch.tensor([1], dtype=torch.int64),
                    target_mask=low_res_mask,
                    dataset_name=entry.get("dataset_name", "coco2017"),
                    image_path=image_path,
                )
            )
        if self.sample_multiple_segments_per_image:
            return samples
        return samples[0]


class EvalManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str | Path, max_examples: Optional[int] = None):
        raw = read_jsonl(manifest_path)
        manifest_path = str(manifest_path)
        self.entries = []
        for entry in raw:
            for segment in entry["segments"]:
                segment = dict(segment)
                if "mask_path" in segment:
                    segment["mask_path"] = resolve_manifest_path(manifest_path, segment["mask_path"])
                self.entries.append(
                    {
                        "image_path": resolve_manifest_path(manifest_path, entry["image_path"]),
                        "dataset_name": entry.get("dataset_name", "coco2017"),
                        "segment": segment,
                    }
                )
        if max_examples is not None:
            self.entries = self.entries[:max_examples]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> EvalSample:
        entry = self.entries[index]
        image = load_image(entry["image_path"])
        image_size = (image.shape[2], image.shape[1])
        mask = load_mask(entry["segment"], image_size)
        center = _furthest_point_with_fallback(mask)
        return EvalSample(
            image=image,
            mask=mask.bool(),
            center=center,
            dataset_name=entry["dataset_name"],
            image_path=entry["image_path"],
            original_size=(image.shape[1], image.shape[2]),
        )


def collate_mae_samples(samples: list[MAESample]) -> MAESample:
    return MAESample(
        image=torch.stack([sample.image for sample in samples], dim=0),
        dataset_name=samples[0].dataset_name if samples else "unknown",
        image_path=samples[0].image_path if samples else "",
    )


def collate_segmentation_samples(samples: list[SegmentationSample | list[SegmentationSample]]) -> SegmentationSample:
    flat: list[SegmentationSample] = []
    for sample in samples:
        if isinstance(sample, list):
            flat.extend(sample)
        else:
            flat.append(sample)
    return SegmentationSample(
        image=torch.stack([sample.image for sample in flat], dim=0),
        point_coords=torch.stack([sample.point_coords for sample in flat], dim=0),
        point_labels=torch.stack([sample.point_labels for sample in flat], dim=0),
        target_mask=torch.stack([sample.target_mask for sample in flat], dim=0),
        dataset_name=flat[0].dataset_name if flat else "unknown",
        image_path=flat[0].image_path if flat else "",
    )


def resolve_eval_manifests(eval_config) -> list[tuple[str, str]]:
    manifests = []
    if eval_config.eval_manifest:
        manifests.append(("default", eval_config.eval_manifest))
    manifests.extend(sorted(eval_config.named_eval_manifests.items()))
    return manifests
