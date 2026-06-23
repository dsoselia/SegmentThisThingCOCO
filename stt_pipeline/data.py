from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import get_worker_info


def _optional_import_pycocotools():
    try:
        from pycocotools import mask as mask_utils  # type: ignore

        return mask_utils
    except Exception:
        return None


@dataclass
class Sample:
    image: torch.Tensor
    mask: torch.Tensor
    center: torch.Tensor
    dataset_name: str
    image_path: str
    preprocessing: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationBatch:
    images: list[torch.Tensor]
    masks: list[torch.Tensor]
    centers: torch.Tensor
    dataset_names: list[str]
    image_paths: list[str]
    preprocessing: dict[str, float]

    def pin_memory(self):
        pinned_images: dict[int, torch.Tensor] = {}
        images = []
        for image in self.images:
            key = id(image)
            if key not in pinned_images:
                pinned_images[key] = image.pin_memory()
            images.append(pinned_images[key])
        self.images = images
        self.masks = [mask.pin_memory() for mask in self.masks]
        self.centers = self.centers.pin_memory()
        return self


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


def read_jsonl(path: str | Path) -> list[dict]:
    entries = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def resolve_manifest_path(manifest_path: str | Path, entry_path: str | Path) -> str:
    entry = Path(entry_path)
    if entry.is_absolute():
        return str(entry)
    return str((Path(manifest_path).resolve().parent / entry).resolve())


class IndexedJsonl:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets = _compute_jsonl_offsets(path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        with self.path.open() as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


def load_image(path: str | Path) -> torch.Tensor:
    array = np.array(Image.open(path).convert("RGB"), copy=True)
    return torch.from_numpy(array).byte()


def _decode_polygon_mask(polygons: list[list[float]], image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for polygon in polygons:
        xy = list(zip(polygon[0::2], polygon[1::2]))
        drawer.polygon(xy, fill=255)
    return torch.from_numpy(np.array(canvas) > 0)


def _decode_rle_mask(rle, image_size: tuple[int, int]) -> torch.Tensor:
    mask_utils = _optional_import_pycocotools()
    if mask_utils is None:
        raise RuntimeError("pycocotools is required for RLE masks.")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return torch.from_numpy(decoded.astype(bool))


def load_mask(mask_entry: dict, image_size: tuple[int, int]) -> torch.Tensor:
    if "mask_path" in mask_entry:
        mask = np.array(Image.open(mask_entry["mask_path"]).convert("L"), copy=True) > 0
        return torch.from_numpy(mask)
    if "polygon" in mask_entry:
        return _decode_polygon_mask(mask_entry["polygon"], image_size)
    if "rle" in mask_entry:
        return _decode_rle_mask(mask_entry["rle"], image_size)
    raise ValueError(f"Unsupported mask entry keys: {sorted(mask_entry.keys())}")


def _sample_point(mask: torch.Tensor, margin: int, rng: random.Random) -> torch.Tensor:
    ys, xs = mask.nonzero(as_tuple=True)
    if len(xs) == 0:
        raise ValueError("Cannot sample a prompt from an empty mask.")
    if margin > 0:
        valid = (xs >= margin) & (ys >= margin) & (xs < (mask.shape[1] - margin)) & (ys < (mask.shape[0] - margin))
        if valid.any():
            xs = xs[valid]
            ys = ys[valid]
    idx = rng.randrange(len(xs))
    return torch.tensor([xs[idx].item(), ys[idx].item()], dtype=torch.int64)


def _sample_uniform_center(image_size: tuple[int, int], margin: int, rng: random.Random) -> torch.Tensor:
    width, height = image_size
    min_x = margin
    min_y = margin
    max_x = max(margin, width - margin - 1)
    max_y = max(margin, height - margin - 1)
    if min_x > max_x:
        min_x = 0
        max_x = width - 1
    if min_y > max_y:
        min_y = 0
        max_y = height - 1
    return torch.tensor([rng.randint(min_x, max_x), rng.randint(min_y, max_y)], dtype=torch.int64)


class MAETrainingManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str | Path, margin: int, seed: int = 1337):
        self.manifest = IndexedJsonl(manifest_path)
        self.manifest_path = str(manifest_path)
        self.margin = margin
        self.seed = seed

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Sample:
        worker = get_worker_info()
        worker_seed = 0 if worker is None else worker.seed
        rng = random.Random(worker_seed + self.seed + index * 9973)
        entry = self.manifest[index]
        image_path = resolve_manifest_path(self.manifest_path, entry["image_path"])
        image = load_image(image_path)
        image_size = (image.shape[1], image.shape[0])
        center = _sample_uniform_center(image_size, self.margin, rng)
        return Sample(
            image=image,
            mask=torch.zeros((1, 1), dtype=torch.bool),
            center=center,
            dataset_name=entry.get("dataset_name", "sa1b"),
            image_path=image_path,
        )


class SegmentationTrainingManifestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        margin: int,
        model_config,
        max_segments_per_image: int = 16,
        seed: int = 1337,
        prompt_noise_std: float = 0.0,
        sample_multiple_segments_per_image: bool = False,
    ):
        self.manifest = IndexedJsonl(manifest_path)
        self.manifest_path = str(manifest_path)
        self.margin = margin
        self.model_config = model_config
        self.max_segments_per_image = max_segments_per_image
        self.seed = seed
        self.prompt_noise_std = prompt_noise_std
        self.sample_multiple_segments_per_image = sample_multiple_segments_per_image

    def __len__(self) -> int:
        return len(self.manifest)

    def _sample_segments(self, segments: list[dict], rng: random.Random) -> list[dict]:
        if not segments:
            raise ValueError("Manifest entry contains no segments.")
        if not self.sample_multiple_segments_per_image:
            return [segments[rng.randrange(len(segments))]]
        count = min(self.max_segments_per_image, len(segments))
        if count == len(segments):
            return list(segments)
        indices = rng.sample(range(len(segments)), count)
        return [segments[idx] for idx in indices]

    def __getitem__(self, index: int) -> Sample | list[Sample]:
        worker = get_worker_info()
        worker_seed = 0 if worker is None else worker.seed

        entry = self.manifest[index]
        image_path = resolve_manifest_path(self.manifest_path, entry["image_path"])
        image_load_start = time.perf_counter()
        image = load_image(image_path)
        image_decode_time = time.perf_counter() - image_load_start

        image_size = (image.shape[1], image.shape[0])
        segments = entry["segments"]
        rng = random.Random(worker_seed + self.seed + index * 9973)
        selected_segments = self._sample_segments(segments, rng)
        samples: list[Sample] = []
        for segment_idx, segment in enumerate(selected_segments):
            segment = dict(segment)
            if "mask_path" in segment:
                segment["mask_path"] = resolve_manifest_path(self.manifest_path, segment["mask_path"])
            timings: dict[str, float] = {}
            timings["image_decode_time"] = image_decode_time if segment_idx == 0 else 0.0

            mask_load_start = time.perf_counter()
            mask = load_mask(segment, image_size)
            timings["mask_decode_time"] = time.perf_counter() - mask_load_start

            prompt_start = time.perf_counter()
            center = _sample_point(mask, self.margin, rng)
            if self.prompt_noise_std > 0.0:
                noise = torch.tensor([rng.gauss(0.0, self.prompt_noise_std), rng.gauss(0.0, self.prompt_noise_std)])
                center = (center.float() + noise).round().long()
            center[0] = center[0].clamp(0, image.shape[1] - 1)
            center[1] = center[1].clamp(0, image.shape[0] - 1)
            timings["prompt_sample_time"] = time.perf_counter() - prompt_start

            samples.append(
                Sample(
                    image=image,
                    mask=mask,
                    center=center,
                    dataset_name=entry.get("dataset_name", "sa1b"),
                    image_path=image_path,
                    preprocessing={
                        **timings,
                        "worker_seed": float(worker_seed),
                    },
                )
            )

        if self.sample_multiple_segments_per_image:
            return samples
        return samples[0]


def _furthest_point_with_fallback(mask: torch.Tensor) -> torch.Tensor:
    mask_np = mask.numpy().astype(np.uint8)
    try:
        import cv2  # type: ignore

        distance = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        return torch.tensor([int(x), int(y)], dtype=torch.int64)
    except Exception:
        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore

            distance = distance_transform_edt(mask_np)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            return torch.tensor([int(x), int(y)], dtype=torch.int64)
        except Exception:
            ys, xs = mask.nonzero(as_tuple=True)
            mid = len(xs) // 2
            return torch.tensor([xs[mid].item(), ys[mid].item()], dtype=torch.int64)


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
                        "dataset_name": entry.get("dataset_name", "unknown"),
                        "segment": segment,
                    }
                )
        if max_examples is not None:
            self.entries = self.entries[:max_examples]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Sample:
        entry = self.entries[index]
        image = load_image(entry["image_path"])
        image_size = (image.shape[1], image.shape[0])
        mask = load_mask(entry["segment"], image_size)
        center = _furthest_point_with_fallback(mask)
        return Sample(
            image=image,
            mask=mask,
            center=center,
            dataset_name=entry["dataset_name"],
            image_path=entry["image_path"],
        )


def collate_samples(samples: list[Sample]) -> list[Sample]:
    return samples


def collate_segmentation_samples(samples: list[Sample | list[Sample]]) -> SegmentationBatch:
    flat_samples: list[Sample] = []
    for sample in samples:
        if isinstance(sample, list):
            flat_samples.extend(sample)
        else:
            flat_samples.append(sample)

    samples = flat_samples
    centers = torch.stack([sample.center for sample in samples])
    preprocessing = {
        "image_decode_time": sum(sample.preprocessing.get("image_decode_time", 0.0) for sample in samples),
        "mask_decode_time": sum(sample.preprocessing.get("mask_decode_time", 0.0) for sample in samples),
        "prompt_sample_time": sum(sample.preprocessing.get("prompt_sample_time", 0.0) for sample in samples),
    }
    return SegmentationBatch(
        images=[sample.image for sample in samples],
        masks=[sample.mask for sample in samples],
        centers=centers,
        dataset_names=[sample.dataset_name for sample in samples],
        image_paths=[sample.image_path for sample in samples],
        preprocessing=preprocessing,
    )


def resolve_eval_manifests(eval_config) -> list[tuple[str, str]]:
    manifests: list[tuple[str, str]] = []
    if eval_config.eval_manifest:
        manifests.append(("default", eval_config.eval_manifest))
    manifests.extend(sorted(eval_config.named_eval_manifests.items()))
    return manifests
