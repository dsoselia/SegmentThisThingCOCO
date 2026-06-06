#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline import load_config
from stt_pipeline.data import EvalManifestDataset
from stt_pipeline.modeling import build_foveator, build_model, load_checkpoint
from stt_pipeline.runtime import get_device
from stt_pipeline.transforms import build_model_inputs, maybe_resize_small_image, reconstruct_logits_to_image


def _normalize_tokens(tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    return (tokens / 255.0 - mean_tensor) / std_tensor


def _resolve_nvml_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return 0


class PowerSampler:
    def __init__(self, device_index: int, interval_s: float) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._handle = None

    def start(self) -> None:
        nvmlInit()
        self._handle = nvmlDeviceGetHandleByIndex(self.device_index)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self._handle is not None:
            try:
                self._record_sample()
            except Exception:
                pass
        nvmlShutdown()

    def _record_sample(self) -> None:
        assert self._handle is not None
        now = time.perf_counter()
        power_w = float(nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        util = nvmlDeviceGetUtilizationRates(self._handle)
        mem = nvmlDeviceGetMemoryInfo(self._handle)
        self.samples.append(
            {
                "t": now,
                "power_w": power_w,
                "gpu_util": float(util.gpu),
                "mem_util": float(util.memory),
                "mem_used_mb": float(mem.used) / (1024.0 * 1024.0),
            }
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record_sample()
            time.sleep(self.interval_s)

    def summarize(self, elapsed_s: float) -> dict[str, float]:
        if not self.samples:
            return {
                "energy_joules": 0.0,
                "avg_power_w": 0.0,
                "peak_power_w": 0.0,
                "avg_gpu_util": 0.0,
                "avg_mem_util": 0.0,
                "peak_mem_used_mb": 0.0,
                "num_power_samples": 0,
            }
        if len(self.samples) == 1:
            energy_joules = self.samples[0]["power_w"] * elapsed_s
        else:
            energy_joules = 0.0
            for left, right in zip(self.samples[:-1], self.samples[1:]):
                dt = right["t"] - left["t"]
                energy_joules += 0.5 * (left["power_w"] + right["power_w"]) * dt
        return {
            "energy_joules": energy_joules,
            "avg_power_w": energy_joules / max(elapsed_s, 1e-9),
            "peak_power_w": max(sample["power_w"] for sample in self.samples),
            "avg_gpu_util": mean(sample["gpu_util"] for sample in self.samples),
            "avg_mem_util": mean(sample["mem_util"] for sample in self.samples),
            "peak_mem_used_mb": max(sample["mem_used_mb"] for sample in self.samples),
            "num_power_samples": len(self.samples),
        }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Measure STT inference power on a fixed number of eval samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    args = parser.parse_args()

    config = load_config(args.config)
    config.runtime.device = "cuda"
    config.runtime.wandb_enabled = False
    config.evaluation.max_examples = args.max_examples
    if args.manifest is not None:
        config.evaluation.eval_manifest = args.manifest
        config.evaluation.named_eval_manifests = {}

    manifest_path = args.manifest
    if manifest_path is None:
        if config.evaluation.eval_manifest:
            manifest_path = config.evaluation.eval_manifest
        elif config.evaluation.named_eval_manifests:
            manifest_path = next(iter(config.evaluation.named_eval_manifests.values()))
        else:
            raise ValueError("No eval manifest configured.")

    device = get_device(config.runtime.device)
    foveator = build_foveator(config.model).to(device)
    model = build_model(config.model.size, foveator)
    load_checkpoint(model, args.checkpoint, strict=True)
    model = model.to(device).eval()

    dataset = EvalManifestDataset(manifest_path, config.evaluation.max_examples)
    power_sampler = PowerSampler(_resolve_nvml_index(), args.sample_interval_ms / 1000.0)

    per_sample_latency_ms: list[float] = []

    start = time.perf_counter()
    power_sampler.start()
    try:
        for sample in dataset:
            sample_start = time.perf_counter()
            image = sample.image
            if config.evaluation.upsample_small_images:
                image = maybe_resize_small_image(image, foveator.get_pattern_bounds_size())
                scale_x = image.shape[1] / sample.image.shape[1]
                scale_y = image.shape[0] / sample.image.shape[0]
                center = torch.tensor(
                    [int(round(sample.center[0].item() * scale_x)), int(round(sample.center[1].item() * scale_y))],
                    dtype=torch.int64,
                )
            else:
                center = sample.center

            image = image.to(device)
            center = center.to(device)
            tokens, valid_mask, crop_bounds = build_model_inputs(image, center, foveator)
            norm = _normalize_tokens(tokens.unsqueeze(0).float(), device)
            pred_masks, pred_iou = model(norm, valid_mask.unsqueeze(0))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pred_masks_cpu = pred_masks.squeeze(0).cpu()
            pred_iou_cpu = pred_iou.squeeze(0).cpu()
            center_x = int(center[0].item())
            center_y = int(center[1].item())

            best_index = int(pred_iou_cpu.argmax().item())
            best_score = float("-inf")
            for candidate_index, candidate_logits in enumerate(pred_masks_cpu):
                recon = reconstruct_logits_to_image(
                    foveator,
                    candidate_logits,
                    crop_bounds.cpu(),
                    tuple(image.shape[:2]),
                ).squeeze(0)
                pred = recon.sigmoid() > config.evaluation.threshold
                if bool(pred[center_y, center_x].item()):
                    score = float(pred_iou_cpu[candidate_index].item())
                    if score > best_score:
                        best_score = score
                        best_index = candidate_index

            _selected_mask = reconstruct_logits_to_image(
                foveator,
                pred_masks_cpu[best_index],
                crop_bounds.cpu(),
                tuple(image.shape[:2]),
            ).squeeze(0).sigmoid() > config.evaluation.threshold
            per_sample_latency_ms.append(1000.0 * (time.perf_counter() - sample_start))
    finally:
        elapsed_s = time.perf_counter() - start
        power_sampler.stop()

    summary: dict[str, Any] = {
        "checkpoint_path": args.checkpoint,
        "manifest_path": manifest_path,
        "num_examples": len(per_sample_latency_ms),
        "mode": "inference_image_space_mask_only",
        "timing": {
            "elapsed_s": elapsed_s,
            "mean_latency_ms": mean(per_sample_latency_ms) if per_sample_latency_ms else 0.0,
            "median_latency_ms": sorted(per_sample_latency_ms)[len(per_sample_latency_ms) // 2] if per_sample_latency_ms else 0.0,
            "samples_per_second": len(per_sample_latency_ms) / max(elapsed_s, 1e-9),
        },
        "power": power_sampler.summarize(elapsed_s),
    }
    summary["power"]["joules_per_sample"] = summary["power"]["energy_joules"] / max(summary["num_examples"], 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
