#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUUID,
    nvmlInit,
    nvmlShutdown,
)

from sam_pipeline.config import load_config
from sam_pipeline.data import EvalManifestDataset, prepare_image_mask_point, resolve_eval_manifests
from sam_pipeline.modeling import build_sam_model, load_sam_checkpoint, normalize_image_batch


class PowerSampler:
    def __init__(self, device_index: int, interval_s: float):
        self.device_index = device_index
        self.interval_s = interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.handle = None
        self.gpu_name = ""
        self.gpu_uuid = ""

    def start(self) -> None:
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.device_index)
        gpu_name = nvmlDeviceGetName(self.handle)
        gpu_uuid = nvmlDeviceGetUUID(self.handle)
        self.gpu_name = gpu_name.decode("utf-8") if isinstance(gpu_name, bytes) else str(gpu_name)
        self.gpu_uuid = gpu_uuid.decode("utf-8") if isinstance(gpu_uuid, bytes) else str(gpu_uuid)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.perf_counter()
            power_w = float(nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
            self._samples.append((ts, power_w))
            time.sleep(self.interval_s)
        ts = time.perf_counter()
        power_w = float(nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        self._samples.append((ts, power_w))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        nvmlShutdown()

    def summary(self) -> dict[str, Any]:
        if len(self._samples) < 2:
            return {
                "num_power_samples": len(self._samples),
                "avg_power_w": None,
                "peak_power_w": None,
                "energy_joules": 0.0,
                "energy_wh": 0.0,
            }
        energy_joules = 0.0
        weighted_power = 0.0
        total_dt = 0.0
        peak_power = max(power for _, power in self._samples)
        for (t0, p0), (t1, p1) in zip(self._samples[:-1], self._samples[1:]):
            dt = max(0.0, t1 - t0)
            avg_p = 0.5 * (p0 + p1)
            energy_joules += avg_p * dt
            weighted_power += avg_p * dt
            total_dt += dt
        return {
            "num_power_samples": len(self._samples),
            "avg_power_w": (weighted_power / total_dt) if total_dt > 0 else None,
            "peak_power_w": peak_power,
            "energy_joules": energy_joules,
            "energy_wh": energy_joules / 3600.0,
        }


def _manifest_path(config_path: str) -> str:
    config = load_config(config_path)
    manifests = resolve_eval_manifests(config.evaluation)
    if not manifests:
        raise ValueError("Config does not define an evaluation manifest.")
    return manifests[0][1]


def main() -> None:
    parser = argparse.ArgumentParser("Measure SAM inference power on a validation subset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--power-interval-ms", type=float, default=20.0)
    parser.add_argument("--warmup-examples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    config.runtime.distributed = False
    config.evaluation.max_examples = args.max_examples

    torch.set_float32_matmul_precision(config.runtime.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark

    device = torch.device(config.runtime.device)
    if device.type != "cuda":
        raise ValueError("Power measurement requires a CUDA device.")
    device_index = torch.cuda.current_device()

    manifest_path = _manifest_path(args.config)
    dataset = EvalManifestDataset(manifest_path, max_examples=args.max_examples)
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    model = build_sam_model(
        config.model.size,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
    ).to(device).eval()
    load_sam_checkpoint(model, args.checkpoint)

    def infer_sample(sample) -> None:
        image, point_coords, _, input_size = prepare_image_mask_point(
            sample.image,
            sample.mask.float(),
            sample.center,
            image_size=config.model.image_size,
            upsample_small_images=config.evaluation.upsample_small_images,
        )
        image = image.unsqueeze(0).to(device, non_blocking=True)
        point_coords = point_coords.view(1, 1, 2).to(device, non_blocking=True)
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
            _ = masks.squeeze(0)[best_idx]

    warmup = min(args.warmup_examples, len(dataset))
    for i in range(warmup):
        infer_sample(dataset[i])
    torch.cuda.synchronize(device)

    sampler = PowerSampler(device_index=device_index, interval_s=max(0.001, args.power_interval_ms / 1000.0))
    sample_latencies_ms: list[float] = []
    start_ts = time.perf_counter()
    sampler.start()
    try:
        for idx in range(len(dataset)):
            sample_start = time.perf_counter()
            infer_sample(dataset[idx])
            torch.cuda.synchronize(device)
            sample_latencies_ms.append((time.perf_counter() - sample_start) * 1000.0)
            processed = idx + 1
            if processed % max(1, args.log_every) == 0 or processed == len(dataset):
                elapsed = time.perf_counter() - start_ts
                running = {
                    "processed_examples": processed,
                    "elapsed_s": elapsed,
                    "samples_per_s": processed / elapsed if elapsed > 0 else None,
                }
                print(json.dumps(running), flush=True)
    finally:
        torch.cuda.synchronize(device)
        end_ts = time.perf_counter()
        sampler.stop()

    power_summary = sampler.summary()
    elapsed_s = end_ts - start_ts
    result = {
        "metric": "segmentation_inference_power",
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "max_examples": len(dataset),
        "warmup_examples": warmup,
        "elapsed_s": elapsed_s,
        "samples_per_s": len(dataset) / elapsed_s if elapsed_s > 0 else None,
        "mean_latency_ms": statistics.fmean(sample_latencies_ms) if sample_latencies_ms else None,
        "median_latency_ms": statistics.median(sample_latencies_ms) if sample_latencies_ms else None,
        "energy_joules": power_summary["energy_joules"],
        "energy_wh": power_summary["energy_wh"],
        "energy_per_sample_joules": power_summary["energy_joules"] / len(dataset),
        "avg_power_w": power_summary["avg_power_w"],
        "peak_power_w": power_summary["peak_power_w"],
        "num_power_samples": power_summary["num_power_samples"],
        "power_interval_ms": args.power_interval_ms,
        "gpu": {
            "index": device_index,
            "name": sampler.gpu_name,
            "uuid": sampler.gpu_uuid,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
