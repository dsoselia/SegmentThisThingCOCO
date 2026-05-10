#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline.config import load_config
from stt_pipeline.data import EvalManifestDataset
from stt_pipeline.evaluate import _normalize_tokens
from stt_pipeline.modeling import build_foveator, build_model, load_checkpoint
from stt_pipeline.transforms import build_model_inputs, maybe_resize_small_image


def _load_pynvml():
    try:
        import pynvml  # type: ignore

        return pynvml
    except Exception as exc:
        raise RuntimeError(f"Failed to import pynvml: {exc}") from exc


class PowerSampler:
    def __init__(self, device_index: int, interval_s: float = 0.05):
        self.pynvml = _load_pynvml()
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

    def start(self) -> None:
        self.pynvml.nvmlInit()
        self._handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._handle is not None
        while not self._stop.is_set():
            t = time.perf_counter()
            power_w = float(self.pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            self.samples.append((t, power_w))
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass

    def gpu_name(self) -> str | None:
        if self._handle is None:
            return None
        try:
            return self.pynvml.nvmlDeviceGetName(self._handle)
        except Exception:
            return None

    def total_energy_j(self, start_t: float, end_t: float) -> float:
        points = [(t, p) for t, p in self.samples if start_t <= t <= end_t]
        if len(points) < 2:
            return 0.0
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(points[:-1], points[1:]):
            energy += 0.5 * (p0 + p1) * (t1 - t0)
        return energy

    def peak_power_w(self, start_t: float, end_t: float) -> float:
        points = [p for t, p in self.samples if start_t <= t <= end_t]
        return max(points) if points else 0.0


def build_linear_lookup_map(foveator) -> torch.Tensor:
    pattern = foveator.get_pattern_bounds_size()
    linear = torch.empty((pattern, pattern), dtype=torch.long)
    lower = foveator.bin_lower_pixel_coords.cpu()
    upper = foveator.bin_upper_pixel_coords.cpu()
    token_size = foveator.token_size
    token_area = token_size * token_size
    for token in range(lower.shape[0]):
        token_base = token * token_area
        for row in range(lower.shape[1]):
            row_base = token_base + row * token_size
            for col in range(lower.shape[2]):
                x0 = lower[token, row, col, 0].item()
                y0 = lower[token, row, col, 1].item()
                x1 = upper[token, row, col, 0].item()
                y1 = upper[token, row, col, 1].item()
                linear[y0:y1, x0:x1] = row_base + col
    return linear


def fast_reconstruct_logits_to_image(
    foveator,
    logits: torch.Tensor,
    crop_bounds: torch.Tensor,
    image_size: tuple[int, int],
    linear_lookup_map: torch.Tensor,
) -> torch.Tensor:
    flat = logits.reshape(-1).cpu()
    crop = flat[linear_lookup_map]
    height, width = image_size
    out = torch.zeros((height, width), dtype=crop.dtype)
    lower = crop_bounds[0].cpu()
    upper = crop_bounds[1].cpu()
    x0 = max(lower[0].item(), 0)
    y0 = max(lower[1].item(), 0)
    x1 = min(upper[0].item(), width)
    y1 = min(upper[1].item(), height)
    crop_x0 = x0 - lower[0].item()
    crop_y0 = y0 - lower[1].item()
    crop_x1 = crop_x0 + (x1 - x0)
    crop_y1 = crop_y0 + (y1 - y0)
    out[y0:y1, x0:x1] = crop[crop_y0:crop_y1, crop_x0:crop_x1]
    return out


@torch.inference_mode()
def _run_inference_batch(config, model, foveator, dataset, device: torch.device, count: int, linear_lookup_map) -> list[float]:
    latencies_s: list[float] = []
    for index in range(count):
        start_t = time.perf_counter()
        sample = dataset[index]
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

        image = image.to(device, non_blocking=False)
        center = center.to(device, non_blocking=False)
        tokens, valid_mask, crop_bounds = build_model_inputs(image, center, foveator)
        norm = _normalize_tokens(tokens.unsqueeze(0).float(), device)
        pred_masks, pred_iou = model(norm, valid_mask.unsqueeze(0))
        best_idx = pred_iou.squeeze(0).argmax()
        recon = fast_reconstruct_logits_to_image(
            foveator,
            pred_masks.squeeze(0)[best_idx].cpu(),
            crop_bounds.cpu(),
            tuple(image.shape[:2]),
            linear_lookup_map,
        )
        _ = recon.sigmoid() > config.evaluation.threshold
        torch.cuda.synchronize(device)
        latencies_s.append(time.perf_counter() - start_t)
    return latencies_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure single-GPU inference power with fast image-space reconstruction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--warmup-examples", type=int, default=10)
    parser.add_argument("--sample-interval-ms", type=float, default=50.0)
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("This script is intended for CUDA power measurement.")

    config = load_config(args.config)
    config.evaluation.max_examples = args.max_examples + args.warmup_examples

    device = torch.device("cuda")
    foveator = build_foveator(config.model).to(device)
    model = build_model(config.model.size, foveator).to(device).eval()
    load_checkpoint(model, args.checkpoint, strict=True)
    linear_lookup_map = build_linear_lookup_map(foveator)

    dataset = EvalManifestDataset(args.manifest, config.evaluation.max_examples)
    if len(dataset) < args.max_examples + args.warmup_examples:
        raise ValueError(
            f"Need at least {args.max_examples + args.warmup_examples} examples, got {len(dataset)}"
        )

    if args.warmup_examples > 0:
        _run_inference_batch(config, model, foveator, dataset, device, args.warmup_examples, linear_lookup_map)
        torch.cuda.synchronize(device)

    measured_dataset = torch.utils.data.Subset(
        dataset, list(range(args.warmup_examples, args.warmup_examples + args.max_examples))
    )
    sampler = PowerSampler(device_index=torch.cuda.current_device(), interval_s=args.sample_interval_ms / 1000.0)

    torch.cuda.synchronize(device)
    sampler.start()
    start_t = time.perf_counter()
    latencies_s = _run_inference_batch(config, model, foveator, measured_dataset, device, args.max_examples, linear_lookup_map)
    torch.cuda.synchronize(device)
    end_t = time.perf_counter()
    sampler.stop()

    duration_s = end_t - start_t
    energy_j = sampler.total_energy_j(start_t, end_t)
    avg_power_w = energy_j / duration_s if duration_s > 0 else 0.0
    peak_power_w = sampler.peak_power_w(start_t, end_t)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "examples_measured": args.max_examples,
        "warmup_examples": args.warmup_examples,
        "sample_interval_ms": args.sample_interval_ms,
        "gpu_name": sampler.gpu_name(),
        "duration_s": duration_s,
        "total_energy_j": energy_j,
        "average_power_w": avg_power_w,
        "peak_power_w": peak_power_w,
        "energy_per_example_j": energy_j / args.max_examples if args.max_examples > 0 else None,
        "examples_per_second": args.max_examples / duration_s if duration_s > 0 else None,
        "mean_latency_ms": 1000.0 * (sum(latencies_s) / len(latencies_s)) if latencies_s else None,
        "median_latency_ms": 1000.0 * statistics.median(latencies_s) if latencies_s else None,
        "tokenizer_type": config.model.tokenizer_type,
        "num_tokens": foveator.get_num_tokens(),
        "pattern_size": foveator.get_pattern_bounds_size(),
        "reconstruction_impl": "fast_linear_lookup_cpu",
        "linear_lookup_map_mb": linear_lookup_map.numel() * linear_lookup_map.element_size() / 1024.0 / 1024.0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
