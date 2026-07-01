#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _summarize_metrics(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steady = [row for row in rows if int(row.get("step", 0)) > 0]
    result = {"rows": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"]}
    for key in (
        "seconds_per_step",
        "samples_per_second",
        "tokens_per_second",
        "batch_fetch_time",
        "json_read_time",
        "image_decode_time",
        "mask_decode_time",
        "prompt_sample_time",
        "host_to_device_time",
        "foveation_build_time",
        "target_projection_time",
        "stack_batch_time",
        "model_forward_backward_time",
    ):
        values = [float(row[key]) for row in steady if key in row]
        if values:
            result[key] = {
                "mean": statistics.mean(values),
                "p50": statistics.median(values),
                "p90": _percentile(values, 0.90),
                "p99": _percentile(values, 0.99),
                "min": min(values),
                "max": max(values),
            }
    return result


def _summarize_gpu(path: Path) -> dict | None:
    if not path.exists():
        return None
    utils = []
    memory = []
    with path.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            try:
                utils.append(float(row[3]))
                memory.append(float(row[5]))
            except ValueError:
                continue
    if not utils:
        return None
    return {
        "gpu_util_mean": statistics.mean(utils),
        "gpu_util_p50": statistics.median(utils),
        "gpu_util_p90": _percentile(utils, 0.90),
        "memory_used_mb_mean": statistics.mean(memory) if memory else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Summarize SA-1B speed profile runs.")
    parser.add_argument("--root", default="/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    summaries = {}
    for run_root in sorted((root / "runs").glob("sa1b_speed_profile_*")):
        metrics_files = sorted(run_root.glob("*/metrics.jsonl"))
        if not metrics_files:
            continue
        name = run_root.name
        summary = _summarize_metrics(metrics_files[-1])
        task_variant = name.removeprefix("sa1b_speed_profile_")
        gpu_matches = sorted((root / "slurm").glob(f"sa1b_speed_profile_{task_variant}_*_gpu.csv"))
        if gpu_matches:
            summary["gpu"] = _summarize_gpu(gpu_matches[-1])
        event_files = sorted(run_root.glob("*/profile_events.jsonl"))
        if event_files:
            summary["profile_events"] = [json.loads(line) for line in event_files[-1].read_text().splitlines() if line.strip()]
        summaries[name] = summary

    payload = {"root": str(root), "runs": summaries}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
