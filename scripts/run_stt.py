#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline import (
    evaluate_checkpoint,
    load_config,
    run_mae_pretraining,
    run_segmentation_preflight,
    run_segmentation_train_start_smoke,
    run_segmentation_training,
)
from stt_pipeline.trainers import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser("STT replication pipeline")
    parser.add_argument(
        "command",
        choices=["pretrain-mae", "train-stt", "train-start-smoke-stt", "eval-stt", "benchmark-stt", "preflight-stt"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "pretrain-mae":
        run_dir = run_mae_pretraining(config)
        print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    elif args.command == "train-stt":
        run_dir = run_segmentation_training(config)
        print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    elif args.command == "train-start-smoke-stt":
        run_dir = run_segmentation_train_start_smoke(config)
        print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    elif args.command == "eval-stt":
        summary = evaluate_checkpoint(config, checkpoint_path=args.checkpoint)
        print(json.dumps(summary, indent=2))
    elif args.command == "benchmark-stt":
        summary = run_benchmark(config)
        print(json.dumps(summary, indent=2))
    elif args.command == "preflight-stt":
        summary = run_segmentation_preflight(config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
