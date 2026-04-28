#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_pipeline import evaluate_checkpoint, load_config, run_benchmark, run_mae_pretraining, run_preflight, run_segmentation_training


def main() -> None:
    parser = argparse.ArgumentParser("SAM COCO baseline pipeline")
    parser.add_argument("command", choices=["pretrain-mae", "train-sam", "eval-sam", "benchmark-sam", "preflight-sam"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "pretrain-mae":
        result = {"run_dir": str(run_mae_pretraining(config))}
    elif args.command == "train-sam":
        result = {"run_dir": str(run_segmentation_training(config))}
    elif args.command == "eval-sam":
        result = evaluate_checkpoint(config, checkpoint_path=args.checkpoint)
    elif args.command == "benchmark-sam":
        result = run_benchmark(config)
    else:
        result = run_preflight(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
