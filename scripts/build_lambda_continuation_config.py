#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fork-from")
    source.add_argument("--resume-from")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--unfreeze-step", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--log-every", type=int, required=True)
    parser.add_argument("--val-every", type=int, required=True)
    parser.add_argument("--val-max-examples", type=int, default=128)
    parser.add_argument("--milestone-step", type=int, action="append", default=[])
    args = parser.parse_args()

    config = json.loads(Path(args.base_config).read_text())
    config["model"].update({
        "log_rect_lambda_learnable": True,
        "log_rect_lambda_unfreeze_step": args.unfreeze_step,
        "log_rect_lambda_lr": 0.001,
    })
    config["runtime"].update({
        "output_dir": args.output_dir,
        "profile_name": args.profile_name,
        "resume_from": args.resume_from,
        "fork_from": args.fork_from,
        "save_every": args.save_every,
        "milestone_steps": args.milestone_step,
        "log_every": args.log_every,
        "checkpoint_keep_last": 2,
        "wandb_enabled": True,
        "wandb_project": "SegmentThisThingLogRect2Beacon",
        "wandb_mode": "offline",
        "wandb_run_name": args.wandb_run_name,
        "wandb_tags": [
            "stt", "mae", "coco", "logrect2", "beacon", "h200", "1gpu",
            "lambda-learnable", f"from-{args.unfreeze_step}", "offline",
        ],
        "num_steps": args.num_steps,
    })
    config["mae"].update({
        "val_every": args.val_every,
        "val_max_examples": args.val_max_examples,
    })
    config["assumptions"] = [
        f"Learnable-lambda continuation from global step {args.unfreeze_step}.",
        "Lambda optimizer LR is 1e-3 with zero weight decay.",
        "W&B is explicitly named and recorded offline for later sync.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
