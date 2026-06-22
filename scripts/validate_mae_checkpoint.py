#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline.config import load_config
from stt_pipeline.mae import FoveatedMAE
from stt_pipeline.modeling import build_foveator, build_model, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-completed-steps", type=int, required=True)
    parser.add_argument("--expected-lambda", type=float)
    parser.add_argument("--expect-lambda-changed-from", type=float)
    parser.add_argument("--minimum-change", type=float, default=1e-8)
    parser.add_argument("--expect-learnable", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    foveator = build_foveator(config.model)
    segment_model = build_model(config.model.size, foveator)
    trainer = FoveatedMAE(
        image_encoder=segment_model.image_encoder,
        feature_dim=segment_model.mask_decoder.pos_enc.shape[-1],
        token_size=config.model.token_size,
        foveator=foveator,
    )
    state = load_checkpoint(trainer, args.checkpoint, strict=True)
    completed_steps = int(state.get("extra", {}).get("completed_steps", int(state.get("step", -1)) + 1))
    value = float(trainer.foveator.lambda_scale.detach())
    if completed_steps != args.expected_completed_steps:
        raise RuntimeError(f"Expected {args.expected_completed_steps} completed steps, got {completed_steps}")
    if args.expected_lambda is not None and not math.isclose(value, args.expected_lambda, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Expected lambda {args.expected_lambda}, got {value}")
    if args.expect_lambda_changed_from is not None and abs(value - args.expect_lambda_changed_from) < args.minimum_change:
        raise RuntimeError(f"Lambda did not change enough from {args.expect_lambda_changed_from}: {value}")
    saved_learnable = bool(state.get("config", {}).get("model", {}).get("log_rect_lambda_learnable", False))
    if saved_learnable != args.expect_learnable:
        raise RuntimeError(f"Expected saved learnable={args.expect_learnable}, got {saved_learnable}")
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "completed_steps": completed_steps,
        "lambda_scale": value,
        "lambda_learnable": saved_learnable,
        "status": "ok"
    }, indent=2))


if __name__ == "__main__":
    main()
