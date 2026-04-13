from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import segment_this_thing
from segment_this_thing import Foveator

from .config import ModelConfig
from .runtime import get_rng_state, unwrap_model


def build_foveator(model_config: ModelConfig) -> Foveator:
    if model_config.tokenizer_type != "stt_ring":
        raise ValueError(f"Unsupported tokenizer_type for baseline pipeline: {model_config.tokenizer_type}")
    return Foveator(token_size=model_config.token_size, strides=model_config.strides, grid_sizes=model_config.grid_sizes)


def build_model(size: str, foveator: Foveator):
    builder = getattr(segment_this_thing, f"build_segment_this_thing_{size}")
    return builder(num_tokens=foveator.get_num_tokens(), token_size=foveator.token_size)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    strict: bool = True,
    restore_training_state: bool = False,
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        unwrap_model(model).load_state_dict(state["model"], strict=strict)
        if restore_training_state:
            if optimizer is not None and "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if scaler is not None and "scaler" in state and state["scaler"] is not None:
                scaler.load_state_dict(state["scaler"])
        return state

    unwrap_model(model).load_state_dict(state, strict=strict)
    return {"model": state, "step": 0, "extra": {}}


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    step: int = 0,
    config=None,
    extra: dict[str, Any] | None = None,
    save_optimizer_state: bool = True,
    save_rng_state: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": unwrap_model(model).state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if config is not None:
        payload["config"] = asdict(config)
    if optimizer is not None and save_optimizer_state:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if save_rng_state:
        payload["rng_state"] = get_rng_state()
    torch.save(payload, path)
    return payload
