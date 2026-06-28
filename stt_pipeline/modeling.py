from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import segment_this_thing
from segment_this_thing import Foveator, LogRectilinearFoveator

from .config import ModelConfig
from .runtime import get_rng_state, unwrap_model


def _default_pattern_size(model_config: ModelConfig) -> int:
    if model_config.pattern_size is not None:
        return model_config.pattern_size
    return model_config.token_size * model_config.strides[-1] * model_config.grid_sizes[-1]


def build_foveator(model_config: ModelConfig) -> Foveator | LogRectilinearFoveator:
    if model_config.tokenizer_type == "stt_ring":
        return Foveator(
            token_size=model_config.token_size,
            strides=model_config.strides,
            grid_sizes=model_config.grid_sizes,
        )
    if model_config.tokenizer_type == "log_rect_box":
        if model_config.log_rect_axis_bins is None:
            raise ValueError("log_rect_box requires model.log_rect_axis_bins to be set")
        return LogRectilinearFoveator(
            token_size=model_config.token_size,
            pattern_size=_default_pattern_size(model_config),
            axis_bins=model_config.log_rect_axis_bins,
            exponent=model_config.log_rect_exponent,
            center_width=model_config.log_rect_center_width,
            lambda_scale=model_config.log_rect_lambda_scale,
            lambda_learnable=model_config.log_rect_lambda_learnable,
        )
    raise ValueError(f"Unsupported tokenizer_type: {model_config.tokenizer_type}")


def build_model(size: str, foveator: Foveator | LogRectilinearFoveator):
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


def load_segmentation_components(
    model: torch.nn.Module,
    foveator: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load either a legacy STT checkpoint or a joint segmentation/foveator checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if any(key.startswith("segment_model.") for key in state):
        model_state = {
            key.removeprefix("segment_model."): value
            for key, value in state.items()
            if key.startswith("segment_model.")
        }
        foveator_state = {
            key.removeprefix("foveator."): value
            for key, value in state.items()
            if key.startswith("foveator.")
        }
        model.load_state_dict(model_state, strict=strict)
        if foveator_state:
            foveator.load_state_dict(foveator_state, strict=False)
    else:
        model.load_state_dict(state, strict=strict)
    return payload if isinstance(payload, dict) else {"model": payload, "step": 0, "extra": {}}


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
