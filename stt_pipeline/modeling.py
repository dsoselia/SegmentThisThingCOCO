from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import segment_this_thing
from segment_this_thing import Foveator, LearnableLogRectilinearFoveator, LogRectilinearFoveator

from .config import ModelConfig
from .runtime import get_rng_state, unwrap_model


def _default_pattern_size(model_config: ModelConfig) -> int:
    if model_config.pattern_size is not None:
        return model_config.pattern_size
    return model_config.token_size * model_config.strides[-1] * model_config.grid_sizes[-1]


def build_foveator(model_config: ModelConfig) -> Foveator | LogRectilinearFoveator | LearnableLogRectilinearFoveator:
    if model_config.tokenizer_type == "stt_ring":
        return Foveator(
            token_size=model_config.token_size,
            strides=model_config.strides,
            grid_sizes=model_config.grid_sizes,
        )
    if model_config.tokenizer_type == "log_rect_box":
        if model_config.log_rect_axis_bins is None:
            raise ValueError("log_rect_box requires model.log_rect_axis_bins to be set")
        if model_config.log_rect_learnable:
            return LearnableLogRectilinearFoveator(
                token_size=model_config.token_size,
                pattern_size=_default_pattern_size(model_config),
                axis_bins=model_config.log_rect_axis_bins,
                exponent_init=model_config.log_rect_exponent,
                center_width=model_config.log_rect_center_width,
                learn_scale=model_config.log_rect_learn_scale,
                exponent_min=model_config.log_rect_exponent_min,
                exponent_max=model_config.log_rect_exponent_max,
                scale_min=model_config.log_rect_scale_min,
                scale_max=model_config.log_rect_scale_max,
            )
        return LogRectilinearFoveator(
            token_size=model_config.token_size,
            pattern_size=_default_pattern_size(model_config),
            axis_bins=model_config.log_rect_axis_bins,
            exponent=model_config.log_rect_exponent,
            center_width=model_config.log_rect_center_width,
        )
    raise ValueError(f"Unsupported tokenizer_type: {model_config.tokenizer_type}")


def build_model(size: str, foveator: Foveator | LogRectilinearFoveator | LearnableLogRectilinearFoveator):
    builder = getattr(segment_this_thing, f"build_segment_this_thing_{size}")
    return builder(num_tokens=foveator.get_num_tokens(), token_size=foveator.token_size)


def attach_foveator_if_learnable(model: torch.nn.Module, foveator: torch.nn.Module) -> torch.nn.Module:
    if any(param.requires_grad for param in foveator.parameters()):
        model.foveator = foveator
    return model


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


def load_foveator_checkpoint(
    foveator: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    prefix = "foveator."
    foveator_state = {
        key[len(prefix) :]: value
        for key, value in model_state.items()
        if key.startswith(prefix)
    }
    if not foveator_state:
        raise ValueError(f"No {prefix} parameters found in checkpoint: {checkpoint_path}")
    foveator.load_state_dict(foveator_state, strict=strict)
    return state if isinstance(state, dict) else {"model": state, "step": 0, "extra": {}}


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
