#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline.config import load_config
from stt_pipeline.modeling import build_foveator


def _load_official_foveator_class(official_repo: Path):
    path = official_repo / "segment_this_thing" / "foveation.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("official_stt_foveation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Foveator


def main() -> None:
    parser = argparse.ArgumentParser("Compare local STT ring foveator against the official STT Foveator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.model.tokenizer_type != "stt_ring":
        print(json.dumps({"skipped": True, "reason": f"tokenizer_type={config.model.tokenizer_type}"}))
        return

    official_cls = _load_official_foveator_class(args.official_repo.resolve())
    local = build_foveator(config.model).to(args.device)
    official = official_cls(
        token_size=config.model.token_size,
        strides=config.model.strides,
        grid_sizes=config.model.grid_sizes,
    ).to(args.device)

    if local.get_pattern_bounds_size() != official.get_pattern_bounds_size():
        raise RuntimeError((local.get_pattern_bounds_size(), official.get_pattern_bounds_size()))
    if local.get_num_tokens() != official.get_num_tokens():
        raise RuntimeError((local.get_num_tokens(), official.get_num_tokens()))
    if not torch.equal(local.token_corner_indices.cpu(), official.token_corner_indices.cpu()):
        raise RuntimeError("token_corner_indices differ from official STT Foveator")
    if not torch.equal(local.token_strides.cpu(), official.token_strides.cpu()):
        raise RuntimeError("token_strides differ from official STT Foveator")

    pattern = local.get_pattern_bounds_size()
    image = torch.zeros((3, pattern, pattern), dtype=torch.uint8, device=args.device)
    image[0] = torch.arange(pattern, device=args.device, dtype=torch.uint8).view(1, -1)
    image[1] = torch.arange(pattern, device=args.device, dtype=torch.uint8).view(-1, 1)
    image[2].fill_(127)
    local_tokens = local.extract_foveated_image(image)
    official_tokens = official.extract_foveated_image(image)
    if not torch.equal(local_tokens.cpu(), official_tokens.cpu()):
        max_abs = (local_tokens.float() - official_tokens.float()).abs().max().item()
        raise RuntimeError(f"official parity token mismatch: max_abs={max_abs}")

    print(
        json.dumps(
            {
                "official_repo": str(args.official_repo.resolve()),
                "tokenizer_type": config.model.tokenizer_type,
                "pattern_size": pattern,
                "num_tokens": local.get_num_tokens(),
                "num_tokens_by_level": local.num_tokens_by_level,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
