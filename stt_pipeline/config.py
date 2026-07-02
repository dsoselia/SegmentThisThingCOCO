from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    size: str = "b"
    token_size: int = 16
    tokenizer_type: str = "stt_ring"
    strides: list[int] = field(default_factory=lambda: [1, 2, 4, 6, 8])
    grid_sizes: list[int] = field(default_factory=lambda: [4, 4, 6, 8, 10])
    pattern_size: int | None = None
    log_rect_axis_bins: int | None = None
    log_rect_exponent: float = 4.0
    log_rect_center_width: int | None = None
    log_rect_lambda_scale: float = 1.0
    log_rect_lambda_learnable: bool = False
    log_rect_lambda_unfreeze_step: int = 0
    log_rect_lambda_lr: float = 1e-3
    num_masks: int = 3


@dataclass
class RuntimeConfig:
    output_dir: str = "runs/stt"
    profile_name: str = "default"
    seed: int = 1337
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    matmul_precision: str = "high"
    sharing_strategy: str = "file_system"
    multiprocessing_context: str | None = None
    num_intraop_threads: int = 0
    num_interop_threads: int = 0
    distributed: bool = True
    backend: str = "nccl"
    distributed_timeout_minutes: int = 120
    micro_batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    pin_memory_device: str | None = None
    persistent_workers: bool = True
    prefetch_factor: int = 4
    dataloader_in_order: bool = True
    dataloader_timeout_s: int = 0
    save_every: int = 1000
    milestone_steps: list[int] = field(default_factory=list)
    log_every: int = 50
    eval_every: int | None = None
    resume_from: Optional[str] = None
    fork_from: Optional[str] = None
    checkpoint_keep_last: int = 2
    save_optimizer_state: bool = True
    save_rng_state: bool = True
    run_status_filename: str = "RUN_STATUS.md"
    wandb_enabled: bool = False
    wandb_project: str = "SegmentThisLogRectilinearZaratan"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "offline"
    wandb_dir: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    num_steps: int = 1000


@dataclass
class MAEConfig:
    enabled: bool = True
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    val_every: int | None = None
    val_max_examples: int = 128
    val_views_per_image: int = 1
    val_jitter_radius: int = 0
    val_mask_seed: int = 1234
    lr: float = 2 ** -13
    weight_decay: float = 1e-3
    warmup_steps: int = 10_000
    mask_ratio: float = 0.75
    margin: int = 256
    views_per_image: int = 2
    worker_pre_crop: bool = False
    worker_pre_crop_jitter_radius: int = 8
    effective_batch_size: int = 1024
    target_batch_schedule: Dict[str, int] = field(
        default_factory=lambda: {"0": 1024, "100000": 2048, "200000": 4096, "300000": 8192, "400000": 16384}
    )


@dataclass
class SegmentationConfig:
    enabled: bool = True
    train_manifest: Optional[str] = None
    pretrained_encoder: Optional[str] = None
    pretrained_mae_checkpoint: Optional[str] = None
    init_checkpoint: Optional[str] = None
    lr: float = 2 ** -16
    weight_decay: float = 1e-3
    warmup_steps: int = 5_000
    margin: int = 256
    max_segments_per_image: int = 16
    sample_multiple_segments_per_image: bool = False
    effective_batch_size: int = 2048
    target_batch_schedule: Dict[str, int] = field(
        default_factory=lambda: {"0": 2048, "50000": 4096, "100000": 8192, "150000": 16384, "200000": 32768}
    )
    focal_weight: float = 20.0
    dice_weight: float = 1.0
    iou_weight: float = 0.01
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    prompt_noise_std: float = 0.0


@dataclass
class EvalConfig:
    eval_manifest: Optional[str] = None
    named_eval_manifests: Dict[str, str] = field(default_factory=dict)
    max_examples: Optional[int] = None
    threshold: float = 0.5
    upsample_small_images: bool = True


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    checkpoint: Optional[str] = None
    warmup_steps: int = 10
    measure_steps: int = 50
    image_size: int = 1536


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    mae: MAEConfig = field(default_factory=MAEConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    assumptions: list[str] = field(default_factory=list)


def _construct_dataclass(cls, payload: Dict[str, Any]):
    kwargs = {}
    for name in cls.__dataclass_fields__:
        if name not in payload:
            continue
        kwargs[name] = payload[name]
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text())
    config = ExperimentConfig(
        model=_construct_dataclass(ModelConfig, raw.get("model", {})),
        runtime=_construct_dataclass(RuntimeConfig, raw.get("runtime", {})),
        mae=_construct_dataclass(MAEConfig, raw.get("mae", {})),
        segmentation=_construct_dataclass(SegmentationConfig, raw.get("segmentation", {})),
        evaluation=_construct_dataclass(EvalConfig, raw.get("evaluation", {})),
        benchmark=_construct_dataclass(BenchmarkConfig, raw.get("benchmark", {})),
        assumptions=raw.get("assumptions", []),
    )
    if config.runtime.resume_from and config.runtime.fork_from:
        raise ValueError("runtime.resume_from and runtime.fork_from are mutually exclusive")
    return config
