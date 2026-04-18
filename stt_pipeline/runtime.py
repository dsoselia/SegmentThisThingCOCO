from __future__ import annotations

import json
import os
import random
import resource
import socket
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _optional_import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_autocast_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name, torch.bfloat16)


def configure_torch_runtime(runtime_config) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if runtime_config.sharing_strategy:
        torch.multiprocessing.set_sharing_strategy(runtime_config.sharing_strategy)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(runtime_config.matmul_precision)
    if runtime_config.num_intraop_threads > 0:
        torch.set_num_threads(runtime_config.num_intraop_threads)
    if runtime_config.num_interop_threads > 0:
        torch.set_num_interop_threads(runtime_config.num_interop_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = runtime_config.allow_tf32
        torch.backends.cudnn.allow_tf32 = runtime_config.allow_tf32
        torch.backends.cudnn.benchmark = runtime_config.cudnn_benchmark


def init_distributed(runtime_config) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not runtime_config.distributed or world_size <= 1:
        return DistributedContext(enabled=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    timeout_minutes = max(1, int(getattr(runtime_config, "distributed_timeout_minutes", 120)))
    dist.init_process_group(
        backend=runtime_config.backend,
        timeout=timedelta(minutes=timeout_minutes),
    )
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def distributed_barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def get_device(name: str, ctx: DistributedContext | None = None) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        if ctx is not None and ctx.enabled:
            return torch.device("cuda", ctx.local_rank)
        return torch.device("cuda")
    return torch.device("cpu")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_run_dir(root: str | Path, prefix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ensure_dir(Path(root) / f"{prefix}_{timestamp}")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "eval")
    return run_dir


def infer_run_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def environment_snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": os.sys.version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
        "sharing_strategy": torch.multiprocessing.get_sharing_strategy(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "process_threads": threading.active_count(),
        "rlimit_nofile_soft": get_open_file_limit(),
        "open_file_descriptors": get_open_file_descriptor_count(),
        "torch": torch.__version__,
    }


def save_run_metadata(run_dir: Path, config, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "config": asdict(config),
        "environment": environment_snapshot(),
        "extra": extra or {},
    }
    write_json(run_dir / "metadata.json", payload)


def init_wandb_run(run_dir: Path, config, *, extra: dict[str, Any] | None = None):
    if not getattr(config.runtime, "wandb_enabled", False):
        return None, "disabled"
    wandb = _optional_import_wandb()
    if wandb is None:
        return None, "wandb_not_installed"

    mode = config.runtime.wandb_mode or "offline"
    wandb_dir = Path(config.runtime.wandb_dir) if config.runtime.wandb_dir else (run_dir / "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = mode
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")

    run = wandb.init(
        project=config.runtime.wandb_project,
        entity=config.runtime.wandb_entity,
        name=config.runtime.wandb_run_name or run_dir.name,
        dir=str(wandb_dir),
        mode=mode,
        tags=list(config.runtime.wandb_tags),
        config=asdict(config),
        settings=wandb.Settings(_disable_stats=False),
        reinit="finish_previous",
    )
    if extra:
        for key, value in extra.items():
            run.summary[f"extra/{key}"] = value
    run.summary["run_dir"] = str(run_dir)
    run.summary["wandb_mode"] = mode
    return run, "ok"


def wandb_log(run, payload: dict[str, Any], *, step: int) -> None:
    if run is None:
        return
    run.log(payload, step=step)


def finish_wandb_run(run) -> None:
    if run is None:
        return
    run.finish()


def update_run_status(run_dir: Path, filename: str, lines: list[str]) -> None:
    path = run_dir / filename
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content)


def get_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def gpu_memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_mem_allocated_mb": 0.0,
            "gpu_mem_reserved_mb": 0.0,
        }
    return {
        "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024**2), 2),
        "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved(device) / (1024**2), 2),
    }


def get_open_file_limit() -> int | None:
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        return int(soft_limit)
    except Exception:
        return None


def get_open_file_descriptor_count() -> int | None:
    candidates = ("/proc/self/fd", "/dev/fd")
    for candidate in candidates:
        try:
            return len(os.listdir(candidate))
        except Exception:
            continue
    return None
