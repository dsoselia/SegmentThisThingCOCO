#!/bin/bash
# STT-ring SA-1B 2xH100 launcher.
#
# MAE:
#   sbatch --time=3-00:00:00 --export=ALL,TASK=mae,MODE=long cluster/stt_beacon_stt_sa1b_ddp.sbatch
#
# Segmentation after the paired MAE job:
#   sbatch --dependency=afterok:<mae_job_id> --time=3-00:00:00 \
#     --export=ALL,TASK=seg,MODE=long,SEG_PRETRAINED_MAE_CHECKPOINT=auto,MAE_OUTPUT_DIR=<mae_run_dir> \
#     cluster/stt_beacon_stt_sa1b_ddp.sbatch

#SBATCH --job-name=stt-sa1b-ddp
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

TASK="${TASK:?Set TASK=mae or TASK=seg}"
MODE="${MODE:-smoke}"
EXP="${EXP:-/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/run_stt.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
  else
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
  fi
fi
OFFICIAL_STT_REPO="${OFFICIAL_STT_REPO:-$EXP/src/segment_this_thing_official}"
SOURCE_MANIFEST_DIR="${SOURCE_MANIFEST_DIR:-$EXP/manifests}"

mkdir -p "$EXP/slurm" "$EXP/runs" "$EXP/artifacts" "$EXP/wandb"
cd "$REPO"

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

GPU_WATCH_PID=""
CPU_WATCH_PID=""
DISK_WATCH_PID=""
cleanup_watchers() {
  if [[ -n "${GPU_WATCH_PID:-}" ]]; then
    kill "$GPU_WATCH_PID" 2>/dev/null || true
  fi
  if [[ -n "${CPU_WATCH_PID:-}" ]]; then
    kill "$CPU_WATCH_PID" 2>/dev/null || true
  fi
  if [[ -n "${CPU_WATCH_PID:-}" ]]; then
    wait "$CPU_WATCH_PID" 2>/dev/null || true
  fi
  if [[ -n "${GPU_WATCH_PID:-}" ]]; then
    wait "$GPU_WATCH_PID" 2>/dev/null || true
  fi
  if [[ -n "${DISK_WATCH_PID:-}" ]]; then
    kill "$DISK_WATCH_PID" 2>/dev/null || true
    wait "$DISK_WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup_watchers EXIT

case "$TASK" in
  mae)
    BASE_CONFIG="$REPO/configs/stt_b_sa1b_stt_small_mae_base.json"
    COMMAND=pretrain-mae
    DEFAULT_MICRO_BATCHES="64"
    DEFAULT_EFFECTIVE_BATCH="128"
    DEFAULT_WORKERS="6"
    DEFAULT_GRAD_TASK="mae"
    ;;
  seg)
    BASE_CONFIG="$REPO/configs/stt_b_sa1b_stt_small_seg_base.json"
    COMMAND=train-stt
    DEFAULT_MICRO_BATCHES="8"
    DEFAULT_EFFECTIVE_BATCH="32"
    DEFAULT_WORKERS="4"
    DEFAULT_GRAD_TASK="seg"
    ;;
  *) echo "Unknown TASK=$TASK" >&2; exit 1 ;;
esac

case "$MODE" in
  smoke) NUM_STEPS="${NUM_STEPS:-20}" ;;
  probe) NUM_STEPS="${NUM_STEPS:-100}" ;;
  confirm) NUM_STEPS="${NUM_STEPS:-500}" ;;
  long) NUM_STEPS="${NUM_STEPS:-$([[ "$TASK" == "mae" ]] && echo 500000 || echo 250000)}" ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 1 ;;
esac

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MICRO_BATCHES="${MICRO_BATCHES:-$DEFAULT_MICRO_BATCHES}"
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"
PREFETCH="${PREFETCH:-8}"
IN_ORDER="${IN_ORDER:-true}"
WORKER_PRE_CROP="${WORKER_PRE_CROP:-$([[ "$TASK" == "mae" ]] && echo true || echo false)}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-$DEFAULT_EFFECTIVE_BATCH}"
PROFILE_SUFFIX="${PROFILE_SUFFIX:-_stt_small}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-true}"
RUN_FOVEATOR_CHECK="${RUN_FOVEATOR_CHECK:-true}"
RUN_OFFICIAL_PARITY="${RUN_OFFICIAL_PARITY:-true}"
RUN_GRAD_CHECK="${RUN_GRAD_CHECK:-true}"
GRAD_TASK="${GRAD_TASK:-$DEFAULT_GRAD_TASK}"
GPUS_TAG="${GPUS_TAG:-${NPROC_PER_NODE}xh100}"
WANDB_PROJECT="${WANDB_PROJECT:-SegmentThisThingSTTSA1BSmall}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
MAE_TARGET_BATCH_SCHEDULE="${MAE_TARGET_BATCH_SCHEDULE:-}"
MAE_WARMUP_STEPS="${MAE_WARMUP_STEPS:-10000}"
MAE_VAL_EVERY="${MAE_VAL_EVERY:-1000}"
MAE_VAL_MAX_EXAMPLES="${MAE_VAL_MAX_EXAMPLES:-32}"
SEG_WARMUP_STEPS="${SEG_WARMUP_STEPS:-5000}"
SEG_EVAL_EVERY="${SEG_EVAL_EVERY:-1000}"
SEG_EVAL_MAX_EXAMPLES="${SEG_EVAL_MAX_EXAMPLES:-100}"
SEG_PRETRAINED_MAE_CHECKPOINT="${SEG_PRETRAINED_MAE_CHECKPOINT:-}"
MAE_OUTPUT_DIR="${MAE_OUTPUT_DIR:-}"
if [[ "$MODE" == "long" ]]; then
  SAVE_EVERY="${SAVE_EVERY:-$([[ "$TASK" == "mae" ]] && echo 10000 || echo 5000)}"
  MILESTONE_EVERY="${MILESTONE_EVERY:-50000}"
  LOG_EVERY="${LOG_EVERY:-50}"
  CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-4}"
  GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-10}"
  CPU_MONITOR_INTERVAL="${CPU_MONITOR_INTERVAL:-60}"
  DISK_MONITOR_INTERVAL="${DISK_MONITOR_INTERVAL:-60}"
else
  SAVE_EVERY="${SAVE_EVERY:-$NUM_STEPS}"
  MILESTONE_EVERY="${MILESTONE_EVERY:-$NUM_STEPS}"
  LOG_EVERY="${LOG_EVERY:-10}"
  CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-2}"
  GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
  CPU_MONITOR_INTERVAL="${CPU_MONITOR_INTERVAL:-5}"
  DISK_MONITOR_INTERVAL="${DISK_MONITOR_INTERVAL:-0}"
fi

stage_manifests() {
  local task="$1"
  local stage_base="${SLURM_TMPDIR:-/tmp}"
  local stage_root="${STAGE_ROOT:-$stage_base/foveatedseg-stt-sa1b-manifests-${USER}-${SLURM_JOB_ID:-$$}-${task}}"
  local names=()
  if [[ "$task" == "mae" ]]; then
    names=(sa1b_subset_mae_train_images.jsonl sa1b_subset_mae_val_images.jsonl)
  else
    names=(sa1b_subset_train.jsonl sa1b_subset_val.jsonl)
  fi

  rm -rf "$stage_root"
  mkdir -p "$stage_root"
  for name in "${names[@]}"; do
    local src="$SOURCE_MANIFEST_DIR/$name"
    if [[ ! -f "$src" ]]; then
      echo "Missing manifest: $src" >&2
      exit 1
    fi
    cp "$src" "$stage_root/$name"
    if [[ -f "$src.offsets.npy" ]]; then
      cp "$src.offsets.npy" "$stage_root/$name.offsets.npy"
    fi
    wc -l "$stage_root/$name" >&2
    ls -lh "$stage_root/$name" "$stage_root/$name.offsets.npy" 2>/dev/null >&2 || true
  done
  df -h "$stage_root" >&2
  echo "$stage_root"
}

record_launch_artifacts() {
  local profile="$1"
  local effective_config="$2"
  local artifact_dir="$EXP/artifacts/launch_${SLURM_JOB_ID:-manual}_${profile}"
  mkdir -p "$artifact_dir"
  git rev-parse HEAD > "$artifact_dir/git_rev.txt" 2>/dev/null || true
  git status --short --branch > "$artifact_dir/git_status.txt" 2>/dev/null || true
  git diff > "$artifact_dir/git_diff.patch" 2>/dev/null || true
  cp "$effective_config" "$artifact_dir/effective_config.json"
  cp "$SCRIPT_DIR/$(basename "$0")" "$artifact_dir/launcher.sbatch" 2>/dev/null || true
  tar \
    --exclude=.git \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    -czf "$artifact_dir/source_snapshot.tgz" .
  echo "artifact_dir=$artifact_dir"
}

resolve_mae_checkpoint() {
  if [[ -n "$SEG_PRETRAINED_MAE_CHECKPOINT" && "$SEG_PRETRAINED_MAE_CHECKPOINT" != "auto" ]]; then
    echo "$SEG_PRETRAINED_MAE_CHECKPOINT"
    return
  fi
  if [[ -z "$MAE_OUTPUT_DIR" ]]; then
    echo "Set MAE_OUTPUT_DIR when SEG_PRETRAINED_MAE_CHECKPOINT=auto" >&2
    exit 1
  fi
  python - "$MAE_OUTPUT_DIR" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
ckpt_dir = root / "checkpoints"
if not ckpt_dir.exists():
    raise SystemExit(f"missing checkpoint directory: {ckpt_dir}")

def step(path: Path) -> int:
    match = re.search(r"_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1

candidates = list(ckpt_dir.glob("checkpoint_milestone_*.pt")) + list(ckpt_dir.glob("checkpoint_step_*.pt"))
if not candidates:
    last = ckpt_dir / "last_checkpoint.txt"
    if last.exists():
        value = last.read_text().strip()
        path = Path(value)
        print(path if path.is_absolute() else ckpt_dir / path)
        raise SystemExit(0)
    raise SystemExit(f"no MAE checkpoints found under {ckpt_dir}")
print(max(candidates, key=step))
PY
}

echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname) task=$TASK mode=$MODE nproc_per_node=$NPROC_PER_NODE micro_batches=$MICRO_BATCHES cpus=${SLURM_CPUS_PER_TASK:-unset} workers_per_rank=$WORKERS prefetch=$PREFETCH in_order=$IN_ORDER worker_pre_crop=$WORKER_PRE_CROP effective_batch=$EFFECTIVE_BATCH"
echo "num_steps=$NUM_STEPS save_every=$SAVE_EVERY milestone_every=$MILESTONE_EVERY log_every=$LOG_EVERY checkpoint_keep_last=$CHECKPOINT_KEEP_LAST wandb_project=$WANDB_PROJECT wandb_mode=$WANDB_MODE"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "repo=$REPO commit=$COMMIT official_stt_repo=$OFFICIAL_STT_REPO"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

SOURCE_MANIFESTS=()
if [[ "$TASK" == "mae" ]]; then
  SOURCE_MANIFESTS+=("$SOURCE_MANIFEST_DIR/sa1b_subset_mae_train_images.jsonl" "$SOURCE_MANIFEST_DIR/sa1b_subset_mae_val_images.jsonl")
else
  SOURCE_MANIFESTS+=("$SOURCE_MANIFEST_DIR/sa1b_subset_train.jsonl" "$SOURCE_MANIFEST_DIR/sa1b_subset_val.jsonl")
fi
python scripts/build_jsonl_offsets.py "${SOURCE_MANIFESTS[@]}"
STAGED_MANIFEST_ROOT="$(stage_manifests "$TASK")"
echo "staged_manifest_root=$STAGED_MANIFEST_ROOT"

if [[ "$TASK" == "seg" ]]; then
  RESOLVED_SEG_MAE_CHECKPOINT="$(resolve_mae_checkpoint)"
  if [[ ! -f "$RESOLVED_SEG_MAE_CHECKPOINT" ]]; then
    echo "Resolved MAE checkpoint does not exist: $RESOLVED_SEG_MAE_CHECKPOINT" >&2
    exit 1
  fi
  echo "resolved_seg_pretrained_mae_checkpoint=$RESOLVED_SEG_MAE_CHECKPOINT"
else
  RESOLVED_SEG_MAE_CHECKPOINT=""
fi

if [[ "$RUN_UNIT_TESTS" == "true" ]]; then
  python -m unittest -v tests/test_logrect_learnable.py
fi

for MICRO_BATCH in $MICRO_BATCHES; do
  PRE_CROP_TAG=""
  if [[ "$WORKER_PRE_CROP" == "true" ]]; then
    PRE_CROP_TAG="_workerprecrop"
  fi
  PROFILE="stt_sa1b_ddp_${TASK}_${GPUS_TAG}_mb${MICRO_BATCH}_eb${EFFECTIVE_BATCH}_wpr${WORKERS}_p${PREFETCH}_io${IN_ORDER}_${MODE}${NUM_STEPS}${PRE_CROP_TAG}${PROFILE_SUFFIX}"
  EFFECTIVE_CONFIG="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID:-manual}.json"
  python - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$TASK" "$PROFILE" "$MICRO_BATCH" "$WORKERS" "$PREFETCH" "$IN_ORDER" "$NUM_STEPS" "$STAGED_MANIFEST_ROOT" "$WORKER_PRE_CROP" "$EFFECTIVE_BATCH" "$EXP" "$NPROC_PER_NODE" "$GPUS_TAG" "$SAVE_EVERY" "$MILESTONE_EVERY" "$LOG_EVERY" "$CHECKPOINT_KEEP_LAST" "$WANDB_PROJECT" "$WANDB_RUN_NAME" "$WANDB_MODE" "$MAE_TARGET_BATCH_SCHEDULE" "$MAE_WARMUP_STEPS" "$MAE_VAL_EVERY" "$MAE_VAL_MAX_EXAMPLES" "$SEG_WARMUP_STEPS" "$SEG_EVAL_EVERY" "$SEG_EVAL_MAX_EXAMPLES" "$RESOLVED_SEG_MAE_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

(
    base,
    out,
    task,
    profile,
    micro_batch,
    workers,
    prefetch,
    in_order,
    num_steps,
    staged,
    worker_pre_crop,
    effective_batch,
    exp,
    nproc,
    gpus_tag,
    save_every,
    milestone_every,
    log_every,
    checkpoint_keep_last,
    wandb_project,
    wandb_run_name,
    wandb_mode,
    mae_target_batch_schedule,
    mae_warmup_steps,
    mae_val_every,
    mae_val_max_examples,
    seg_warmup_steps,
    seg_eval_every,
    seg_eval_max_examples,
    seg_pretrained_mae_checkpoint,
) = sys.argv[1:]

config = json.loads(Path(base).read_text())
root = Path(exp)
micro_i = int(micro_batch)
workers_i = int(workers)
prefetch_i = int(prefetch)
steps_i = int(num_steps)
effective_i = int(effective_batch)
save_every_i = int(save_every)
milestone_every_i = int(milestone_every)
log_every_i = int(log_every)
in_order_b = in_order.lower() == "true"
worker_pre_crop_b = worker_pre_crop.lower() == "true"
milestones = list(range(milestone_every_i, steps_i + 1, milestone_every_i)) if milestone_every_i > 0 else []
if steps_i not in milestones:
    milestones.append(steps_i)
run_name = wandb_run_name or f"beacon-ddp-{profile}"

tags = sorted(
    set(
        config["runtime"].get("wandb_tags", [])
        + [
            task,
            "ddp",
            "sa1b",
            "stt-ring",
            "small-schedule",
            gpus_tag,
            "beacon",
            "h100",
            f"ranks-{int(nproc)}",
            f"mb-per-rank-{micro_i}",
            f"workers-per-rank-{workers_i}",
            f"prefetch-{prefetch_i}",
            f"in-order-{int(in_order_b)}",
            f"worker-precrop-{int(worker_pre_crop_b)}",
            f"steps-{steps_i}",
        ]
    )
)
config["runtime"].update(
    {
        "output_dir": str(root / "runs" / profile),
        "profile_name": profile,
        "distributed": True,
        "micro_batch_size": micro_i,
        "num_workers": workers_i,
        "prefetch_factor": prefetch_i,
        "dataloader_in_order": in_order_b,
        "num_steps": steps_i,
        "save_every": save_every_i,
        "milestone_steps": milestones,
        "log_every": log_every_i,
        "checkpoint_keep_last": int(checkpoint_keep_last),
        "wandb_project": wandb_project or config["runtime"].get("wandb_project", "SegmentThisThingSTTSA1BSmall"),
        "wandb_run_name": run_name,
        "wandb_mode": wandb_mode or config["runtime"].get("wandb_mode", "offline"),
        "wandb_dir": str(root / "wandb"),
        "wandb_tags": tags,
    }
)
if task == "mae":
    config["runtime"]["eval_every"] = None
    config["mae"]["train_manifest"] = str(Path(staged) / "sa1b_subset_mae_train_images.jsonl")
    config["mae"]["val_manifest"] = str(Path(staged) / "sa1b_subset_mae_val_images.jsonl")
    config["mae"]["val_every"] = int(mae_val_every) if mae_val_every else None
    config["mae"]["val_max_examples"] = int(mae_val_max_examples)
    config["mae"]["effective_batch_size"] = effective_i
    config["mae"]["target_batch_schedule"] = json.loads(mae_target_batch_schedule) if mae_target_batch_schedule else {"0": effective_i}
    config["mae"]["warmup_steps"] = int(mae_warmup_steps)
    config["mae"]["views_per_image"] = 1
    config["mae"]["worker_pre_crop"] = worker_pre_crop_b
else:
    config["runtime"]["eval_every"] = int(seg_eval_every) if seg_eval_every else None
    config["segmentation"]["train_manifest"] = str(Path(staged) / "sa1b_subset_train.jsonl")
    config["segmentation"]["pretrained_encoder"] = None
    config["segmentation"]["pretrained_mae_checkpoint"] = seg_pretrained_mae_checkpoint or None
    config["segmentation"]["effective_batch_size"] = effective_i
    config["segmentation"]["target_batch_schedule"] = {"0": effective_i}
    config["segmentation"]["warmup_steps"] = int(seg_warmup_steps)
    config["evaluation"]["named_eval_manifests"]["sa1b_subset_val"] = str(Path(staged) / "sa1b_subset_val.jsonl")
    config["evaluation"]["max_examples"] = int(seg_eval_max_examples) if seg_eval_max_examples else None
Path(out).write_text(json.dumps(config, indent=2) + "\n")
print(out)
PY

  echo "DDP_START profile=$PROFILE config=$EFFECTIVE_CONFIG"
  record_launch_artifacts "$PROFILE" "$EFFECTIVE_CONFIG"
  if [[ "$RUN_FOVEATOR_CHECK" == "true" ]]; then
    python scripts/check_foveator_numeric_ranges.py --config "$EFFECTIVE_CONFIG" --max-samples 4 --device cuda
  fi
  if [[ "$RUN_OFFICIAL_PARITY" == "true" ]]; then
    python scripts/check_stt_official_foveator_parity.py --config "$EFFECTIVE_CONFIG" --official-repo "$OFFICIAL_STT_REPO" --device cpu
  fi
  if [[ "$RUN_GRAD_CHECK" == "true" ]]; then
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="$NPROC_PER_NODE" \
      scripts/check_ddp_gradient_flow.py --config "$EFFECTIVE_CONFIG" --task "$GRAD_TASK"
  fi
  GPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID:-manual}_gpu.csv"
  CPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID:-manual}_cpu.txt"
  DISK_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID:-manual}_disk.txt"
  ( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep "$GPU_MONITOR_INTERVAL"; done ) &
  GPU_WATCH_PID=$!
  ( while true; do date -Is >> "$CPU_WATCH"; ps -u "$USER" -o pid,ppid,pcpu,pmem,rss,comm --sort=-pcpu | head -40 >> "$CPU_WATCH"; df -h "$EXP" "$STAGED_MANIFEST_ROOT" >> "$CPU_WATCH"; sleep "$CPU_MONITOR_INTERVAL"; done ) &
  CPU_WATCH_PID=$!
  if [[ "$DISK_MONITOR_INTERVAL" -gt 0 ]] && command -v iostat >/dev/null 2>&1; then
    ( iostat -xm "$DISK_MONITOR_INTERVAL" >> "$DISK_WATCH" ) &
    DISK_WATCH_PID=$!
  fi
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/run_stt.py "$COMMAND" --config "$EFFECTIVE_CONFIG"
  cleanup_watchers
  GPU_WATCH_PID=""
  CPU_WATCH_PID=""
  DISK_WATCH_PID=""
  echo "DDP_COMPLETE profile=$PROFILE config=$EFFECTIVE_CONFIG"
done
