#!/bin/bash
# Examples:
#   sbatch --export=ALL,TASK=mae,MODE=smoke,MICRO_BATCHES=32,EFFECTIVE_BATCH=64,WORKERS=4,PREFETCH=8 cluster/stt_beacon_sa1b_ddp_sweep.sbatch
#   sbatch --export=ALL,TASK=seg,MODE=smoke,MICRO_BATCHES=4,EFFECTIVE_BATCH=32,WORKERS=4,PREFETCH=8 cluster/stt_beacon_sa1b_ddp_sweep.sbatch
#   sbatch --export=ALL,TASK=mae,MODE=probe,MICRO_BATCHES=64,EFFECTIVE_BATCH=128,WORKERS=6,PREFETCH=8,WORKER_PRE_CROP=true cluster/stt_beacon_sa1b_ddp_sweep.sbatch
#   sbatch --time=3-00:00:00 --export=ALL,TASK=mae,MODE=long,MICRO_BATCHES=64,EFFECTIVE_BATCH=128,WORKERS=6,PREFETCH=8,WORKER_PRE_CROP=true cluster/stt_beacon_sa1b_ddp_sweep.sbatch
#   sbatch --time=3-00:00:00 --export=ALL,TASK=mae,MODE=long,RESUME_FROM=/path/to/checkpoint.pt,VAL_EVERY=1000,VAL_MAX_EXAMPLES=32 cluster/stt_beacon_sa1b_ddp_sweep.sbatch

#SBATCH --job-name=sa1b-ddp
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

TASK="${TASK:?Set TASK=mae or TASK=seg}"
MODE="${MODE:-smoke}"
EXP="${EXP:-/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/run_stt.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
  else
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
  fi
fi
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
  if [[ -n "${DISK_WATCH_PID:-}" ]]; then
    kill "$DISK_WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup_watchers EXIT

case "$TASK" in
  mae)
    BASE_CONFIG="$REPO/configs/stt_b_sa1b_logrect_speed_profile_mae_base.json"
    COMMAND=pretrain-mae
    DEFAULT_MICRO_BATCHES="32"
    DEFAULT_EFFECTIVE_BATCH="64"
    DEFAULT_GRAD_TASK="mae"
    ;;
  seg)
    BASE_CONFIG="$REPO/configs/stt_b_sa1b_logrect_speed_profile_seg_base.json"
    COMMAND=train-stt
    DEFAULT_MICRO_BATCHES="4"
    DEFAULT_EFFECTIVE_BATCH="32"
    DEFAULT_GRAD_TASK="seg"
    ;;
  *) echo "Unknown TASK=$TASK" >&2; exit 1 ;;
esac

case "$MODE" in
  smoke) NUM_STEPS="${NUM_STEPS:-20}" ;;
  probe) NUM_STEPS="${NUM_STEPS:-100}" ;;
  confirm) NUM_STEPS="${NUM_STEPS:-500}" ;;
  util) NUM_STEPS="${NUM_STEPS:-100}" ;;
  long) NUM_STEPS="${NUM_STEPS:-500000}" ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 1 ;;
esac

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_TYPE="${GPU_TYPE:-h100}"
MICRO_BATCHES="${MICRO_BATCHES:-$DEFAULT_MICRO_BATCHES}"
WORKERS="${WORKERS:-4}"
PREFETCH="${PREFETCH:-8}"
IN_ORDER="${IN_ORDER:-true}"
WORKER_PRE_CROP="${WORKER_PRE_CROP:-false}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-$DEFAULT_EFFECTIVE_BATCH}"
PROFILE_SUFFIX="${PROFILE_SUFFIX:-}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-true}"
RUN_NUMERIC_CHECK="${RUN_NUMERIC_CHECK:-true}"
RUN_GRAD_CHECK="${RUN_GRAD_CHECK:-true}"
GRAD_TASK="${GRAD_TASK:-$DEFAULT_GRAD_TASK}"
GPUS_TAG="${GPUS_TAG:-${NPROC_PER_NODE}x${GPU_TYPE}}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
RESUME_FROM="${RESUME_FROM:-}"
FORK_FROM="${FORK_FROM:-}"
VAL_EVERY="${VAL_EVERY:-}"
VAL_MAX_EXAMPLES="${VAL_MAX_EXAMPLES:-}"
TARGET_BATCH_SCHEDULE="${TARGET_BATCH_SCHEDULE:-${MAE_TARGET_BATCH_SCHEDULE:-}}"
WARMUP_STEPS="${WARMUP_STEPS:-${MAE_WARMUP_STEPS:-}}"
BATCH_DOUBLE_EVERY="${BATCH_DOUBLE_EVERY:-${MAE_BATCH_DOUBLE_EVERY:-}}"
VIEWS_PER_IMAGE="${VIEWS_PER_IMAGE:-}"
MAX_SEGMENTS_PER_IMAGE="${MAX_SEGMENTS_PER_IMAGE:-}"
PRETRAINED_ENCODER="${PRETRAINED_ENCODER:-}"
PRETRAINED_MAE_CHECKPOINT="${PRETRAINED_MAE_CHECKPOINT:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
if [[ -n "$RESUME_FROM" && -n "$FORK_FROM" ]]; then
  echo "RESUME_FROM and FORK_FROM are mutually exclusive" >&2
  exit 1
fi
if [[ "$MODE" == "long" ]]; then
  SAVE_EVERY="${SAVE_EVERY:-10000}"
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
  local source_dir="$SOURCE_MANIFEST_DIR"
  local stage_base="${SLURM_TMPDIR:-/tmp}"
  local stage_root="${STAGE_ROOT:-$stage_base/foveatedseg-sa1b-manifests-${USER}-${SLURM_JOB_ID:-$$}-${task}}"
  local names=()
  if [[ "$task" == "mae" ]]; then
    names=(sa1b_subset_mae_train_images.jsonl sa1b_subset_mae_val_images.jsonl)
  else
    names=(sa1b_subset_train.jsonl sa1b_subset_val.jsonl)
  fi

  rm -rf "$stage_root"
  mkdir -p "$stage_root"
  for name in "${names[@]}"; do
    local src="$source_dir/$name"
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

echo "job_id=$SLURM_JOB_ID host=$(hostname) task=$TASK mode=$MODE nproc_per_node=$NPROC_PER_NODE micro_batches=$MICRO_BATCHES cpus=${SLURM_CPUS_PER_TASK:-unset} workers_per_rank=$WORKERS prefetch=$PREFETCH in_order=$IN_ORDER worker_pre_crop=$WORKER_PRE_CROP effective_batch=$EFFECTIVE_BATCH"
echo "num_steps=$NUM_STEPS save_every=$SAVE_EVERY milestone_every=$MILESTONE_EVERY log_every=$LOG_EVERY checkpoint_keep_last=$CHECKPOINT_KEEP_LAST wandb_project=${WANDB_PROJECT:-config-default} wandb_mode=$WANDB_MODE"
echo "resume_from=${RESUME_FROM:-none} fork_from=${FORK_FROM:-none} val_every=${VAL_EVERY:-disabled} val_max_examples=${VAL_MAX_EXAMPLES:-config-default} gpu_type=$GPU_TYPE"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "repo=$REPO commit=$COMMIT"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

SOURCE_MANIFEST_DIR="${SOURCE_DIR:-$EXP/manifests}"
SOURCE_MANIFESTS=()
if [[ "$TASK" == "mae" ]]; then
  SOURCE_MANIFESTS+=(
    "$SOURCE_MANIFEST_DIR/sa1b_subset_mae_train_images.jsonl"
    "$SOURCE_MANIFEST_DIR/sa1b_subset_mae_val_images.jsonl"
  )
else
  SOURCE_MANIFESTS+=(
    "$SOURCE_MANIFEST_DIR/sa1b_subset_train.jsonl"
    "$SOURCE_MANIFEST_DIR/sa1b_subset_val.jsonl"
  )
fi
python scripts/build_jsonl_offsets.py "${SOURCE_MANIFESTS[@]}"
STAGED_MANIFEST_ROOT="$(stage_manifests "$TASK")"
echo "staged_manifest_root=$STAGED_MANIFEST_ROOT"

if [[ "$RUN_UNIT_TESTS" == "true" ]]; then
  python -m unittest -v tests/test_logrect_learnable.py
fi

for MICRO_BATCH in $MICRO_BATCHES; do
  PRE_CROP_TAG=""
  if [[ "$WORKER_PRE_CROP" == "true" ]]; then
    PRE_CROP_TAG="_workerprecrop"
  fi
  PROFILE="sa1b_ddp_${TASK}_${GPUS_TAG}_mb${MICRO_BATCH}_eb${EFFECTIVE_BATCH}_wpr${WORKERS}_p${PREFETCH}_io${IN_ORDER}_${MODE}${NUM_STEPS}${PRE_CROP_TAG}${PROFILE_SUFFIX}"
  EFFECTIVE_CONFIG="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}.json"
  python scripts/build_sa1b_ddp_config.py \
    --base "$BASE_CONFIG" \
    --output "$EFFECTIVE_CONFIG" \
    --task "$TASK" \
    --profile "$PROFILE" \
    --experiment-root "$EXP" \
    --staged-manifest-root "$STAGED_MANIFEST_ROOT" \
    --micro-batch "$MICRO_BATCH" \
    --effective-batch "$EFFECTIVE_BATCH" \
    --workers "$WORKERS" \
    --prefetch "$PREFETCH" \
    --in-order "$IN_ORDER" \
    --worker-pre-crop "$WORKER_PRE_CROP" \
    --num-steps "$NUM_STEPS" \
    --save-every "$SAVE_EVERY" \
    --milestone-every "$MILESTONE_EVERY" \
    --log-every "$LOG_EVERY" \
    --checkpoint-keep-last "$CHECKPOINT_KEEP_LAST" \
    --nproc "$NPROC_PER_NODE" \
    --gpus-tag "$GPUS_TAG" \
    --gpu-type "$GPU_TYPE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --resume-from "$RESUME_FROM" \
    --fork-from "$FORK_FROM" \
    --val-every "$VAL_EVERY" \
    --val-max-examples "$VAL_MAX_EXAMPLES" \
    --target-batch-schedule "$TARGET_BATCH_SCHEDULE" \
    --batch-double-every "$BATCH_DOUBLE_EVERY" \
    --warmup-steps "$WARMUP_STEPS" \
    --views-per-image "$VIEWS_PER_IMAGE" \
    --max-segments-per-image "$MAX_SEGMENTS_PER_IMAGE" \
    --pretrained-encoder "$PRETRAINED_ENCODER" \
    --pretrained-mae-checkpoint "$PRETRAINED_MAE_CHECKPOINT" \
    --init-checkpoint "$INIT_CHECKPOINT"

  echo "DDP_START profile=$PROFILE config=$EFFECTIVE_CONFIG"
  record_launch_artifacts "$PROFILE" "$EFFECTIVE_CONFIG"
  if [[ "$RUN_NUMERIC_CHECK" == "true" ]]; then
    python scripts/check_logrect_numeric_ranges.py --config "$EFFECTIVE_CONFIG" --max-samples 4 --device cuda
  fi
  if [[ "$RUN_GRAD_CHECK" == "true" ]]; then
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="$NPROC_PER_NODE" \
      scripts/check_ddp_gradient_flow.py --config "$EFFECTIVE_CONFIG" --task "$GRAD_TASK"
  fi
  GPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}_gpu.csv"
  CPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}_cpu.txt"
  DISK_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}_disk.txt"
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
  wait "$GPU_WATCH_PID" "$CPU_WATCH_PID" 2>/dev/null || true
  GPU_WATCH_PID=""
  CPU_WATCH_PID=""
  DISK_WATCH_PID=""
  echo "DDP_COMPLETE profile=$PROFILE config=$EFFECTIVE_CONFIG"
done
