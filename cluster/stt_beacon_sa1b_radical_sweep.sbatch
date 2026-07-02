#!/bin/bash
# Examples:
#   sbatch --cpus-per-task=16 --export=ALL,TASK=mae,MODE=smoke,MICRO_BATCHES=64,WORKERS=8,PREFETCH=4 cluster/stt_beacon_sa1b_radical_sweep.sbatch
#   sbatch --cpus-per-task=32 --export=ALL,TASK=mae,MODE=smoke,MICRO_BATCHES=64,WORKERS=16,PREFETCH=8,WORKER_PRE_CROP=true cluster/stt_beacon_sa1b_radical_sweep.sbatch
#   sbatch --gres=gpu:nvidia_h200:1 --cpus-per-task=64 --export=ALL,TASK=mae,MODE=util,MICRO_BATCHES=128,EFFECTIVE_BATCH=128 cluster/stt_beacon_sa1b_radical_sweep.sbatch

#SBATCH --job-name=sa1b-radical
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" WANDB_MODE=offline
GPU_WATCH_PID=""
CPU_WATCH_PID=""
cleanup_watchers() {
  if [[ -n "${GPU_WATCH_PID:-}" ]]; then
    kill "$GPU_WATCH_PID" 2>/dev/null || true
  fi
  if [[ -n "${CPU_WATCH_PID:-}" ]]; then
    kill "$CPU_WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup_watchers EXIT

case "$TASK" in
  mae)
    BASE_CONFIG="$REPO/configs/stt_b_sa1b_logrect_speed_profile_mae_base.json"
    COMMAND=pretrain-mae
    DEFAULT_MICRO_BATCHES="64"
    ;;
  seg)
    BASE_CONFIG="$REPO/configs/stt_b_sa1b_logrect_speed_profile_seg_base.json"
    COMMAND=train-stt
    DEFAULT_MICRO_BATCHES="8"
    ;;
  *) echo "Unknown TASK=$TASK" >&2; exit 1 ;;
esac

case "$MODE" in
  smoke) NUM_STEPS="${NUM_STEPS:-50}" ;;
  probe) NUM_STEPS="${NUM_STEPS:-100}" ;;
  confirm) NUM_STEPS="${NUM_STEPS:-500}" ;;
  util) NUM_STEPS="${NUM_STEPS:-100}" ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 1 ;;
esac

MICRO_BATCHES="${MICRO_BATCHES:-$DEFAULT_MICRO_BATCHES}"
WORKERS="${WORKERS:-8}"
PREFETCH="${PREFETCH:-4}"
IN_ORDER="${IN_ORDER:-true}"
WORKER_PRE_CROP="${WORKER_PRE_CROP:-false}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-}"
PROFILE_SUFFIX="${PROFILE_SUFFIX:-}"

echo "job_id=$SLURM_JOB_ID host=$(hostname) task=$TASK mode=$MODE micro_batches=$MICRO_BATCHES cpus=${SLURM_CPUS_PER_TASK:-unset} workers=$WORKERS prefetch=$PREFETCH in_order=$IN_ORDER worker_pre_crop=$WORKER_PRE_CROP effective_batch=${EFFECTIVE_BATCH:-default}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "repo=$REPO commit=$COMMIT"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

SOURCE_MANIFESTS=(
  "$EXP/manifests/sa1b_subset_train.jsonl"
  "$EXP/manifests/sa1b_subset_val.jsonl"
  "$EXP/manifests/sa1b_subset_mae_train_images.jsonl"
  "$EXP/manifests/sa1b_subset_mae_val_images.jsonl"
)
python scripts/build_jsonl_offsets.py "${SOURCE_MANIFESTS[@]}"
STAGED_MANIFEST_ROOT="$(EXP="$EXP" scripts/stage_sa1b_manifests_node_local.sh)"

python -m unittest -v tests/test_logrect_learnable.py

for MICRO_BATCH in $MICRO_BATCHES; do
  PRE_CROP_TAG=""
  if [[ "$WORKER_PRE_CROP" == "true" ]]; then
    PRE_CROP_TAG="_workerprecrop"
  fi
  EFF_TAG=""
  if [[ -n "$EFFECTIVE_BATCH" ]]; then
    EFF_TAG="_eb${EFFECTIVE_BATCH}"
  fi
  PROFILE="sa1b_radical_${TASK}_mb${MICRO_BATCH}_w${WORKERS}_p${PREFETCH}_io${IN_ORDER}_${MODE}${NUM_STEPS}${PRE_CROP_TAG}${EFF_TAG}${PROFILE_SUFFIX}"
  EFFECTIVE_CONFIG="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}.json"
  python - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$TASK" "$PROFILE" "$MICRO_BATCH" "$WORKERS" "$PREFETCH" "$IN_ORDER" "$NUM_STEPS" "$STAGED_MANIFEST_ROOT" "$WORKER_PRE_CROP" "$EFFECTIVE_BATCH" "$EXP" <<'PY'
import json
import sys
from pathlib import Path

base, out, task, profile, micro_batch, workers, prefetch, in_order, num_steps, staged, worker_pre_crop, effective_batch, exp = sys.argv[1:]
config = json.loads(Path(base).read_text())
root = Path(exp)
micro_i = int(micro_batch)
workers_i = int(workers)
prefetch_i = int(prefetch)
steps_i = int(num_steps)
in_order_b = in_order.lower() == "true"
worker_pre_crop_b = worker_pre_crop.lower() == "true"

config["runtime"].update(
    {
        "output_dir": str(root / "runs" / profile),
        "profile_name": profile,
        "micro_batch_size": micro_i,
        "num_workers": workers_i,
        "prefetch_factor": prefetch_i,
        "dataloader_in_order": in_order_b,
        "num_steps": steps_i,
        "save_every": steps_i,
        "milestone_steps": [steps_i],
        "eval_every": None,
        "wandb_run_name": f"beacon-radical-{profile}",
        "wandb_tags": sorted(
            set(
                config["runtime"].get("wandb_tags", [])
                + [
                    task,
                    "radical-throughput",
                    f"mb-{micro_i}",
                    f"steps-{steps_i}",
                    f"workers-{workers_i}",
                    f"prefetch-{prefetch_i}",
                    f"in-order-{int(in_order_b)}",
                    f"worker-precrop-{int(worker_pre_crop_b)}",
                ]
            )
        ),
    }
)
if effective_batch:
    if task == "mae":
        config["mae"]["effective_batch_size"] = int(effective_batch)
        config["mae"]["target_batch_schedule"] = {"0": int(effective_batch)}
    else:
        config["segmentation"]["effective_batch_size"] = int(effective_batch)
        config["segmentation"]["target_batch_schedule"] = {"0": int(effective_batch)}
if task == "mae":
    config["mae"]["train_manifest"] = str(Path(staged) / "sa1b_subset_mae_train_images.jsonl")
    config["mae"]["val_manifest"] = str(Path(staged) / "sa1b_subset_mae_val_images.jsonl")
    config["mae"]["val_every"] = None
    config["mae"]["worker_pre_crop"] = worker_pre_crop_b
else:
    config["segmentation"]["train_manifest"] = str(Path(staged) / "sa1b_subset_train.jsonl")
    config["evaluation"]["named_eval_manifests"]["sa1b_subset_val"] = str(Path(staged) / "sa1b_subset_val.jsonl")
Path(out).write_text(json.dumps(config, indent=2) + "\n")
print(out)
PY

  echo "RADICAL_START profile=$PROFILE config=$EFFECTIVE_CONFIG"
  python scripts/check_logrect_numeric_ranges.py --config "$EFFECTIVE_CONFIG" --max-samples 4 --device cuda
  GPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}_gpu.csv"
  CPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}_cpu.txt"
  ( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 1; done ) &
  GPU_WATCH_PID=$!
  ( while true; do date -Is >> "$CPU_WATCH"; ps -u "$USER" -o pid,ppid,pcpu,pmem,rss,comm --sort=-pcpu | head -30 >> "$CPU_WATCH"; sleep 5; done ) &
  CPU_WATCH_PID=$!
  python scripts/run_stt.py "$COMMAND" --config "$EFFECTIVE_CONFIG"
  cleanup_watchers
  wait "$GPU_WATCH_PID" "$CPU_WATCH_PID" 2>/dev/null || true
  GPU_WATCH_PID=""
  CPU_WATCH_PID=""
  echo "RADICAL_COMPLETE profile=$PROFILE config=$EFFECTIVE_CONFIG"
done
