#!/bin/bash
# Submit examples:
#   sbatch --export=ALL,TASK=mae,MODE=smoke cluster/stt_beacon_h100_sa1b_microbatch_sweep.sbatch
#   sbatch --export=ALL,TASK=seg,MODE=smoke cluster/stt_beacon_h100_sa1b_microbatch_sweep.sbatch
#   sbatch --export=ALL,TASK=mae,MODE=confirm,MICRO_BATCHES=32 cluster/stt_beacon_h100_sa1b_microbatch_sweep.sbatch

#SBATCH --job-name=sa1b-mb-sweep
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

TASK="${TASK:?Set TASK=mae or TASK=seg}"
MODE="${MODE:-smoke}"
EXP=/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments
REPO="$EXP/src/SegmentThisThingCOCO"
mkdir -p "$EXP/slurm" "$EXP/runs" "$EXP/artifacts" "$EXP/wandb"
cd "$REPO"

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline
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
    DEFAULT_MICRO_BATCHES="16 32 64"
    ;;
  seg)
    BASE_CONFIG="$REPO/configs/stt_b_sa1b_logrect_speed_profile_seg_base.json"
    COMMAND=train-stt
    DEFAULT_MICRO_BATCHES="2 4 8"
    ;;
  *) echo "Unknown TASK=$TASK" >&2; exit 1 ;;
esac

case "$MODE" in
  smoke) NUM_STEPS="${NUM_STEPS:-50}" ;;
  confirm) NUM_STEPS="${NUM_STEPS:-500}" ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 1 ;;
esac

MICRO_BATCHES="${MICRO_BATCHES:-$DEFAULT_MICRO_BATCHES}"
WORKERS="${WORKERS:-8}"
PREFETCH="${PREFETCH:-4}"
IN_ORDER="${IN_ORDER:-true}"

echo "job_id=$SLURM_JOB_ID host=$(hostname) task=$TASK mode=$MODE micro_batches=$MICRO_BATCHES cpus=${SLURM_CPUS_PER_TASK:-unset}"
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
  PROFILE="sa1b_microbatch_${TASK}_mb${MICRO_BATCH}_${MODE}${NUM_STEPS}"
  EFFECTIVE_CONFIG="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}.json"
  python - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$TASK" "$PROFILE" "$MICRO_BATCH" "$WORKERS" "$PREFETCH" "$IN_ORDER" "$NUM_STEPS" "$STAGED_MANIFEST_ROOT" <<'PY'
import json
import sys
from pathlib import Path

base, out, task, profile, micro_batch, workers, prefetch, in_order, num_steps, staged = sys.argv[1:]
config = json.loads(Path(base).read_text())
root = Path("/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments")
micro_i = int(micro_batch)
workers_i = int(workers)
prefetch_i = int(prefetch)
steps_i = int(num_steps)
in_order_b = in_order.lower() == "true"

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
        "wandb_run_name": f"beacon-h100-{profile}",
        "wandb_tags": sorted(
            set(
                config["runtime"].get("wandb_tags", [])
                + [
                    task,
                    "microbatch",
                    f"mb-{micro_i}",
                    f"steps-{steps_i}",
                    f"workers-{workers_i}",
                    f"prefetch-{prefetch_i}",
                    f"in-order-{int(in_order_b)}",
                ]
            )
        ),
    }
)
if task == "mae":
    config["mae"]["train_manifest"] = str(Path(staged) / "sa1b_subset_mae_train_images.jsonl")
    config["mae"]["val_manifest"] = str(Path(staged) / "sa1b_subset_mae_val_images.jsonl")
    config["mae"]["val_every"] = None
else:
    config["segmentation"]["train_manifest"] = str(Path(staged) / "sa1b_subset_train.jsonl")
    config["evaluation"]["named_eval_manifests"]["sa1b_subset_val"] = str(Path(staged) / "sa1b_subset_val.jsonl")
Path(out).write_text(json.dumps(config, indent=2) + "\n")
print(out)
PY

  echo "MICROBATCH_START profile=$PROFILE config=$EFFECTIVE_CONFIG"
  python scripts/check_logrect_numeric_ranges.py --config "$EFFECTIVE_CONFIG" --max-samples 4 --device cuda
  GPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}_gpu.csv"
  CPU_WATCH="$EXP/slurm/${PROFILE}_${SLURM_JOB_ID}_cpu.txt"
  ( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 5; done ) &
  GPU_WATCH_PID=$!
  ( while true; do date -Is >> "$CPU_WATCH"; ps -u "$USER" -o pid,ppid,pcpu,pmem,rss,comm --sort=-pcpu | head -20 >> "$CPU_WATCH"; sleep 10; done ) &
  CPU_WATCH_PID=$!
  python scripts/run_stt.py "$COMMAND" --config "$EFFECTIVE_CONFIG"
  cleanup_watchers
  wait "$GPU_WATCH_PID" "$CPU_WATCH_PID" 2>/dev/null || true
  GPU_WATCH_PID=""
  CPU_WATCH_PID=""
  echo "MICROBATCH_COMPLETE profile=$PROFILE config=$EFFECTIVE_CONFIG"
done
