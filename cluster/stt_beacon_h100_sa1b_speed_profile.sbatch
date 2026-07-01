#!/bin/bash
# Submit with:
#   sbatch --cpus-per-task=8  --export=ALL,TASK=mae,VARIANT=baseline cluster/stt_beacon_h100_sa1b_speed_profile.sbatch
#   sbatch --cpus-per-task=16 --export=ALL,TASK=mae,VARIANT=candidate_a cluster/stt_beacon_h100_sa1b_speed_profile.sbatch
#   sbatch --cpus-per-task=16 --export=ALL,TASK=mae,VARIANT=candidate_b cluster/stt_beacon_h100_sa1b_speed_profile.sbatch

#SBATCH --job-name=sa1b-speed-profile
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

TASK="${TASK:?Set TASK=mae or TASK=seg}"
VARIANT="${VARIANT:?Set VARIANT=baseline, candidate_a, or candidate_b}"
EXP=/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments
REPO="$EXP/src/SegmentThisThingCOCO"
mkdir -p "$EXP/slurm" "$EXP/runs" "$EXP/artifacts" "$EXP/wandb"
cd "$REPO"

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline

case "$TASK" in
  mae) BASE_CONFIG="$REPO/configs/stt_b_sa1b_logrect_speed_profile_mae_base.json"; COMMAND=pretrain-mae ;;
  seg) BASE_CONFIG="$REPO/configs/stt_b_sa1b_logrect_speed_profile_seg_base.json"; COMMAND=train-stt ;;
  *) echo "Unknown TASK=$TASK" >&2; exit 1 ;;
esac

case "$VARIANT" in
  baseline) WORKERS=4; PREFETCH=2; IN_ORDER=true ;;
  candidate_a) WORKERS=8; PREFETCH=4; IN_ORDER=true ;;
  candidate_b) WORKERS=12; PREFETCH=4; IN_ORDER=false ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 1 ;;
esac

echo "job_id=$SLURM_JOB_ID host=$(hostname) task=$TASK variant=$VARIANT cpus=${SLURM_CPUS_PER_TASK:-unset}"
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

EFFECTIVE_CONFIG="$EXP/slurm/sa1b_speed_profile_${TASK}_${VARIANT}_${SLURM_JOB_ID}.json"
python - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$TASK" "$VARIANT" "$WORKERS" "$PREFETCH" "$IN_ORDER" "$STAGED_MANIFEST_ROOT" <<'PY'
import json
import sys
from pathlib import Path

base, out, task, variant, workers, prefetch, in_order, staged = sys.argv[1:]
config = json.loads(Path(base).read_text())
workers_i = int(workers)
prefetch_i = int(prefetch)
in_order_b = in_order.lower() == "true"
root = Path("/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments")
profile = f"sa1b_speed_profile_{task}_{variant}"
config["runtime"].update(
    {
        "output_dir": str(root / "runs" / profile),
        "profile_name": profile,
        "num_workers": workers_i,
        "prefetch_factor": prefetch_i,
        "dataloader_in_order": in_order_b,
        "wandb_run_name": f"beacon-h100-{profile}",
        "wandb_tags": sorted(set(config["runtime"].get("wandb_tags", []) + [task, variant, f"workers-{workers_i}", f"prefetch-{prefetch_i}", f"in-order-{int(in_order_b)}"])),
    }
)
if task == "mae":
    config["mae"]["train_manifest"] = str(Path(staged) / "sa1b_subset_mae_train_images.jsonl")
    config["mae"]["val_manifest"] = str(Path(staged) / "sa1b_subset_mae_val_images.jsonl")
else:
    config["segmentation"]["train_manifest"] = str(Path(staged) / "sa1b_subset_train.jsonl")
    config["evaluation"]["named_eval_manifests"]["sa1b_subset_val"] = str(Path(staged) / "sa1b_subset_val.jsonl")
Path(out).write_text(json.dumps(config, indent=2) + "\n")
print(out)
PY

GPU_WATCH="$EXP/slurm/sa1b_speed_profile_${TASK}_${VARIANT}_${SLURM_JOB_ID}_gpu.csv"
CPU_WATCH="$EXP/slurm/sa1b_speed_profile_${TASK}_${VARIANT}_${SLURM_JOB_ID}_cpu.txt"
( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 10; done ) &
GPU_WATCH_PID=$!
( while true; do date -Is >> "$CPU_WATCH"; ps -u "$USER" -o pid,ppid,pcpu,pmem,rss,comm --sort=-pcpu | head -20 >> "$CPU_WATCH"; sleep 10; done ) &
CPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" "$CPU_WATCH_PID" 2>/dev/null || true' EXIT

python -m unittest -v tests/test_logrect_learnable.py
python scripts/check_logrect_numeric_ranges.py --config "$EFFECTIVE_CONFIG" --max-samples 4 --device cuda
python scripts/run_stt.py "$COMMAND" --config "$EFFECTIVE_CONFIG"

echo "PROFILE_COMPLETE task=$TASK variant=$VARIANT config=$EFFECTIVE_CONFIG"
