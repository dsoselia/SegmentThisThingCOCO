#!/bin/bash
# Segmentation finetune - 20K steps, 1x Beacon H100, fixed lambda=1.0.

#SBATCH --job-name=stt-logrect-fixed-seg-h100
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-projects/foveatedseg/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

PROJECT=/beacon-projects/foveatedseg
REPO="$PROJECT/src/SegmentThisThingCOCO-logrect2_parameter_search"
BASE_CONFIG="$REPO/configs/stt_b_beacon_h100_logrect_opencl_fixedlambda_seg_from_mae10k.json"
RUN_ROOT="$PROJECT/runs/logrect_opencl_adoption/seg_h100_fixed_lambda_from_mae10k"
ENCODER="$PROJECT/artifacts/logrect_opencl_adoption/stt_b_beacon_h100_logrect_opencl_fixedlambda_mae_10k_final_encoder.pt"
mkdir -p "$PROJECT/slurm" "$RUN_ROOT"
cd "$REPO"

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline

if [[ ! -f "$ENCODER" ]]; then
  echo "ERROR: Pretrained encoder not found at $ENCODER"
  exit 1
fi

EFFECTIVE_CONFIG="$BASE_CONFIG"
LAST_CKPT_FILE=$(ls -t "$RUN_ROOT"/seg_*/last_checkpoint.txt 2>/dev/null | head -1 || true)
if [[ -n "$LAST_CKPT_FILE" && -f "$LAST_CKPT_FILE" ]]; then
  LATEST_CKPT=$(tr -d '[:space:]' < "$LAST_CKPT_FILE")
  if [[ -f "$LATEST_CKPT" ]]; then
    EFFECTIVE_CONFIG="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-resume.json"
    python - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$LATEST_CKPT" <<'PY'
import json, sys
base, output, checkpoint = sys.argv[1:]
with open(base) as handle:
    config = json.load(handle)
config["runtime"]["resume_from"] = checkpoint
with open(output, "w") as handle:
    json.dump(config, handle, indent=2)
print(f"resume_config={output} resume_from={checkpoint}")
PY
  fi
fi

GPU_WATCH="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 10; done ) &
GPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$EFFECTIVE_CONFIG"
echo "encoder=$ENCODER"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python -m unittest -v tests/test_logrect_learnable.py
python scripts/check_logrect_numeric_ranges.py --config "$EFFECTIVE_CONFIG" --max-samples 4 --device cuda
python scripts/run_stt.py train-stt --config "$EFFECTIVE_CONFIG"

LATEST_RUN=$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'seg_*' | sort | tail -1)
if [[ -f "$LATEST_RUN/final_model.pt" ]]; then
  echo "TRAINING_COMPLETE run_dir=$LATEST_RUN"
else
  echo "Training did not produce final_model.pt"
  exit 1
fi
