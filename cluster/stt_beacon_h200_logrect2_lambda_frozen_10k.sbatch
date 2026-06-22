#!/bin/bash
#SBATCH --job-name=stt-logrect2-lambda-10k
#SBATCH --partition=beacon
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-projects/foveatedseg/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

PROJECT=/beacon-projects/foveatedseg
REPO="$PROJECT/src/SegmentThisThingCOCO-logrect2_parameter_search"
BASE_CONFIG="$REPO/configs/stt_b_beacon_h200_logrect2_lambda_frozen_10k.json"
RUN_ROOT="$PROJECT/runs/logrect2_learnable_lambda/frozen_c10_10k"
ARTIFACT_ROOT="$PROJECT/artifacts/logrect2_learnable_lambda"
mkdir -p "$PROJECT/slurm" "$RUN_ROOT" "$ARTIFACT_ROOT"
cd "$REPO"

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=disabled

LATEST_RUN=$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'mae_*' | sort | tail -1 || true)
if [[ -n "$LATEST_RUN" && -f "$LATEST_RUN/final_encoder.pt" ]]; then
  echo "Run already complete: $LATEST_RUN"
  exit 0
fi

EFFECTIVE_CONFIG="$BASE_CONFIG"
if [[ -n "$LATEST_RUN" && -f "$LATEST_RUN/last_checkpoint.txt" ]]; then
  RESUME_FROM=$(tr -d '[:space:]' < "$LATEST_RUN/last_checkpoint.txt")
  if [[ -f "$RESUME_FROM" ]]; then
    EFFECTIVE_CONFIG="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-resume.json"
    python - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$RESUME_FROM" <<'PY'
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
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 10
  done
) &
GPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$EFFECTIVE_CONFIG"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

set +e
python scripts/run_stt.py pretrain-mae --config "$EFFECTIVE_CONFIG"
TRAIN_STATUS=$?
set -e
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  echo "Training failed with exit status $TRAIN_STATUS"
  exit "$TRAIN_STATUS"
fi

LATEST_RUN=$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'mae_*' | sort | tail -1)
if [[ -f "$LATEST_RUN/final_encoder.pt" ]]; then
  cp "$LATEST_RUN/final_encoder.pt" "$ARTIFACT_ROOT/stt_b_logrect2_lambda_frozen_c10_10k_final_encoder.pt"
  test -f "$LATEST_RUN/checkpoints/checkpoint_milestone_0002500.pt"
  test -f "$LATEST_RUN/checkpoints/checkpoint_milestone_0005000.pt"
  test -f "$LATEST_RUN/checkpoints/checkpoint_milestone_0007500.pt"
  test -f "$LATEST_RUN/checkpoints/checkpoint_milestone_0010000.pt"
  echo "TRAINING_COMPLETE run_dir=$LATEST_RUN"
else
  echo "Graceful interruption detected; requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
fi
