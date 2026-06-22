#!/bin/bash
#SBATCH --job-name=stt-logrect2-lambda-smoke
#SBATCH --partition=beacon
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-projects/foveatedseg/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

PROJECT=/beacon-projects/foveatedseg
REPO="$PROJECT/src/SegmentThisThingCOCO-logrect2_parameter_search"
CONFIG="$REPO/configs/stt_b_beacon_h200_logrect2_lambda_frozen_smoke.json"
mkdir -p "$PROJECT/slurm" "$PROJECT/runs/logrect2_learnable_lambda" "$PROJECT/artifacts/logrect2_learnable_lambda"
cd "$REPO"

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=disabled

GPU_WATCH="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$CONFIG"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python -m unittest -v tests.test_logrect_learnable
python scripts/preflight_learnable_lambda.py --config "$CONFIG" --device cuda
python scripts/run_stt.py pretrain-mae --config "$CONFIG"

LATEST_RUN=$(find "$PROJECT/runs/logrect2_learnable_lambda/smoke_frozen_c10" -maxdepth 1 -type d -name 'mae_*' | sort | tail -1)
CHECKPOINT="$LATEST_RUN/checkpoints/checkpoint_milestone_0000005.pt"
test -f "$LATEST_RUN/final_encoder.pt"
test -f "$LATEST_RUN/metrics.jsonl"
test -f "$LATEST_RUN/mae_val_metrics.jsonl"
test -f "$CHECKPOINT"
python scripts/validate_mae_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --expected-completed-steps 5 \
  --expected-lambda 1.0
echo "SMOKE_OK run_dir=$LATEST_RUN checkpoint=$CHECKPOINT"
