#!/bin/bash
# Segmentation finetune - 20K steps, 1x H200 SXM, learned lambda from MAE 7500-unfreeze run.
# Auto-resumes from latest checkpoint and self-resubmits until completion.

#SBATCH --job-name=stt-seg-lambda75
#SBATCH --output=/cmlscratch/dsoselia/logrect_learnable/SegmentThisThingCOCO/slurm/%x-%j.out
#SBATCH --error=/cmlscratch/dsoselia/logrect_learnable/SegmentThisThingCOCO/slurm/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --exclude=cml36
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

ROOT=/cmlscratch/dsoselia/logrect_learnable/SegmentThisThingCOCO
BASE_CONFIG="$ROOT/configs/stt_b_nexus_h200_logrect2_lambda_seg_from_7500_unfrozen.json"
CHECKPOINT="$ROOT/artifacts/logrect2_learnable_lambda/learnable_from_7500/mae_20260622T235831Z/checkpoints/checkpoint_milestone_0010000.pt"
OUTPUT_ROOT="$ROOT/runs/logrect2_learnable_lambda/seg_from_7500_unfrozen_mae10k"
mkdir -p "$ROOT/slurm" "$OUTPUT_ROOT"

source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate py12

cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: MAE checkpoint not found at $CHECKPOINT"
  exit 1
fi

LATEST_CKPT=""
LAST_CKPT_FILE=$(ls -t "$OUTPUT_ROOT"/seg_*/last_checkpoint.txt 2>/dev/null | head -1 || true)
if [[ -n "$LAST_CKPT_FILE" && -f "$LAST_CKPT_FILE" ]]; then
  LATEST_CKPT=$(tr -d '[:space:]' < "$LAST_CKPT_FILE")
fi

EFFECTIVE_CONFIG="$BASE_CONFIG"
if [[ -n "$LATEST_CKPT" && -f "$LATEST_CKPT" ]]; then
  echo "Resuming from checkpoint: $LATEST_CKPT"
  EFFECTIVE_CONFIG="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-resume.json"
  python - "$BASE_CONFIG" "$EFFECTIVE_CONFIG" "$LATEST_CKPT" <<'PY'
import json
import sys

source, output, checkpoint = sys.argv[1:]
config = json.load(open(source))
config["runtime"]["resume_from"] = checkpoint
with open(output, "w") as handle:
    json.dump(config, handle, indent=2)
print(f"Wrote resume config to {output}")
PY
else
  echo "No checkpoint found - starting from 7500-unfrozen MAE checkpoint"
fi

GPU_WATCH="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 10; done ) &
GPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$EFFECTIVE_CONFIG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

set +e
python scripts/run_stt.py train-stt --config "$EFFECTIVE_CONFIG"
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi

LATEST_RUN=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'seg_*' | sort | tail -1)
if [[ -n "$LATEST_RUN" && -f "$LATEST_RUN/final_model.pt" ]]; then
  echo "TRAINING_COMPLETE run_dir=$LATEST_RUN"
  exit 0
fi

LAST_STEP=$(python - "$OUTPUT_ROOT" <<'PY'
import glob
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
files = sorted(root.glob("seg_*/last_checkpoint.txt"))
if not files:
    print(0)
    raise SystemExit
ckpt = Path(files[-1].read_text().strip())
try:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(int(state.get("step", 0)))
except Exception:
    print(0)
PY
)
echo "Last saved step: $LAST_STEP"
if [[ "$LAST_STEP" -lt 19999 ]]; then
  NEW_JOB=$(sbatch --parsable cluster/stt_nexus_h200_logrect2_lambda_seg_from_7500_unfrozen.sbatch)
  echo "Resubmitted: job $NEW_JOB (continuing from step $LAST_STEP)"
else
  echo "Training complete at step $LAST_STEP - not resubmitting"
fi
