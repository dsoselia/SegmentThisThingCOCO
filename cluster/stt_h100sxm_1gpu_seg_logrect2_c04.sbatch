#!/bin/bash
# Segmentation finetune — 20K steps, 1× H100 SXM, LogRect2 lambda_scale c=0.4 probe (exponent 4.0).
# Auto-resumes from latest checkpoint if one exists.
#
# Submit standalone:  sbatch cluster/stt_h100sxm_1gpu_seg_logrect2_c04.sbatch
# Submit with dep:    sbatch --dependency=afterok:<MAE_JOBID> cluster/stt_h100sxm_1gpu_seg_logrect2_c04.sbatch

#SBATCH --job-name=stt-seg-logrect2-c04
#SBATCH --output=/cmlscratch/dsoselia/logrect2/slurm/seg_logrect2_c04_%j.out
#SBATCH --error=/cmlscratch/dsoselia/logrect2/slurm/seg_logrect2_c04_%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --gres=gpu:h100-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

ROOT=/cmlscratch/dsoselia/logrect2
mkdir -p "$ROOT/slurm"

CONDA_SH="/cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12

cd "$ROOT"
export PYTHONUNBUFFERED=1

ENCODER="$ROOT/artifacts/stt_b_coco_log_rect2_gb10_10k_eb1024_c04_final_encoder.pt"
if [[ ! -f "$ENCODER" ]]; then
  echo "ERROR: Pretrained encoder not found at $ENCODER"
  exit 1
fi

# Detect latest checkpoint for auto-resume
LATEST_CKPT=""
LAST_CKPT_FILE=$(ls -t "$ROOT/runs/log_rect2_seg_gb10_20k_eb2048_c04"/seg_*/last_checkpoint.txt 2>/dev/null | head -1 || true)
if [[ -n "$LAST_CKPT_FILE" && -f "$LAST_CKPT_FILE" ]]; then
  LATEST_CKPT=$(cat "$LAST_CKPT_FILE" | tr -d '[:space:]')
fi

BASE_CONFIG="configs/stt_b_coco_log_rect2_seg_gb10_20k_eb2048_c04.json"
if [[ -n "$LATEST_CKPT" && -f "$LATEST_CKPT" ]]; then
  echo "Resuming from checkpoint: $LATEST_CKPT"
  EFFECTIVE_CONFIG="/tmp/seg_logrect2_c04_resume_${SLURM_JOB_ID}.json"
  python3 -c "
import json
with open('$BASE_CONFIG') as f:
    cfg = json.load(f)
cfg['runtime']['resume_from'] = '$LATEST_CKPT'
with open('$EFFECTIVE_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Wrote resume config to $EFFECTIVE_CONFIG')
"
else
  echo "No checkpoint found — starting from scratch"
  EFFECTIVE_CONFIG="$BASE_CONFIG"
fi

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "config=$EFFECTIVE_CONFIG"
echo "encoder=$ENCODER"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/run_stt.py train-stt --config "$EFFECTIVE_CONFIG" || true

# Self-resubmit if training didn't finish (time limit expiry)
LAST_STEP=$(python3 - <<'PY'
import glob, sys
files = sorted(glob.glob("runs/log_rect2_seg_gb10_20k_eb2048_c04/seg_*/last_checkpoint.txt"))
if not files:
    print(0); sys.exit()
ckpt = open(files[-1]).read().strip()
import torch
try:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(int(state.get("step", 0)))
except Exception:
    print(0)
PY
)
echo "Last saved step: $LAST_STEP"
if [[ "$LAST_STEP" -lt 19999 ]]; then
  NEW_JOB=$(sbatch --parsable cluster/stt_h100sxm_1gpu_seg_logrect2_c04.sbatch)
  echo "Resubmitted: job $NEW_JOB (continuing from step $LAST_STEP)"
else
  echo "Training complete at step $LAST_STEP — not resubmitting"
fi
