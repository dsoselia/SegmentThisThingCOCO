#!/bin/bash
# MAE pretrain — 10K steps, 1× H100 SXM, LogRect2 lambda_scale c=0.4 probe (exponent 4.0).
# After completion, copies final_encoder.pt to artifacts/ for seg finetune.
#
# Submit:  sbatch cluster/stt_h100sxm_1gpu_mae_logrect2_c04.sbatch

#SBATCH --job-name=stt-mae-logrect2-c04
#SBATCH --output=/cmlscratch/dsoselia/logrect2/slurm/mae_logrect2_c04_%j.out
#SBATCH --error=/cmlscratch/dsoselia/logrect2/slurm/mae_logrect2_c04_%j.err
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
mkdir -p "$ROOT/slurm" "$ROOT/artifacts"

CONDA_SH="/cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12

cd "$ROOT"
export PYTHONUNBUFFERED=1

CONFIG="configs/stt_b_coco_log_rect2_mae_gb10_10k_eb1024_c04.json"
ARTIFACT="$ROOT/artifacts/stt_b_coco_log_rect2_gb10_10k_eb1024_c04_final_encoder.pt"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "config=$CONFIG"
echo "artifact=$ARTIFACT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/run_stt.py pretrain-mae --config "$CONFIG"

# Copy final encoder to known artifact path for seg finetune
ENCODER=$(find "$ROOT/runs/log_rect2_mae_gb10_10k_eb1024_c04" -name "final_encoder.pt" 2>/dev/null | head -1 || true)
if [[ -n "$ENCODER" && -f "$ENCODER" ]]; then
  cp "$ENCODER" "$ARTIFACT"
  echo "Copied encoder artifact: $ENCODER -> $ARTIFACT"
else
  echo "WARNING: Could not find final_encoder.pt to copy to artifacts/"
fi
