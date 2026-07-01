#!/bin/bash
# Segmentation finetune - 20K steps, 1x Nexus H200 SXM, fixed lambda=1.0.

#SBATCH --job-name=stt-logrect-fixed-seg
#SBATCH --output=/cmlscratch/dsoselia/logrect_opencl_adoption/slurm/%x-%j.out
#SBATCH --error=/cmlscratch/dsoselia/logrect_opencl_adoption/slurm/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --nodes=1
#SBATCH --nodelist=cml35
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

ROOT=/cmlscratch/dsoselia/logrect_opencl_adoption
BASE_CONFIG="$ROOT/configs/stt_b_nexus_h200_logrect_opencl_fixedlambda_seg_from_mae10k.json"
RUN_ROOT="$ROOT/runs/seg_h200_fixed_lambda_from_mae10k"
ENCODER="$ROOT/artifacts/stt_b_nexus_h200_logrect_opencl_fixedlambda_mae_10k_final_encoder.pt"
mkdir -p "$ROOT/slurm" "$RUN_ROOT"
source ~/.bashrc
conda activate py12
cd "$ROOT"
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
    EFFECTIVE_CONFIG="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-resume.json"
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

GPU_WATCH="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 10; done ) &
GPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$EFFECTIVE_CONFIG"
echo "encoder=$ENCODER"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python -m unittest discover -s tests -p 'test_logrect_learnable.py' -v
python scripts/check_logrect_numeric_ranges.py --config "$EFFECTIVE_CONFIG" --max-samples 4 --device cuda
python scripts/run_stt.py train-stt --config "$EFFECTIVE_CONFIG"

LATEST_RUN=$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'seg_*' | sort | tail -1)
if [[ -f "$LATEST_RUN/final_model.pt" ]]; then
  echo "TRAINING_COMPLETE run_dir=$LATEST_RUN"
else
  echo "Training did not produce final_model.pt"
  exit 1
fi
