#!/bin/bash
# MAE pretrain - 10K steps, 1x Nexus H200 SXM, fixed lambda=1.0 OpenCL-logrect geometry.

#SBATCH --job-name=stt-logrect-fixed-mae
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
CONFIG="$ROOT/configs/stt_b_nexus_h200_logrect_opencl_fixedlambda_mae_10k.json"
ARTIFACT_ROOT="$ROOT/artifacts"
ARTIFACT="$ARTIFACT_ROOT/stt_b_nexus_h200_logrect_opencl_fixedlambda_mae_10k_final_encoder.pt"
RUN_ROOT="$ROOT/runs/mae_h200_10k_fixed_lambda"

mkdir -p "$ROOT/slurm" "$ARTIFACT_ROOT" "$RUN_ROOT"
source ~/.bashrc
conda activate py12
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$CONFIG"
echo "artifact=$ARTIFACT"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python -m unittest discover -s tests -p 'test_logrect_learnable.py' -v
python scripts/check_logrect_numeric_ranges.py --config "$CONFIG" --max-samples 4 --device cuda
python scripts/run_stt.py pretrain-mae --config "$CONFIG"

LATEST_RUN=$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'mae_*' | sort | tail -1)
if [[ -f "$LATEST_RUN/final_encoder.pt" ]]; then
  cp "$LATEST_RUN/final_encoder.pt" "$ARTIFACT"
  echo "TRAINING_COMPLETE run_dir=$LATEST_RUN artifact=$ARTIFACT"
else
  echo "Training did not produce final_encoder.pt"
  exit 1
fi
