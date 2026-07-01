#!/bin/bash
# MAE pretrain - 10K steps, 1x Beacon H100, fixed lambda-scale c=0.8.

#SBATCH --job-name=stt-logrect-c08-mae
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
CONFIG="$REPO/configs/stt_b_beacon_h100_logrect_opencl_fixedlambda_c08_mae_10k.json"
RUN_ROOT="$PROJECT/runs/logrect_opencl_adoption/lambda_c08_h100_10k_fixed"
ARTIFACT_ROOT="$PROJECT/artifacts/logrect_opencl_adoption"
ARTIFACT="$ARTIFACT_ROOT/stt_b_beacon_h100_logrect_opencl_fixedlambda_c08_mae_10k_final_encoder.pt"
mkdir -p "$PROJECT/slurm" "$RUN_ROOT" "$ARTIFACT_ROOT"
cd "$REPO"

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$CONFIG"
echo "artifact=$ARTIFACT"
echo "python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python -m unittest -v tests/test_logrect_learnable.py
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
