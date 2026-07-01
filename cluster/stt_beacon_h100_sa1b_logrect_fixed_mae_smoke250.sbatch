#!/bin/bash
# SA-1B MAE extended smoke, 250 steps, 1x Beacon H100.

#SBATCH --job-name=sa1b-logrect-mae250
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

EXP=/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments
REPO="$EXP/src/SegmentThisThingCOCO"
CONFIG="$REPO/configs/stt_b_sa1b_logrect_opencl_fixedlambda_mae_smoke250.json"
RUN_ROOT="$EXP/runs/mae_smoke250"
ARTIFACT="$EXP/artifacts/stt_b_sa1b_logrect_opencl_fixedlambda_mae_smoke250_final_encoder.pt"
mkdir -p "$EXP/slurm" "$EXP/artifacts" "$RUN_ROOT"
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
test -n "$LATEST_RUN"
test -f "$LATEST_RUN/final_encoder.pt"
cp "$LATEST_RUN/final_encoder.pt" "$ARTIFACT"
echo "TRAINING_COMPLETE run_dir=$LATEST_RUN artifact=$ARTIFACT"
