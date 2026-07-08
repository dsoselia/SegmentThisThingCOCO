#!/bin/bash
# Run mini_external_eval_200x5 for an STT-ring segmentation checkpoint.

#SBATCH --job-name=stt-mini-ext
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=08:00:00
#SBATCH --output=/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

ART="${ART:?Set ART to the prepared external-eval artifact directory}"
REPO="${REPO:-/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/src/SegmentThisThingCOCO}"
CHECKPOINT="${CHECKPOINT:-}"
SMOKE_CONFIG="${SMOKE_CONFIG:-$ART/external_eval_mini_200x5_smoke_config.json}"
FULL_CONFIG="${FULL_CONFIG:-$ART/external_eval_mini_200x5_full_config.json}"

if [[ -z "$CHECKPOINT" ]]; then
  mapfile -t checkpoints < <(find "$ART" -maxdepth 1 -type f -name "checkpoint*.pt" | sort)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Set CHECKPOINT explicitly; found ${#checkpoints[@]} checkpoint files in $ART" >&2
    exit 1
  fi
  CHECKPOINT="${checkpoints[0]}"
fi

source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
cd "$REPO"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_DIR="$ART/wandb"
export WANDB_MODE=offline
mkdir -p "$ART/wandb"

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname) artifact_dir=$ART checkpoint=$CHECKPOINT"
nvidia-smi || true
python -m unittest -v tests/test_logrect_learnable.py
python scripts/check_foveator_numeric_ranges.py --config "$FULL_CONFIG" --synthetic --device cuda
python - <<PYCHECK
from pathlib import Path
import torch
from stt_pipeline import load_config
from stt_pipeline.modeling import build_foveator, build_model, load_segmentation_components

cfg = load_config(Path("$FULL_CONFIG"))
foveator = build_foveator(cfg.model)
model = build_model(cfg.model.size, foveator)
load_segmentation_components(model, foveator, Path("$CHECKPOINT"), strict=True)
state = torch.load(Path("$CHECKPOINT"), map_location="cpu", weights_only=False)
lambda_param = getattr(foveator, "lambda_scale", None)
payload = {
    "checkpoint_step": state.get("step") if isinstance(state, dict) else None,
    "tokenizer_type": cfg.model.tokenizer_type,
    "token_count": foveator.get_num_tokens(),
    "pattern_size": foveator.get_pattern_bounds_size(),
}
if lambda_param is not None:
    payload["lambda_scale"] = float(lambda_param.detach().cpu().item())
print(payload)
PYCHECK

python scripts/run_stt.py eval-stt --config "$SMOKE_CONFIG" --checkpoint "$CHECKPOINT" | tee "$ART/smoke_eval_stdout.json"
python scripts/run_stt.py eval-stt --config "$FULL_CONFIG" --checkpoint "$CHECKPOINT" | tee "$ART/full_eval_stdout.json"
python - <<PYCOPY
from pathlib import Path
import shutil
art = Path("$ART")
for label, root_name in [("smoke", "eval_runs_smoke"), ("full", "eval_runs_full")]:
    summaries = sorted((art / root_name).glob("eval_*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise SystemExit(f"missing {label} summary")
    shutil.copy2(summaries[-1], art / f"{label}_summary.json")
    print({"label": label, "summary": str(art / f"{label}_summary.json"), "run_summary": str(summaries[-1])})
PYCOPY

if command -v wandb >/dev/null 2>&1; then
  mapfile -t offline_runs < <(find "$ART/wandb/wandb" -maxdepth 1 -type d -name "offline-run-*" 2>/dev/null | sort)
  if [[ "${#offline_runs[@]}" -gt 0 ]]; then
    wandb sync --include-synced "${offline_runs[@]}" | tee "$ART/wandb_sync.log" || true
  fi
fi
