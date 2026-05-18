#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/dsoselia/SegmentThisThingLogRect2GB10}"
REPO="${REPO:-$ROOT/segment_this_thing}"
ENV_NAME="${ENV_NAME:-stt-logrect2-gb10}"
SMOKE_ENCODER="$ROOT/artifacts/stt_b_coco_log_rect2_gb10_smoke_final_encoder.pt"

if ! command -v conda >/dev/null 2>&1; then
  source /home/dsoselia/miniforge3/etc/profile.d/conda.sh
fi
conda activate "$ENV_NAME"

export WANDB_MODE=offline
export WANDB_PROJECT=SegmentThisThingLogRect2GB10

cd "$REPO"

python - <<'PY'
import torch
from segment_this_thing.foveation import LogRectilinearFoveator

foveator = LogRectilinearFoveator(
    token_size=16,
    pattern_size=1280,
    axis_bins=13,
    exponent=4.0,
    center_width=16,
)
edges = foveator._build_axis_edges()
widths = [right - left for left, right in zip(edges[:-1], edges[1:])]
print("num_tokens", foveator.get_num_tokens())
print("pattern_bounds", foveator.get_pattern_bounds_size())
print("axis_widths", widths)
assert foveator.get_num_tokens() == 169
assert widths == [390, 153, 41, 16, 16, 16, 16, 16, 16, 16, 40, 153, 391]
assert foveator.get_pattern_bounds_size() == 1280
tokens = foveator.extract_foveated_image(torch.zeros(3, 1280, 1280, dtype=torch.uint8))
recon = foveator.generate_foveated_visualization(tokens)
print("token_shape", tuple(tokens.shape))
print("recon_shape", tuple(recon.shape))
assert tuple(tokens.shape) == (169, 3, 16, 16)
assert tuple(recon.shape) == (3, 1280, 1280)
PY

python scripts/run_stt.py pretrain-mae \
  --config configs/stt_b_coco_log_rect2_mae_gb10_smoke.json

LATEST_RUN="$(find "$ROOT/runs/smoke_log_rect2_mae_gb10" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "$LATEST_RUN" || ! -f "$LATEST_RUN/final_encoder.pt" ]]; then
  echo "Missing MAE smoke final_encoder.pt under $ROOT/runs/smoke_log_rect2_mae_gb10" >&2
  exit 1
fi
cp "$LATEST_RUN/final_encoder.pt" "$SMOKE_ENCODER"
echo "Copied smoke encoder to $SMOKE_ENCODER"

python scripts/run_stt.py train-stt \
  --config configs/stt_b_coco_log_rect2_seg_gb10_smoke.json

echo "GB10 LogRect2 smoke complete"
