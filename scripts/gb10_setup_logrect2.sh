#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/dsoselia/SegmentThisThingLogRect2GB10}"
REPO="${REPO:-$ROOT/segment_this_thing}"
ENV_NAME="${ENV_NAME:-stt-logrect2-gb10}"
BASE_ENV="${BASE_ENV:-sam-ft}"

mkdir -p "$ROOT"/{runs,artifacts,logs,data/coco2017/manifests}

source /home/dsoselia/miniforge3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -e "$REPO"
python -m pip install wandb einops

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
PY

echo "GB10 setup complete at $ROOT using env $ENV_NAME"
