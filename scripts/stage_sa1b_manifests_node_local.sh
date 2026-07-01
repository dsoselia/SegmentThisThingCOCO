#!/bin/bash

set -euo pipefail

EXP="${EXP:-/beacon-homes/dsoselia/foveatedseg/SA1B_Experiments}"
SOURCE_DIR="${SOURCE_DIR:-$EXP/manifests}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/foveatedseg-sa1b-manifests-${USER}}"
LOCK_FILE="${STAGE_ROOT}.lock"
MARKER="$STAGE_ROOT/.complete"
MANIFEST_NAMES=(
  sa1b_subset_train.jsonl
  sa1b_subset_val.jsonl
  sa1b_subset_mae_train_images.jsonl
  sa1b_subset_mae_val_images.jsonl
)

exec 9>"$LOCK_FILE"
flock 9

mkdir -p "$STAGE_ROOT"
if [[ -f "$MARKER" ]]; then
  echo "$STAGE_ROOT"
  exit 0
fi

BUILD_ROOT="${STAGE_ROOT}.building-${SLURM_JOB_ID:-$$}"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

for name in "${MANIFEST_NAMES[@]}"; do
  src="$SOURCE_DIR/$name"
  if [[ ! -f "$src" ]]; then
    echo "Missing manifest: $src" >&2
    exit 1
  fi
  cp "$src" "$BUILD_ROOT/$name"
  if [[ -f "$src.offsets.npy" ]]; then
    cp "$src.offsets.npy" "$BUILD_ROOT/$name.offsets.npy"
  fi
done

touch "$BUILD_ROOT/.complete"
rm -rf "$STAGE_ROOT"
mv "$BUILD_ROOT" "$STAGE_ROOT"
df -h "$STAGE_ROOT" >&2
echo "$STAGE_ROOT"
