# Nexus Cluster Primer for STT / LogRect Experiments

This document covers everything a new agent needs to run SegmentThisThing (STT) LogRect experiments on the Nexus cluster as user `dsoselia`.

---

## 1. GPU Availability

Check available GPUs right now:

```bash
bash ~/check_avail_l.sh
```

### Node/GPU Reference

| Node | GPUs | GRES string | Partition | Notes |
|------|------|-------------|-----------|-------|
| cml33 | 4× H100 SXM 80 GB | `gpu:h100-sxm:N` | `cml-scavenger` / `cml-wriva` | Primary target for STT runs |
| cml35 | 8× H200 SXM 141 GB | `gpu:h200-sxm:N` | `cml-scavenger` | More VRAM; rarer availability |
| cml34 | 8× L40S 48 GB | `gpu:l40s:N` | `cml-scavenger` | Good for smaller runs |
| cml32 | 4× A100 80 GB | `gpu:a100:N` | `cml-scavenger` | Older but reliable |
| clip12/13 | A6000 | `gpu:a6000:N` | `clip` | Needs `--qos=high`, 24h limit |
| gammagpu18-21 | L40S | varies | `scavenger` | Cross-lab scavenger |

### Partitions

| Partition | Account | QoS | Max wall time | Notes |
|-----------|---------|-----|---------------|-------|
| `cml-scavenger` | `cml-scavenger` | `cml-scavenger` | unlimited (but preemptable) | Primary research partition |
| `scavenger` | — | — | — | Broader cross-lab scavenger |
| `cml-wriva` | `cml-wriva` | `cml-wriva` | — | Dedicated H100 SXM on cml33 |
| `cml-wriva-high` | `cml-wriva` | `cml-wriva-high` | — | Higher-priority dedicated |
| `clip` | — | `high` | 24h | CLIP lab A6000s |

**Standard header for cml-scavenger H100 SXM jobs:**

```bash
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --gres=gpu:h100-sxm:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120
```

---

## 2. Conda Environments

Both envs are at `/cmlscratch/dsoselia/miniconda3/envs/`.

```bash
CONDA_SH="/cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12   # or torch-py313
```

| Env | Python | PyTorch | CUDA | Status |
|-----|--------|---------|------|--------|
| `py12` | 3.12 | 2.5.1 | 12.4 | Has wandb, einops, pycocotools |
| `torch-py313` | 3.13 | 2.11.0 | 12.8 | Newer; use if py12 breaks |

If a package is missing from `py12`:

```bash
source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate py12
pip install <package>
```

---

## 3. Datasets

All datasets live under `/fs/cml-datasets/` on a shared NFS mount accessible from all nodes.

### COCO 2017 (primary dataset for STT)

```
/fs/cml-datasets/coco/
  images/
    train2017/          # ~118k images
    val2017/            # ~5k images
  annotations/
    instances_train2017.json
    instances_val2017.json
    captions_train2017.json
    captions_val2017.json
```

Pre-built JSONL manifests (ready to use, no rebuild needed):

```
/cmlscratch/dsoselia/STTLogRectLearnable/data/coco2017/manifests/
  coco2017_mae_train_images.jsonl
  coco2017_mae_val_images.jsonl
  coco2017_train_instances.jsonl
  coco2017_val_instances.jsonl
```

To reuse in a new repo, symlink:

```bash
mkdir -p /cmlscratch/dsoselia/<REPO>/data/coco2017
ln -s /cmlscratch/dsoselia/STTLogRectLearnable/data/coco2017/manifests \
      /cmlscratch/dsoselia/<REPO>/data/coco2017/manifests
```

### Other Notable Datasets

| Dataset | Path |
|---------|------|
| ImageNet (ILSVRC2012) | `/fs/cml-datasets/ImageNet/ILSVRC2012/` |
| ADE20K | `/fs/cml-datasets/ade20k/` |
| Cityscapes | `/fs/cml-datasets/cityscapes/` |
| LAION | `/fs/cml-datasets/laion/` |
| LibriSpeech | `/fs/cml-datasets/LibriSpeech/` |
| Kinetics-400 | `/fs/cml-datasets/Kinetics-400/` |
| LSUN | `/fs/cml-datasets/LSUN/` |
| SA-1B (SAM) | `/fs/cml-datasets/sa1b/` (may be empty) |

---

## 4. STT Framework: JSON-Config Pipeline

This repo uses a **JSON-config-driven** training pipeline — NOT the older `train_mae.py` / `train_seg.py` torchrun approach.

### Commands

```bash
# MAE pretrain
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/run_stt.py pretrain-mae --config configs/<mae_config>.json

# Segmentation finetune
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/run_stt.py train-stt --config configs/<seg_config>.json
```

### Config Structure

**MAE config** key fields:
```json
{
  "model": { "size": "b", "tokenizer_type": "log_rect_box", ... },
  "runtime": {
    "output_dir": "/cmlscratch/dsoselia/<REPO>/runs/<run_name>",
    "distributed": true,
    "micro_batch_size": 2,
    "num_workers": 8,
    "num_steps": 10000,
    "wandb_project": "...",
    "wandb_run_name": "...",
    "resume_from": null
  },
  "mae": {
    "enabled": true,
    "train_manifest": "/cmlscratch/dsoselia/<REPO>/data/coco2017/manifests/coco2017_mae_train_images.jsonl",
    "effective_batch_size": 1024,
    ...
  }
}
```

**Seg config** key fields:
```json
{
  "segmentation": {
    "enabled": true,
    "train_manifest": "...coco2017_train_instances.jsonl",
    "pretrained_encoder": ".../artifacts/final_encoder.pt",
    "effective_batch_size": 2048,
    "log_rect_warp_lr": 1e-06,   // only for learnable variant
    ...
  }
}
```

**Always set `"distributed": true`** for multi-GPU jobs.

### Batch Math

- **MAE** (2 GPU, mb=2, views_per_image=2): 2 × 2 × 2 = 8 samples/step → accum = 1024/8 = 128
- **Seg** (2 GPU, mb=1, max_segments=16): 2 × 1 × 16 = 32 samples/step → accum = 2048/32 = 64

### Encoder Artifact Export

After MAE finishes, `final_encoder.pt` is inside a timestamped directory:

```
runs/<mae_run_name>/mae_<TIMESTAMP>/final_encoder.pt
```

The seg config expects it at a deterministic path. The MAE SLURM script copies it:

```bash
ENCODER=$(ls -t "$ROOT/runs/log_rect2_mae_gb10_10k_eb1024"/mae_*/final_encoder.pt 2>/dev/null | head -1)
cp "$ENCODER" "$ROOT/artifacts/stt_b_coco_log_rect2_gb10_10k_eb1024_final_encoder.pt"
```

---

## 5. SLURM Script Patterns

### Reference Scripts

| Location | What it shows |
|----------|---------------|
| `/cmlscratch/dsoselia/SegmentThisThing/slurm/` | 20+ examples: single-GPU, DDP, A100/H100, array jobs |
| `/cmlscratch/dsoselia/STTLogRectLearnable/cluster/stt_h100sxm_2gpu_mae_gb10.sbatch` | H100 SXM MAE pretrain, encoder copy, self-resubmit |
| `/cmlscratch/dsoselia/STTLogRectLearnable/cluster/stt_h100sxm_2gpu_seg_gb10.sbatch` | H100 SXM seg finetune, auto-resume, self-resubmit |

### Auto-Resume Pattern

Scavenger jobs get preempted or hit wall time. The seg script handles both:

```bash
# 1. Detect latest checkpoint at job start
LAST_CKPT_FILE=$(ls -t "$ROOT/runs/<run_name>"/seg_*/last_checkpoint.txt 2>/dev/null | head -1 || true)
if [[ -n "$LAST_CKPT_FILE" && -f "$LAST_CKPT_FILE" ]]; then
  LATEST_CKPT=$(cat "$LAST_CKPT_FILE" | tr -d '[:space:]')
fi

# 2. If checkpoint found, patch config with resume_from
if [[ -n "$LATEST_CKPT" && -f "$LATEST_CKPT" ]]; then
  EFFECTIVE_CONFIG="/tmp/resume_${SLURM_JOB_ID}.json"
  python3 -c "
import json
with open('$BASE_CONFIG') as f: cfg = json.load(f)
cfg['runtime']['resume_from'] = '$LATEST_CKPT'
with open('$EFFECTIVE_CONFIG', 'w') as f: json.dump(cfg, f, indent=2)
"
else
  EFFECTIVE_CONFIG="$BASE_CONFIG"
fi
```

### Self-Resubmit Pattern (for time-limit expiry)

`--requeue` handles preemption but NOT wall-time exits. Add after torchrun:

```bash
LAST_STEP=$(python3 - <<'PY'
import glob, torch, sys
files = sorted(glob.glob("runs/<run_name>/seg_*/last_checkpoint.txt"))
if not files: print(0); sys.exit()
ckpt = open(files[-1]).read().strip()
try:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(int(state.get("step", 0)))
except Exception: print(0)
PY
)
if [[ "$LAST_STEP" -lt <TOTAL_STEPS - 1> ]]; then
  sbatch cluster/<this_script>.sbatch
fi
```

### Dependency Chaining

```bash
PRETRAIN_JOB=$(sbatch --parsable cluster/stt_h100sxm_2gpu_mae_gb10.sbatch)
FINETUNE_JOB=$(sbatch --parsable --dependency=afterok:$PRETRAIN_JOB cluster/stt_h100sxm_2gpu_seg_gb10.sbatch)
```

---

## 6. LogRect2 Repo Layout (`/cmlscratch/dsoselia/logrect2/`)

This is the **non-learnable** LogRect2 variant (branch `LogRect2-gb10`).

```
logrect2/
  configs/
    stt_b_coco_log_rect2_mae_gb10_short.json   # 1k-step smoke test, mb=1, eb=8
    stt_b_coco_log_rect2_seg_gb10_short.json   # 1k-step smoke test, eval_every=250
    stt_b_coco_log_rect2_mae_gb10_10k_eb1024.json   # full MAE (needs path update)
    stt_b_coco_log_rect2_seg_gb10_20k_eb2048.json   # full seg (needs path update)
  cluster/
    stt_a100_*.sbatch / stt_h100_*.sbatch   # old-cluster scripts; adapt for Nexus
  scripts/
    run_stt.py             # main entry point
    make_image_manifest.py
    make_instance_manifest.py
```

### Path Update Required

Short smoke configs still have `/home/dsoselia/SegmentThisThingLogRect2GB10/...` — update to `/cmlscratch/dsoselia/logrect2/...` before use.

Key paths to update in any config:
- `runtime.output_dir`
- `mae.train_manifest` / `mae.val_manifest`
- `segmentation.train_manifest`
- `segmentation.pretrained_encoder`
- `evaluation.named_eval_manifests.coco2017_val`

### Manifests

Symlink from the already-built set:

```bash
mkdir -p /cmlscratch/dsoselia/logrect2/data/coco2017
ln -s /cmlscratch/dsoselia/STTLogRectLearnable/data/coco2017/manifests \
      /cmlscratch/dsoselia/logrect2/data/coco2017/manifests
```

---

## 7. Known Gotchas

### DDP + Learnable Foveation: `broadcast_buffers=False`

**Symptom**: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1, 17]] is at version 2; expected version 1 instead.`

**Cause**: PyTorch DDP defaults to `broadcast_buffers=True`, which broadcasts all registered buffers **in-place** at each `forward()`. The learnable log-rect code saves a view of `bin_fraction_edges` (a registered buffer) into the autograd graph *before* `model.forward()` is called. The in-place broadcast increments the version counter from 1 → 2, and `backward()` fails the version check.

**Fix** (already applied in `STTLogRectLearnable/stt_pipeline/trainers.py`):

```python
return DistributedDataParallel(
    model,
    device_ids=[device.index] if device.type == "cuda" else None,
    output_device=device.index if device.type == "cuda" else None,
    broadcast_buffers=False,   # ← critical for learnable log-rect
)
```

This is NOT needed for non-learnable runs in `logrect2/`, but keep it in mind if you ever enable `log_rect_learnable: true`.

### `save_every` and Preemption Loss

Default `save_every=1000` at ~16 sec/step = up to 4.4h of lost work per preemption. Reduce to `500` for seg runs: max ~2.2h lost.

### W&B Offline Mode

Configs use `"wandb_mode": "offline"`. After training, sync with:

```bash
wandb sync runs/<run_name>/mae_<timestamp>/wandb/offline-run-*/
```

---

## 8. Active Runs (as of 2026-06-01)

| Job | Type | Steps | Config | Status |
|-----|------|-------|--------|--------|
| 6951634 | seg finetune (learnable) | 0→20000 | `STTLogRectLearnable/configs/stt_b_coco_learnable_log_rect_seg_gb10_20k_eb2048.json` | Running on cml33, step ~28+ |

The MAE pretrain (10k steps) completed earlier; the encoder artifact is at:
`/cmlscratch/dsoselia/STTLogRectLearnable/artifacts/stt_b_coco_log_rect2_gb10_10k_eb1024_final_encoder.pt`

---

## 9. Quick-Start Checklist for a New Run

1. `bash ~/check_avail_l.sh` — find available nodes
2. Copy/adapt a config from `STTLogRectLearnable/configs/` into your repo's `configs/`
3. Update all absolute paths to your repo root
4. Set `"distributed": true` in config
5. Symlink manifests: `ln -s .../STTLogRectLearnable/data/coco2017/manifests .../data/coco2017/manifests`
6. Copy/adapt a SLURM script from `STTLogRectLearnable/cluster/` into your `cluster/`
7. `mkdir -p slurm artifacts runs`
8. `sbatch cluster/<your_script>.sbatch`
9. Check logs: `tail -f slurm/<jobname>_<jobid>.out`
