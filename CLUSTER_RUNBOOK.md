# Zaratan STT-B Runbook

## Paths

- Cluster root: `/home/dsoselia/scratch.varshney-prj/SegmentThisThing`
- Repo checkout: `/home/dsoselia/scratch.varshney-prj/SegmentThisThing/segment_this_thing`
- Docs dir: `/home/dsoselia/scratch.varshney-prj/SegmentThisThing/docs`
- Weights path: `/home/dsoselia/scratch.varshney-prj/SegmentThisThing/stt-b-qbkbmb5qsb4q2.pth`

## Environment

Create a fresh env derived from the existing `py12` stack, then install the repo editable and extras on the cluster.

Example:

```bash
ssh Zaratan
source /home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh
conda create -y -n stt-fullrun --clone py12
conda activate stt-fullrun
cd /home/dsoselia/scratch.varshney-prj/SegmentThisThing/segment_this_thing
pip install -e .
pip install pycocotools scipy
mkdir -p /home/dsoselia/scratch.varshney-prj/SegmentThisThing/docs
```

If a package needs GitHub or external internet, install it on the login node, not on `zaratan-compute`.

## Validation Manifests

Build external eval manifests only after the raw datasets are staged:

```bash
python scripts/make_timberseg_manifest.py \
  --root /path/to/TimberSeg \
  --split val \
  --output /home/dsoselia/scratch.varshney-prj/SegmentThisThing/manifests/timberseg_val.jsonl

python scripts/make_zerowaste_f_manifest.py \
  --root /path/to/ZeroWaste-f \
  --split val \
  --output /home/dsoselia/scratch.varshney-prj/SegmentThisThing/manifests/zerowaste_f_val.jsonl
```

## COCO 2017 Staging

Use `Zaratan` only for the raw downloads and extraction into the project root:

```bash
ssh Zaratan
BASE=/home/dsoselia/scratch.varshney-prj/SegmentThisThing
mkdir -p "$BASE/data/coco2017/raw" "$BASE/downloads/coco2017"
cd "$BASE/downloads/coco2017"
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -n train2017.zip -d "$BASE/data/coco2017/raw"
unzip -n val2017.zip -d "$BASE/data/coco2017/raw"
unzip -n annotations_trainval2017.zip -d "$BASE/data/coco2017/raw"
```

Build manifests and validate them on `zaratan-compute`:

```bash
ssh zaratan-compute
source /home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh
conda activate stt-coco
cd /home/dsoselia/scratch.varshney-prj/SegmentThisThing/segment_this_thing

python scripts/make_image_manifest.py \
  --images-dir /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/raw/train2017 \
  --output /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/manifests/coco2017_mae_train_images.jsonl \
  --dataset-name coco2017 \
  --split-name train

python scripts/make_image_manifest.py \
  --images-dir /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/raw/val2017 \
  --output /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/manifests/coco2017_mae_val_images.jsonl \
  --dataset-name coco2017 \
  --split-name val

python scripts/make_instance_manifest.py \
  --annotations /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/raw/annotations/instances_train2017.json \
  --images-dir /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/raw/train2017 \
  --output /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/manifests/coco2017_train_instances.jsonl \
  --dataset-name coco2017 \
  --split-name train

python scripts/make_instance_manifest.py \
  --annotations /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/raw/annotations/instances_val2017.json \
  --images-dir /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/raw/val2017 \
  --output /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/manifests/coco2017_val_instances.jsonl \
  --dataset-name coco2017 \
  --split-name val

python scripts/validate_stt_manifest.py \
  --manifest /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/manifests/coco2017_mae_train_images.jsonl \
  --kind mae

python scripts/validate_stt_manifest.py \
  --manifest /home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017/manifests/coco2017_train_instances.jsonl \
  --kind seg
```

## Single-GPU Debug

Use this first on `zaratan-compute`:

```bash
ssh zaratan-compute
source /home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh
conda activate stt-fullrun
cd /home/dsoselia/scratch.varshney-prj/SegmentThisThing/segment_this_thing
python scripts/run_stt.py train-stt --config configs/stt_b_cluster_debug.json
```

## 2-GPU DDP Smoke

Use this before any longer run:

```bash
torchrun --standalone --nproc_per_node=2 scripts/run_stt.py \
  train-stt \
  --config configs/stt_b_cluster_ddp_smoke.json
```

Required checks after the smoke:

- `metrics.jsonl` exists and only rank 0 wrote it
- `RUN_STATUS.md` exists
- `last_checkpoint.txt` points to the newest checkpoint
- `eval/summary_step_*.json` exists

## Resume

Point `runtime.resume_from` at the checkpoint listed in `last_checkpoint.txt`, then relaunch the same command.

Example:

```bash
cat /home/dsoselia/scratch.varshney-prj/SegmentThisThing/runs/stt_b_cluster_ddp_smoke/<run_dir>/last_checkpoint.txt
```

Update the config and rerun:

```bash
torchrun --standalone --nproc_per_node=2 scripts/run_stt.py \
  train-stt \
  --config configs/stt_b_cluster_ddp_smoke.json
```

## Full-Run Template

For the eventual 2-8 H100 launch, start from:

- `configs/stt_b_cluster_full_template.json`

Tune at minimum:

- `micro_batch_size`
- `num_workers`
- `save_every`
- `eval_every`
- manifest paths

## Notes To Keep In `docs/`

For each debug or smoke run, write a short markdown note in `/home/dsoselia/scratch.varshney-prj/SegmentThisThing/docs` covering:

- exact config used
- GPU count and host
- samples/sec and seconds/step
- GPU memory
- whether bottleneck looks like dataloader, tokenization/target projection, model compute, or checkpoint I/O
