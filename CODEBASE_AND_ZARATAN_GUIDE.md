# STT Codebase And Zaratan Guide

This document is the practical overview of the current repository and cluster workflow.

It focuses on:

- what lives where in the codebase
- how the MAE pretraining and segmentation training pipeline works
- how to work with Zaratan safely
- how to submit the current `3x A100` jobs

## Repository Layout

### Core code

- `stt_pipeline/`
  - training, evaluation, runtime, manifest loading, checkpointing, W&B, and cluster-oriented orchestration
- `scripts/run_stt.py`
  - single CLI entrypoint for pretraining, segmentation training, eval, smoke startup checks, and benchmarks
- `segment_this_thing/`
  - model implementation and supporting code used by the replication pipeline

### Run configuration

- `configs/`
  - JSON configs for smoke runs, paper-shaped runs, COCO runs, A100/H100 variants, and resume jobs
- `cluster/`
  - `sbatch` wrappers for cluster submission

### Data and outputs

- `manifests/`
  - JSONL manifests for image-only MAE training or instance-mask segmentation/eval
- `runs/`
  - local run outputs when working in this checkout
- cluster outputs are written under:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/runs/...`
- cluster logs are written under:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm/...`

### Existing documentation

- [README.md](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/README.md)
- [REPLICATION_PIPELINE.md](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/REPLICATION_PIPELINE.md)
- [CLUSTER_RUNBOOK.md](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/CLUSTER_RUNBOOK.md)

This file is the concise operator-facing version of those notes.

## Training Pipeline

The current intended training path is:

1. MAE pretraining from random STT weights
2. export the encoder-only artifact
3. segmentation training using that encoder as initialization
4. periodic checkpointing and periodic eval

The segmentation stage should not initialize from the released STT checkpoint when running the paper-shaped training path. It should load only the encoder exported from MAE.

## MAE Pretraining

### Entry point

```bash
python scripts/run_stt.py pretrain-mae --config <config>
```

### What it does

- loads an image-only manifest
- samples foveation centers uniformly from the image subject to a margin
- creates foveated token inputs using the STT ring tokenizer
- trains the image encoder with MAE masking
- writes periodic checkpoints
- writes `final_encoder.pt` at successful completion

### Current paper-shaped COCO path

Current production `3x A100` config:

- [stt_b_coco_paper_mae_3xa100.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_coco_paper_mae_3xa100.json)

Important settings:

- dataset: COCO 2017 train images
- model: `STT-B`
- tokenizer: `stt_ring`
- initialization: random STT weights
- `micro_batch_size = 8`
- `num_workers = 4` per rank
- `world_size = 3`
- `effective_batch_size = 1024`
- `save_every = 500`
- `num_steps = 10000`

### MAE artifact handoff

Successful MAE writes:

- run-local encoder artifact:
  - `<run_dir>/final_encoder.pt`
- cluster-exported handoff artifact:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/artifacts/stt_b_coco_paper_mae_3xa100_final_encoder.pt`

That exported artifact is what the segmentation config should point at.

## Segmentation Training

### Entry point

```bash
python scripts/run_stt.py train-stt --config <config>
```

### What it does

- loads an instance-mask manifest
- samples up to `max_segments_per_image` segments per image when enabled
- samples prompt centers within those segments
- builds foveated token inputs and real-valued foveated mask targets
- trains the full segmentation model using:
  - focal loss
  - dice loss
  - IoU prediction loss
- writes periodic checkpoints
- runs periodic evaluation against configured eval manifests

### Current paper-shaped COCO path

Current production `3x A100` config:

- [stt_b_coco_paper_seg_3xa100.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_coco_paper_seg_3xa100.json)

Important settings:

- dataset: COCO 2017 train instances
- eval dataset: COCO 2017 val instances
- initialization:
  - `pretrained_encoder = /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/artifacts/stt_b_coco_paper_mae_3xa100_final_encoder.pt`
  - `init_checkpoint = null`
- `micro_batch_size = 2`
- `num_workers = 4` per rank
- `world_size = 3`
- `max_segments_per_image = 16`
- `effective_batch_size = 2048`
- `save_every = 500`
- `eval_every = 500`
- `num_steps = 20000`

### Resume behavior

Resume is handled through:

- `runtime.resume_from`

When that is set, the training code:

- reuses the original run directory inferred from the checkpoint path
- restores model, optimizer, scaler, and RNG state
- continues counting steps from the checkpoint

Current resume config:

- [stt_b_coco_paper_seg_3xa100_resume_from_6000.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_coco_paper_seg_3xa100_resume_from_6000.json)

Current resume wrapper:

- [stt_a100_3gpu_coco_seg_resume.sbatch](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/cluster/stt_a100_3gpu_coco_seg_resume.sbatch)

## Manifest Types

There are two relevant manifest shapes:

### MAE image manifest

One line per image. No instance masks required.

Used by:

- `pretrain-mae`

### Segmentation instance manifest

One line per image with one or more segment definitions.

Supported encodings:

- `mask_path`
- `polygon`
- `rle` if `pycocotools` is installed

Used by:

- `train-stt`
- `eval-stt`

## Zaratan Workflow

There are two SSH targets with different purposes.

## `ssh Zaratan`

Use the login node for:

- editing files in the cluster checkout
- staging datasets and manifests
- activating conda environments
- validating configs and scripts
- submitting jobs with `sbatch`
- checking scheduler state with `squeue`, `sacct`, and reading log files

This is the correct default entrypoint for cluster work.

### Typical login-node use

```bash
ssh Zaratan
cd /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing
sbatch ./cluster/stt_a100_3gpu_coco_mae.sbatch
```

## `ssh zaratan-compute`

Use a compute node only when you already have an active interactive allocation or a direct compute-node session that the scheduler allows.

Use it for:

- short interactive debugging
- direct `torchrun` tests
- fast inspection while attached to an active allocation

Do not rely on it for normal job submission.

Important constraint:

- if you do not currently own an active job or allocation on that node, access may fail with `pam_slurm_adopt`
- that is expected cluster behavior, not a repo bug

So the safe rule is:

- use `ssh Zaratan` for submission and scheduler/accounting work
- use `ssh zaratan-compute` only when you intentionally need the live allocated node

## Current 3xA100 Submission Commands

All current production jobs should be submitted from `ssh Zaratan`.

### MAE production run

```bash
ssh Zaratan
cd /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing
sbatch ./cluster/stt_a100_3gpu_coco_mae.sbatch
```

### Segmentation production run

This should be submitted only after the MAE artifact exists:

- `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/artifacts/stt_b_coco_paper_mae_3xa100_final_encoder.pt`

```bash
ssh Zaratan
cd /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing
sbatch ./cluster/stt_a100_3gpu_coco_seg.sbatch
```

### Segmentation resume run

```bash
ssh Zaratan
cd /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing
sbatch ./cluster/stt_a100_3gpu_coco_seg_resume.sbatch
```

## Monitoring Commands

From `ssh Zaratan`:

### Queue status

```bash
squeue -j <jobid> -o "%.18i %.9P %.30j %.8T %.10M %.20R"
```

### Accounting after finish or timeout

```bash
sacct -j <jobid> --format=JobID,JobName%30,Partition,State,Elapsed,ExitCode,NodeList -P
```

### Read Slurm log

```bash
tail -n 80 /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm/<jobname>-<jobid>.out
```

### Read run status

```bash
sed -n '1,220p' /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/runs/<profile>/<run_dir>/RUN_STATUS.md
```

### Check recent checkpoints

```bash
ls -1 /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/runs/<profile>/<run_dir>/checkpoints | tail
```

### Check recent eval summaries

```bash
ls -1 /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/runs/<profile>/<run_dir>/eval | tail
```

## W&B Offline Runs

The code uses offline W&B.

Each run writes under:

- `<run_dir>/wandb/wandb/offline-run-*`

For sync later, only the `offline-run-*` directories matter. Ignore:

- `debug.log`
- `debug-internal.log`
- `latest-run`

For resumed training, because the run directory is reused, multiple `offline-run-*` directories may accumulate under the same run directory. Sync all of them.

## Practical Notes

- `RUN_STATUS.md` is useful, but the most reliable truth is usually:
  - `metrics.jsonl`
  - checkpoint files
  - eval summaries
  - Slurm accounting
- if Slurm says `TIMEOUT`, treat that as infrastructure termination even if `RUN_STATUS.md` still says `running`
- current `3x A100` segmentation throughput is good enough to progress cleanly, but not fast enough to finish `20000` steps inside an `8h` wall-clock limit, which is why the resume path exists

## Current Canonical COCO Path

Use this sequence:

1. submit [stt_a100_3gpu_coco_mae.sbatch](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/cluster/stt_a100_3gpu_coco_mae.sbatch)
2. wait for exported encoder artifact
3. submit [stt_a100_3gpu_coco_seg.sbatch](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/cluster/stt_a100_3gpu_coco_seg.sbatch)
4. if interrupted by wall-clock limit, continue with [stt_a100_3gpu_coco_seg_resume.sbatch](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/cluster/stt_a100_3gpu_coco_seg_resume.sbatch)

That is the current operational path for training STT-B from scratch on COCO on Zaratan.
