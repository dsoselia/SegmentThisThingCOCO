# Agent Start Here

This checkout is an isolated derivative of the current STT COCO training stack.

Use it as a clean experimental branch, not as the primary working tree.

## Primary Paths

- Local repo:
  - `/Users/davitsoselia/Downloads/SegmentThisLogRectilinear`
- Cluster root:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect`
- Cluster repo checkout:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing`

## Working Rules

- Keep code, configs, run outputs, artifacts, and Slurm logs under the new local/cluster roots.
- Do not point new runs back at the original repository output directories.
- Do not duplicate large shared datasets unless there is a hard requirement.

## Shared Data Policy

COCO should be reused, not copied.

The intended cluster setup is:

- new repo/output root under:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect`
- shared COCO data reused from:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017`

Preferred mechanism:

- create `data/coco2017` in the new cluster root as a symlink to the shared source

That allows configs in this repo to use the new root consistently while still avoiding dataset duplication.

## Current Training Shape

The current baseline in this repo is:

1. MAE pretraining from random STT-B weights
2. export encoder artifact
3. segmentation training from that encoder
4. periodic checkpointing and periodic eval

Current canonical COCO-oriented configs are under:

- `configs/stt_b_coco_paper_mae_3xa100.json`
- `configs/stt_b_coco_paper_seg_3xa100.json`
- `configs/stt_b_coco_paper_seg_3xa100_resume_from_6000.json`

Current cluster launchers are under:

- `cluster/stt_a100_3gpu_coco_mae.sbatch`
- `cluster/stt_a100_3gpu_coco_seg.sbatch`
- `cluster/stt_a100_3gpu_coco_seg_resume.sbatch`

## Zaratan Usage

Use `ssh Zaratan` for:

- editing files in the cluster checkout
- validating configs
- `sbatch`
- `squeue`
- `sacct`
- reading logs and run artifacts

Use `ssh zaratan-compute` only when you intentionally need the active compute node and the scheduler allows it.

If `pam_slurm_adopt` blocks access on `zaratan-compute`, that is a cluster access issue, not a repo issue.

## Git Remotes

This derivative checkout is expected to push to its own repository.

- `origin` should be the derivative GitHub repo
- `upstream` should remain the source checkout or source repository for comparison

## Before Making Architectural Changes

- read:
  - `README.md`
  - `REPLICATION_PIPELINE.md`
  - `CLUSTER_RUNBOOK.md`
  - `CODEBASE_AND_ZARATAN_GUIDE.md`
- verify the exact config you plan to modify
- verify whether the change affects:
  - tokenizer assumptions
  - target projection
  - eval assumptions
  - checkpoint compatibility

## Operational Caution

- Many configs contain explicit absolute paths.
- If you create new configs, keep path roots consistent with this derivative repo.
- If you resume jobs, verify `resume_from`, `output_dir`, and artifact paths together.
- If you add new cluster scripts, keep Slurm output paths under the new cluster root.

## Current Intent

This repo exists to isolate the next round of architectural experiments from the already-running COCO training line.

Keep the baseline working while making changes.
