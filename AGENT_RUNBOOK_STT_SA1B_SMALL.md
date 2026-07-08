# STT SA-1B Small Runbook

This workspace is separate from the logrect SA-1B experiments.

## Roots

- Local workspace: `/Users/dsoselia/Documents/STT_SA1b_small_run`
- Beacon workspace: `/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments`
- Training repo on Beacon: `/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/src/SegmentThisThingCOCO`
- Official STT reference repo on Beacon: `/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/src/segment_this_thing_official`
- W&B project: `SegmentThisThingSTTSA1BSmall`

## What This Runs

- Tokenizer: original STT ring foveation, not logrect.
- Pattern: `token_size=16`, `strides=[1,2,4,6,8]`, `grid_sizes=[4,4,6,8,10]`, `pattern_size=1280`.
- MAE schedule: 2xH100, fixed effective batch `128`, micro-batch `64` per rank, `500000` steps.
- Segmentation schedule: 2xH100, fixed effective batch `32`, micro-batch `8` per rank, `250000` steps, dependent on paired MAE completion.

## Submit

Run from the Beacon training repo.

```bash
cd /beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/src/SegmentThisThingCOCO

MAE_PROFILE_SUFFIX=_stt_small_main
MAE_PROFILE=stt_sa1b_ddp_mae_2xh100_mb64_eb128_wpr6_p8_iotrue_long500000_workerprecrop${MAE_PROFILE_SUFFIX}
MAE_RUN_DIR=/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/runs/$MAE_PROFILE

MAE_JOB=$(
  sbatch --parsable --time=3-00:00:00 \
    --export=ALL,TASK=mae,MODE=long,PROFILE_SUFFIX=$MAE_PROFILE_SUFFIX,WANDB_RUN_NAME=beacon-2xh100-sa1b-stt-small-mae-500k \
    cluster/stt_beacon_stt_sa1b_ddp.sbatch
)

SEG_JOB=$(
  sbatch --parsable --dependency=afterok:$MAE_JOB --time=3-00:00:00 \
    --export=ALL,TASK=seg,MODE=long,PROFILE_SUFFIX=_stt_small_from_mae500k,SEG_PRETRAINED_MAE_CHECKPOINT=auto,MAE_OUTPUT_DIR=$MAE_RUN_DIR,WANDB_RUN_NAME=beacon-2xh100-sa1b-stt-small-seg-from-mae500k \
    cluster/stt_beacon_stt_sa1b_ddp.sbatch
)

echo "MAE_JOB=$MAE_JOB"
echo "SEG_JOB=$SEG_JOB"
echo "MAE_RUN_DIR=$MAE_RUN_DIR"
```

## Monitor

```bash
squeue -u dsoselia -o "%.18i %.9P %.30j %.8T %.10M %.10l %.6D %R"
tail -f /beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/slurm/stt-sa1b-ddp-<jobid>.out
```

MAE:

```bash
tail -n 20 $MAE_RUN_DIR/metrics.jsonl
tail -n 20 $MAE_RUN_DIR/mae_val_metrics.jsonl
ls -lh $MAE_RUN_DIR/checkpoints
```

Segmentation:

```bash
SEG_RUN_DIR=/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/runs/stt_sa1b_ddp_seg_2xh100_mb8_eb32_wpr4_p8_iotrue_long250000_stt_small_from_mae500k
tail -n 20 $SEG_RUN_DIR/metrics.jsonl
tail -n 20 $SEG_RUN_DIR/eval_metrics.jsonl
ls -lh $SEG_RUN_DIR/checkpoints
```

GPU/CPU monitors are in:

```bash
ls -lh /beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/slurm/*_<jobid>_gpu.csv
ls -lh /beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/slurm/*_<jobid>_cpu.txt
```

## External Eval

Prepare an eval artifact directory for a segmentation checkpoint:

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
ART=/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/artifacts/external_eval_mini_200x5_$TS
PACK=/beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/external_eval/mini_external_eval_200x5
CKPT=/path/to/segmentation/checkpoints/checkpoint_step_XXXXXXX.pt
WANDB_PROJECT_EXTERNAL=SegmentThisThing-STT-SA1B-Small-ExternalEval

python scripts/prepare_mini_external_eval.py \
  --pack "$PACK" \
  --artifact-dir "$ART" \
  --base-config configs/stt_b_sa1b_stt_small_seg_base.json \
  --checkpoint "$CKPT" \
  --run-name "mini-200x5-stt-small-seg-step-XXXXXXX" \
  --profile-prefix "mini_200x5_stt_small_step_XXXXXXX" \
  --selection-policy "manual STT small checkpoint selection" \
  --no-copy-checkpoint \
  --wandb-project "$WANDB_PROJECT_EXTERNAL" \
  --wandb-tags stt-ring sa1b external-eval mini-200x5 small-schedule h100 offline

sbatch --export=ALL,ART="$ART",CHECKPOINT="$CKPT" cluster/stt_beacon_stt_mini_external_eval.sbatch
```

Results:

```bash
cat $ART/full_summary.json
cat $ART/selection.json
```

## W&B Sync

Beacon runs in offline mode. Sync one or more offline runs from the STT workspace:

```bash
find /beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/wandb/wandb -maxdepth 1 -type d -name 'offline-run-*' -print
wandb sync --include-synced /beacon-homes/dsoselia/foveatedseg/STT_SA1B_Experiments/wandb/wandb/offline-run-*
```

External eval W&B files live under the artifact directory's `wandb/` subdirectory.
