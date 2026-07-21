from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_sa1b_ddp_config import build_config, build_parser


ROOT = Path(__file__).resolve().parents[1]


class SA1BDDPConfigTest(unittest.TestCase):
    def _args(self, task: str, *extra: str):
        base = ROOT / "configs" / f"stt_b_sa1b_logrect_speed_profile_{task}_base.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "config.json"
            argv = [
                "--base",
                str(base),
                "--output",
                str(output),
                "--task",
                task,
                "--profile",
                f"test-{task}",
                "--experiment-root",
                "/experiment",
                "--staged-manifest-root",
                "/staged",
                "--micro-batch",
                "64" if task == "mae" else "8",
                "--effective-batch",
                "1024" if task == "mae" else "2048",
                "--workers",
                "6",
                "--prefetch",
                "8",
                "--in-order",
                "true",
                "--worker-pre-crop",
                "true" if task == "mae" else "false",
                "--num-steps",
                "500000" if task == "mae" else "250000",
                "--save-every",
                "10000",
                "--milestone-every",
                "50000",
                "--log-every",
                "50",
                "--checkpoint-keep-last",
                "4",
                "--nproc",
                "2",
                "--gpus-tag",
                "2xh200",
                "--gpu-type",
                "h200",
                *extra,
            ]
            return build_parser().parse_args(argv)

    def test_mae_resume_validation_and_paper_schedule(self):
        checkpoint = "/runs/mae/checkpoints/checkpoint_step_0119999.pt"
        args = self._args(
            "mae",
            "--resume-from",
            checkpoint,
            "--val-every",
            "1000",
            "--val-max-examples",
            "32",
            "--batch-double-every",
            "100000",
            "--warmup-steps",
            "10000",
            "--views-per-image",
            "2",
        )
        config = build_config(args)

        self.assertEqual(config["runtime"]["resume_from"], checkpoint)
        self.assertIsNone(config["runtime"]["fork_from"])
        self.assertEqual(config["mae"]["val_every"], 1000)
        self.assertEqual(config["mae"]["val_max_examples"], 32)
        self.assertEqual(config["mae"]["views_per_image"], 2)
        self.assertEqual(config["mae"]["warmup_steps"], 10000)
        self.assertEqual(
            config["mae"]["target_batch_schedule"],
            {"0": 1024, "100000": 2048, "200000": 4096, "300000": 8192, "400000": 16384},
        )
        self.assertFalse(config["model"]["log_rect_lambda_learnable"])
        self.assertEqual(config["model"]["log_rect_lambda_scale"], 1.0)
        self.assertIn("h200", config["runtime"]["wandb_tags"])
        self.assertNotIn("h100", config["runtime"]["wandb_tags"])

    def test_segmentation_initialization_eval_and_schedule(self):
        mae_checkpoint = "/runs/mae/checkpoints/checkpoint_step_0499999.pt"
        args = self._args(
            "seg",
            "--pretrained-mae-checkpoint",
            mae_checkpoint,
            "--val-every",
            "1000",
            "--val-max-examples",
            "100",
            "--target-batch-schedule",
            '{"0": 2048, "50000": 4096, "100000": 8192}',
            "--warmup-steps",
            "5000",
            "--max-segments-per-image",
            "16",
        )
        config = build_config(args)

        self.assertEqual(config["runtime"]["eval_every"], 1000)
        self.assertEqual(config["segmentation"]["pretrained_mae_checkpoint"], mae_checkpoint)
        self.assertIsNone(config["segmentation"]["pretrained_encoder"])
        self.assertIsNone(config["segmentation"]["init_checkpoint"])
        self.assertEqual(config["segmentation"]["max_segments_per_image"], 16)
        self.assertEqual(config["segmentation"]["warmup_steps"], 5000)
        self.assertEqual(
            config["segmentation"]["target_batch_schedule"],
            {"0": 2048, "50000": 4096, "100000": 8192},
        )
        self.assertEqual(config["evaluation"]["max_examples"], 100)

    def test_resume_and_fork_are_mutually_exclusive(self):
        args = self._args("mae", "--resume-from", "/a.pt", "--fork-from", "/b.pt")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_config(args)

    def test_segmentation_initialization_sources_are_mutually_exclusive(self):
        args = self._args(
            "seg",
            "--pretrained-encoder",
            "/encoder.pt",
            "--pretrained-mae-checkpoint",
            "/mae.pt",
        )
        with self.assertRaisesRegex(ValueError, "initialization options are mutually exclusive"):
            build_config(args)


if __name__ == "__main__":
    unittest.main()
