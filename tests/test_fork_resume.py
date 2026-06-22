import json
import tempfile
import unittest
from pathlib import Path

from stt_pipeline.config import ExperimentConfig, load_config
from stt_pipeline.trainers import _make_run_dir


class ForkResumeTests(unittest.TestCase):
    def test_fork_creates_new_run_under_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig()
            config.runtime.output_dir = str(Path(tmp) / "fork-output")
            config.runtime.fork_from = str(Path(tmp) / "source" / "checkpoints" / "source.pt")
            run_dir = _make_run_dir(config, "mae")
            self.assertEqual(run_dir.parent, Path(config.runtime.output_dir))
            self.assertTrue(run_dir.name.startswith("mae_"))

    def test_resume_reuses_checkpoint_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "mae_source"
            checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            config = ExperimentConfig()
            config.runtime.resume_from = str(checkpoint)
            self.assertEqual(_make_run_dir(config, "mae"), run_dir.resolve())

    def test_resume_and_fork_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"runtime": {"resume_from": "a", "fork_from": "b"}}))
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
