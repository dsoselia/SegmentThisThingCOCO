import math
import unittest

import torch

from segment_this_thing.foveation import LogRectilinearFoveator
from stt_pipeline.data import SegmentationBatch
from stt_pipeline.trainers import _materialize_segmentation_batch


def _raw_for_lambda(value: float, epsilon: float = 1e-6) -> float:
    return math.log(math.expm1(max(value - epsilon, 1e-8)))


class LearnableLogRectTests(unittest.TestCase):
    def test_axis_geometry_stays_valid_and_symmetric(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=True)
        for value in (0.01, 1.0, 100.0):
            with torch.no_grad():
                foveator.raw_lambda_scale.fill_(_raw_for_lambda(value))
            edges = foveator._build_axis_edges_tensor()
            widths = edges.diff()
            self.assertTrue(torch.all(widths > 0))
            self.assertTrue(torch.isclose(edges[0], torch.tensor(0.0)))
            self.assertTrue(torch.isclose(edges[-1], torch.tensor(256.0)))
            self.assertTrue(torch.allclose(widths, widths.flip(0), atol=1e-4, rtol=1e-4))

    def test_sampling_has_finite_nonzero_lambda_gradient(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=True)
        generator = torch.Generator().manual_seed(7)
        image = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8, generator=generator)
        tokens = foveator.extract_foveated_image(image)
        weights = torch.linspace(0.0, 1.0, tokens.numel()).reshape_as(tokens)
        (tokens * weights).mean().backward()
        gradient = foveator.raw_lambda_scale.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient))
        self.assertGreater(gradient.abs(), 0)

    def test_frozen_lambda_does_not_change(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        before = foveator.lambda_scale.detach().clone()
        image = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        tokens = foveator.extract_foveated_image(image)
        self.assertFalse(tokens.requires_grad)
        self.assertIsNone(foveator.raw_lambda_scale.grad)
        self.assertTrue(torch.equal(before, foveator.lambda_scale.detach()))

    def test_default_shapes_and_crop_coverage(self):
        foveator = LogRectilinearFoveator(16, 1280, 13, lambda_scale=1.0)
        lower, upper, area = foveator.get_bin_coordinates()
        self.assertEqual(lower.shape, (169, 16, 16, 2))
        self.assertEqual(upper.shape, lower.shape)
        self.assertEqual(area.shape, (169, 16, 16))
        self.assertEqual(foveator.get_num_tokens(), 169)
        self.assertTrue(torch.all(area > 0))
        self.assertTrue(torch.isclose(foveator._build_axis_edges_tensor()[-1], torch.tensor(1280.0)))

    def test_default_layout_matches_legacy_integer_widths(self):
        foveator = LogRectilinearFoveator(16, 1280, 13, lambda_scale=1.0)
        expected = torch.tensor([390, 153, 41, 16, 16, 16, 16, 16, 16, 16, 40, 153, 391], dtype=torch.float32)
        self.assertTrue(torch.allclose(foveator._build_axis_edges_tensor().diff(), expected, atol=1.0, rtol=0.0))

    def test_segmentation_target_is_detached_while_image_path_trains_lambda(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=True)
        image = torch.randint(0, 256, (300, 300, 3), dtype=torch.uint8)
        mask = torch.zeros((300, 300), dtype=torch.bool)
        mask[80:220, 80:220] = True
        batch = SegmentationBatch(
            images=[image], masks=[mask], centers=torch.tensor([[150, 150]]),
            dataset_names=["test"], image_paths=["test.png"], preprocessing={},
        )
        tokens, _, target, _ = _materialize_segmentation_batch(
            batch, foveator, torch.device("cpu"), non_blocking=False
        )
        self.assertTrue(tokens.requires_grad)
        self.assertFalse(target.requires_grad)
        weights = torch.linspace(0.0, 1.0, tokens.numel()).reshape_as(tokens)
        (tokens * weights).mean().backward()
        self.assertIsNotNone(foveator.raw_lambda_scale.grad)
        self.assertTrue(torch.isfinite(foveator.raw_lambda_scale.grad))
        self.assertGreater(foveator.raw_lambda_scale.grad.abs(), 0)


if __name__ == "__main__":
    unittest.main()
