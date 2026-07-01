import math
import unittest

import torch

from segment_this_thing.foveation import LogRectilinearFoveator
from stt_pipeline.data import SegmentationBatch
from stt_pipeline.trainers import _materialize_segmentation_batch


def _raw_for_lambda(value: float, epsilon: float = 1e-6) -> float:
    return math.log(math.expm1(max(value - epsilon, 1e-8)))


class FixedLogRectTests(unittest.TestCase):
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

    def test_sampling_is_finite_with_fixed_opencl_integer_edges(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        generator = torch.Generator().manual_seed(7)
        image = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8, generator=generator)
        tokens = foveator.extract_foveated_image(image)
        self.assertTrue(torch.isfinite(tokens.float()).all())
        self.assertGreaterEqual(float(tokens.min()), 0.0)
        self.assertLessEqual(float(tokens.max()), 255.0)
        self.assertIsNone(foveator.raw_lambda_scale.grad)

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
        self.assertEqual(foveator._build_full_bin_edges_tensor().numel(), 209)

    def test_default_layout_uses_full_bin_half_offset_widths(self):
        foveator = LogRectilinearFoveator(16, 1280, 13, lambda_scale=1.0)
        widths = foveator._build_axis_edges_tensor().diff()
        self.assertEqual(widths.numel(), 13)
        self.assertTrue(torch.all(widths > 0))
        self.assertGreater(float(widths[0]), float(widths[6]))
        self.assertGreater(float(widths[-1]), float(widths[6]))
        self.assertLessEqual(float(widths[6]), 16.0)

    def test_bin_coordinates_are_contiguous_full_bin_edges(self):
        foveator = LogRectilinearFoveator(16, 1280, 13, lambda_scale=1.0)
        full_edges = foveator._build_full_bin_edges_tensor()
        lower, upper, _ = foveator.get_bin_coordinates()
        self.assertTrue(torch.allclose(lower[0, 0, :, 0], full_edges[:16]))
        self.assertTrue(torch.allclose(upper[0, 0, :, 0], full_edges[1:17]))
        self.assertTrue(torch.allclose(lower[0, :, 0, 1], full_edges[:16]))
        self.assertTrue(torch.allclose(upper[0, :, 0, 1], full_edges[1:17]))
        self.assertTrue(torch.allclose(lower[-1, 0, :, 0], full_edges[-17:-1]))
        self.assertTrue(torch.allclose(upper[-1, 0, :, 0], full_edges[-16:]))

    def test_visualization_paths_preserve_crop_shape(self):
        foveator = LogRectilinearFoveator(16, 1280, 13, lambda_scale=1.0)
        image = torch.zeros(3, 1280, 1280, dtype=torch.uint8)
        tokens = foveator.extract_foveated_image(image)
        blocky = foveator.generate_foveated_visualization(tokens)
        smooth = foveator.generate_smooth_foveated_visualization(tokens)
        self.assertEqual(tuple(blocky.shape), (3, 1280, 1280))
        self.assertEqual(tuple(smooth.shape), (3, 1280, 1280))

    def test_segmentation_target_is_detached_with_fixed_lambda_image_path(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
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
        self.assertFalse(tokens.requires_grad)
        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.isfinite(tokens.float()).all())
        self.assertTrue(torch.isfinite(target.float()).all())
        self.assertGreaterEqual(float(target.min()), 0.0)
        self.assertLessEqual(float(target.max()), 1.0)
        self.assertIsNone(foveator.raw_lambda_scale.grad)


if __name__ == "__main__":
    unittest.main()
