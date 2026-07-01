import math
import unittest

import torch

from segment_this_thing.foveation import LogRectilinearFoveator
from stt_pipeline.data import SegmentationBatch
from stt_pipeline.transforms import build_model_inputs, build_model_inputs_batch, project_mask_to_foveation, project_masks_to_foveation_batch
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

    def test_frozen_bin_coordinate_cache_matches_uncached_path(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        cached_lower, cached_upper, cached_area = foveator.get_bin_coordinates()
        cached_again_lower, _, _ = foveator.get_bin_coordinates()
        self.assertIs(cached_lower, cached_again_lower)

        foveator.set_lambda_learnable(True)
        uncached_lower, uncached_upper, uncached_area = foveator.get_bin_coordinates()
        self.assertTrue(torch.allclose(cached_lower, uncached_lower))
        self.assertTrue(torch.allclose(cached_upper, uncached_upper))
        self.assertTrue(torch.allclose(cached_area, uncached_area))

    def test_frozen_bin_coordinate_cache_invalidates_when_lambda_changes(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        original_lower, _, _ = foveator.get_bin_coordinates()
        with torch.no_grad():
            foveator.raw_lambda_scale.fill_(_raw_for_lambda(0.5))
        updated_lower, _, _ = foveator.get_bin_coordinates()
        self.assertIsNot(original_lower, updated_lower)
        self.assertFalse(torch.allclose(original_lower, updated_lower))

    def test_cached_and_uncached_foveation_outputs_match(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        generator = torch.Generator().manual_seed(11)
        image = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8, generator=generator)
        cached_tokens = foveator.extract_foveated_image(image)
        foveator.set_lambda_learnable(True)
        uncached_tokens = foveator.extract_foveated_image(image)
        self.assertTrue(torch.allclose(cached_tokens.float(), uncached_tokens.float()))

    def test_batched_foveation_matches_single_image_path(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        generator = torch.Generator().manual_seed(13)
        images = torch.randint(0, 256, (3, 3, 256, 256), dtype=torch.uint8, generator=generator)
        single = torch.stack([foveator.extract_foveated_image(image) for image in images])
        batched = foveator.extract_foveated_images(images)
        self.assertTrue(torch.allclose(single.float(), batched.float()))

    def test_batched_model_input_builder_matches_single_path(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        generator = torch.Generator().manual_seed(17)
        images = [torch.randint(0, 256, (300, 280, 3), dtype=torch.uint8, generator=generator) for _ in range(3)]
        centers = torch.tensor([[140, 150], [30, 40], [260, 250]], dtype=torch.int64)
        single_tokens = []
        single_valid = []
        single_bounds = []
        for image, center in zip(images, centers):
            tokens, valid, bounds = build_model_inputs(image, center, foveator)
            single_tokens.append(tokens)
            single_valid.append(valid)
            single_bounds.append(bounds)
        batch_tokens, batch_valid, batch_bounds = build_model_inputs_batch(images, centers, foveator)
        self.assertTrue(torch.allclose(torch.stack(single_tokens).float(), batch_tokens.float()))
        self.assertTrue(torch.equal(torch.stack(single_valid), batch_valid))
        self.assertTrue(torch.equal(torch.stack(single_bounds), batch_bounds))

    def test_batched_mask_projection_matches_single_path(self):
        foveator = LogRectilinearFoveator(4, 256, 13, lambda_scale=1.0, lambda_learnable=False)
        masks = []
        for offset in (0, 15, 30):
            mask = torch.zeros((300, 280), dtype=torch.bool)
            mask[70 + offset : 190 + offset, 80:210] = True
            masks.append(mask)
        centers = torch.tensor([[140, 150], [30, 40], [260, 250]], dtype=torch.int64)
        single_targets = []
        single_bounds = []
        for mask, center in zip(masks, centers):
            target, bounds = project_mask_to_foveation(foveator, mask, center)
            single_targets.append(target)
            single_bounds.append(bounds)
        batch_targets, batch_bounds = project_masks_to_foveation_batch(foveator, masks, centers)
        self.assertTrue(torch.allclose(torch.stack(single_targets).float(), batch_targets.float()))
        self.assertTrue(torch.equal(torch.stack(single_bounds), batch_bounds))

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
