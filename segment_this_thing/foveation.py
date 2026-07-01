from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import islice
from typing import List

import torch
import torch.nn.functional as F


def is_monotonically_increasing(vals: List[int]) -> bool:
    return all(y > x for x, y in zip(vals[:-1], vals[1:]))


def compute_integral_image(image: torch.Tensor):
    """
    Computes the integral image of the given input image.

    Parameters:
    - image (C, H, W): The input image tensor with dimensions C x H x W.

    Returns:
    - integral image (C, H, W): An integral image tensor with the same dimensions as the input
    """
    padded = torch.nn.functional.pad(image, (1, 0, 1, 0), mode="constant", value=0)
    return padded.cumsum(dim=2, dtype=torch.int32).cumsum(dim=1, dtype=torch.int32)


def generate_grid_coords_2d(grid_size):
    x = torch.arange(grid_size)
    return torch.stack(torch.meshgrid(x, x, indexing="xy"), dim=-1)


def _compute_rect_integral_mean(
    integral_image: torch.Tensor,
    lower_pixel_coords: torch.Tensor,
    upper_pixel_coords: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.stack(
            [
                integral_image_channel[
                    upper_pixel_coords[..., 1], upper_pixel_coords[..., 0]
                ]
                - integral_image_channel[
                    upper_pixel_coords[..., 1], lower_pixel_coords[..., 0]
                ]
                - integral_image_channel[
                    lower_pixel_coords[..., 1], upper_pixel_coords[..., 0]
                ]
                + integral_image_channel[
                    lower_pixel_coords[..., 1], lower_pixel_coords[..., 0]
                ]
                for integral_image_channel in integral_image
            ],
            1,
        )
        .floor_divide(area.view(-1, 1, *area.shape[-2:]).int())
        .byte()
    )


class Foveator(torch.nn.Module):
    """
    The Foveator class is a PyTorch module designed to extract and reconstruct foveated images.
    Foveation is a process that simulates the varying resolution of human vision, where the center
    of the visual field is seen in high detail, and the periphery is seen in lower detail.
    """

    def __init__(
        self, token_size: int, strides: List[int], grid_sizes: List[int]
    ) -> None:
        """
        Initializes the Foveator module, which is used to extract and reconstruct foveated images.

        Parameters:
        - token_size (int): The size of each token in the foveated image.
        - strides (List[int]): A list of stride values for each level of foveation. Must be monotonically increasing.
        - grid_sizes (List[int]): A list of grid sizes corresponding to each level of foveation. Must have the same length as strides.

        Raises:
        - ValueError: If the lengths of strides and grid_sizes do not match, or if they do not lead to monotonically increasing block sizes, or if they do not allow for nestable block sizes.
        """

        super().__init__()

        num_levels = len(strides)
        if len(grid_sizes) != num_levels:
            raise ValueError(
                "[Foveator]: Constructor arguments 'strides' and 'grid_sizes' must have the same length."
            )

        if not is_monotonically_increasing(strides):
            raise ValueError(
                "[Foveator]: Constructor agrument 'strides' should have monotonically increasing values."
            )

        # Block sizes are in multiples of stride-1 tokens.
        self.block_sizes = [
            stride * grid_size for stride, grid_size in zip(strides, grid_sizes)
        ]
        if not is_monotonically_increasing(self.block_sizes):
            raise ValueError(
                "[Foveator]: Constructor arguments 'strides' and 'grid_sizes' do not lead to monotonically increasing block sizes."
            )

        token_corner_indices_by_level = []

        self.ring_thickness_tokens = [None]

        for level_index, (stride, grid_size, block_size) in enumerate(
            zip(strides, grid_sizes, self.block_sizes)
        ):
            grid_coords = generate_grid_coords_2d(grid_size)
            offset = (self.block_sizes[-1] - block_size) // 2
            if level_index == 0:
                token_corner_indices_by_level.append(
                    offset + stride * grid_coords.flatten(0, 1)
                )
                continue

            prior_block_size = self.block_sizes[level_index - 1]
            redundant_grid_size = prior_block_size // stride
            if stride * redundant_grid_size != prior_block_size:
                raise ValueError(
                    "[Foveator]: Constructor arguments 'strides' and 'grid_sizes' do not lead to nestable block sizes."
                )
            ring_thickness_tokens = (grid_size - redundant_grid_size) // 2
            if ring_thickness_tokens * 2 + redundant_grid_size != grid_size:
                raise ValueError(
                    "[Foveator]: Constructor arguments 'strides' and 'grid_sizes' do not lead to evenly nestable block sizes."
                )
            # upper rows, include all columns
            level_corner_indices = [grid_coords[:ring_thickness_tokens].flatten(0, 1)]
            # central rows, exclude already-covered central columns
            for row in grid_coords[ring_thickness_tokens:-ring_thickness_tokens]:
                level_corner_indices.append(row[:ring_thickness_tokens])
                level_corner_indices.append(row[-ring_thickness_tokens:])
            # lower rows, include all columns
            level_corner_indices.append(
                grid_coords[-ring_thickness_tokens:].flatten(0, 1)
            )

            token_corner_indices_by_level.append(
                offset + stride * torch.cat(level_corner_indices)
            )

            self.ring_thickness_tokens.append(ring_thickness_tokens)

        self.token_size = token_size
        self.strides_by_level = strides
        self.grid_sizes_by_level = grid_sizes
        self.num_tokens_by_level = [
            len(corners) for corners in token_corner_indices_by_level
        ]
        self.register_buffer(
            "token_corner_indices",
            token_size * torch.cat(token_corner_indices_by_level),
            persistent=False,
        )
        self.register_buffer(
            "token_strides",
            torch.cat(
                [
                    torch.full((len(corners),), stride)
                    for corners, stride in zip(token_corner_indices_by_level, strides)
                ]
            ),
        )

    def get_pattern_bounds_size(self) -> int:
        return self.block_sizes[-1] * self.token_size

    def get_num_tokens(self) -> int:
        return len(self.token_strides)

    def extract_foveated_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        # This function extracts a set of tokens representing a foveated version of the input image.
        # Input: images (Tensor of shape [C, H, W]).
        # Output: foveated tokens (Tensor of shape [N, C, H, W])
        """
        if not images.ndim == 3:
            raise ValueError(
                "[Foveator.extract_foveated_image]: Expected 3D input Tensor."
            )

        expected_input_size = self.token_size * self.block_sizes[-1]
        if (
            images.shape[-2] != expected_input_size
            or images.shape[-1] != expected_input_size
        ):
            raise ValueError(
                f"[Foveator.extract_foveated_image]: Expected square image of size {expected_input_size}"
            )
        if images.shape[-3] != 3:
            raise ValueError(
                f"[Foveator.extract_foveated_image]: Expected 3-channel image."
            )

        if images.dtype != torch.uint8:
            raise ValueError("[Foveator.extract_foveated_image]: Expected byte images.")

        device = images.device

        integral_image = compute_integral_image(images)

        # self.token_corner_indices is (N, U)
        # self.token_strides is (N)
        # generate_grid_coords_2d return (H, W, U)
        # All get mapped to (N, H, W, U)
        lower_pixel_coords = self.token_corner_indices.view(
            -1, 1, 1, 2
        ) + self.token_strides.view(-1, 1, 1, 1) * generate_grid_coords_2d(
            self.token_size
        ).unsqueeze(0).to(device)
        upper_pixel_coords = lower_pixel_coords + self.token_strides.view(-1, 1, 1, 1)

        return _compute_rect_integral_mean(
            integral_image,
            lower_pixel_coords,
            upper_pixel_coords,
            self.token_strides.square().view(-1, 1, 1),
        )

    def get_in_bounds_tokens(
        self,
        image_size: torch.Tensor,
        crop_bounds: torch.Tensor,
        in_bounds_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        # This function returns a binary mask indicating which tokens are sufficiently in bounds to be considered valid.
        # Input:
        # - image_size (Tensor of shape [2]): The size of the input image (width, height).
        # - crop_bounds (Tensor of shape [2, 2]): The bounds of the crop region ([[x1, y1], [x2, y2]]).
        # - in_bounds_threshold (float): The threshold for considering a token to be in bounds (tokens may be partially in bounds)
        # Output: valid_token_mask (Tensor of shape [N]): A binary mask indicating which tokens are valid.
        """
        foveation_offset = crop_bounds[0]
        token_corner_coordinates = foveation_offset + self.token_corner_indices

        bounded_lower_corners = token_corner_coordinates.clamp(min=0)
        bounded_upper_corners = torch.minimum(
            token_corner_coordinates + self.token_size * self.token_strides.view(-1, 1),
            image_size,
        )

        area_per_pixel = self.token_strides.square()

        in_bounds_area = (
            (bounded_upper_corners - bounded_lower_corners)
            .clamp(min=0)
            .prod(dim=-1)
            .float()
        )
        valid_token_mask = (
            in_bounds_area / area_per_pixel.float()
        ) > in_bounds_threshold

        return valid_token_mask

    def generate_foveated_visualization(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        # This function reconstructs an image given a set of foveated tokens.
        # Input: foveated tokens (Tensor of shape [N, C, H, W]).
        # Output: image (Tensor of shape [C, H, W])
        """
        if not tokens.ndim == 4:
            raise ValueError(
                "[Foveator.generate_foveated_visualization]: Expected 4D input Tensor (N, C, H, W)."
            )
        if tokens.shape[0] != len(self.token_strides):
            raise ValueError(
                f"[Foveator.generate_foveated_visualization]: Expected {len(self.token_strides)} tokens"
            )
        if tokens.shape[-2] != self.token_size or tokens.shape[-1] != self.token_size:
            raise ValueError(
                f"[Foveator.generate_foveated_visualization]: Expected square tokens of size {self.token_size}"
            )

        C = tokens.shape[1]
        reconstruction_size = self.block_sizes[-1] * self.token_size

        reconstructed_image = torch.empty((C, reconstruction_size, reconstruction_size))

        tokens_by_level = torch.split(tokens, self.num_tokens_by_level)

        # insert the innermost block

        # unflatten | N, C, H, W     ->  I, J, C, H, W
        # permute   | I, J, C, H, W  ->  C, I, H, J, W
        # flatten   | C, I, H, J, W  ->  C, H, W
        inner_block = (
            tokens_by_level[0]
            .unflatten(0, (self.grid_sizes_by_level[0], self.grid_sizes_by_level[0]))
            .permute(2, 0, 3, 1, 4)
            .flatten(3, 4)
            .flatten(1, 2)
        )

        offset = self.token_size * (self.block_sizes[-1] - self.block_sizes[0]) // 2

        reconstructed_image[:, offset:-offset, offset:-offset] = inner_block

        # build up the reconstruction ring by ring
        for tokens, stride, grid_size, block_size, ring_thickness in islice(
            zip(
                tokens_by_level,
                self.strides_by_level,
                self.grid_sizes_by_level,
                self.block_sizes,
                self.ring_thickness_tokens,
            ),
            1,
            None,
        ):
            upsampled_tokens = tokens.repeat_interleave(stride, -1).repeat_interleave(
                stride, -2
            )
            row_height = upsampled_tokens.shape[-2]
            offset = self.token_size * (self.block_sizes[-1] - block_size) // 2
            i = 0
            for row in range(grid_size):
                if row < ring_thickness or row >= (grid_size - ring_thickness):
                    # these rows span the block width
                    reconstructed_image[
                        :,
                        offset + row * row_height : offset + (row + 1) * row_height,
                        offset : offset + self.token_size * block_size,
                    ] = (
                        upsampled_tokens[i : i + grid_size]
                        .permute(1, 2, 0, 3)
                        .flatten(2)
                    )
                    i += grid_size
                else:
                    # these rows are split in to by the higher-resolution center region
                    reconstructed_image[
                        :,
                        offset + row * row_height : offset + (row + 1) * row_height,
                        offset : offset + self.token_size * stride * ring_thickness,
                    ] = (
                        upsampled_tokens[i : i + ring_thickness]
                        .permute(1, 2, 0, 3)
                        .flatten(2)
                    )
                    i += ring_thickness
                    reconstructed_image[
                        :,
                        offset + row * row_height : offset + (row + 1) * row_height,
                        offset
                        + self.token_size
                        * stride
                        * (grid_size - ring_thickness) : offset
                        + self.token_size * block_size,
                    ] = (
                        upsampled_tokens[i : i + ring_thickness]
                        .permute(1, 2, 0, 3)
                        .flatten(2)
                    )
                    i += ring_thickness

        return reconstructed_image


def _strictly_increasing_ints(values: list[float], lower: int, upper: int) -> list[int]:
    ints = [int(round(v)) for v in values]
    ints[0] = max(lower, ints[0])
    for index in range(1, len(ints)):
        ints[index] = max(ints[index], ints[index - 1] + 1)
    ints[-1] = min(upper, ints[-1])
    for index in range(len(ints) - 2, -1, -1):
        ints[index] = min(ints[index], ints[index + 1] - 1)
    return ints


def _subdivide_boxes(token_boxes: torch.Tensor, token_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.empty((token_boxes.shape[0], token_size, token_size, 2), dtype=torch.int64)
    upper = torch.empty_like(lower)

    for token_index, (x0, y0, x1, y1) in enumerate(token_boxes.tolist()):
        x_edges = [x0 + (x1 - x0) * i / token_size for i in range(token_size + 1)]
        y_edges = [y0 + (y1 - y0) * i / token_size for i in range(token_size + 1)]
        x_edges = _strictly_increasing_ints(x_edges, x0, x1)
        y_edges = _strictly_increasing_ints(y_edges, y0, y1)
        for row in range(token_size):
            for col in range(token_size):
                lower[token_index, row, col, 0] = x_edges[col]
                lower[token_index, row, col, 1] = y_edges[row]
                upper[token_index, row, col, 0] = x_edges[col + 1]
                upper[token_index, row, col, 1] = y_edges[row + 1]

    return lower, upper


class LogRectilinearFoveator(torch.nn.Module):
    """
    Box-based foveator derived from the log-rectilinear transformation.

    Each token owns an explicit axis-aligned rectangular box in the prompt-centered crop.
    The box is subdivided into token_size x token_size bins and each bin is averaged from an
    integral image. This preserves the STT model interface while changing only the tokenizer.
    """

    def __init__(
        self,
        token_size: int,
        pattern_size: int,
        axis_bins: int,
        exponent: float = 4.0,
        center_width: int | None = None,
        lambda_scale: float = 1.0,
        lambda_learnable: bool = False,
    ) -> None:
        super().__init__()
        if axis_bins < 3 or axis_bins % 2 == 0:
            raise ValueError("[LogRectilinearFoveator]: axis_bins must be an odd integer >= 3.")
        if pattern_size <= 0 or pattern_size % 2 != 0:
            raise ValueError("[LogRectilinearFoveator]: pattern_size must be a positive even integer.")
        if lambda_scale <= 0:
            raise ValueError("[LogRectilinearFoveator]: lambda_scale must be positive.")

        self.token_size = token_size
        self.pattern_size = pattern_size
        self.axis_bins = axis_bins
        self.exponent = exponent
        self.center_width = center_width or token_size
        self.lambda_epsilon = 1e-6
        raw_lambda = math.log(math.expm1(max(lambda_scale - self.lambda_epsilon, 1e-8)))
        self.raw_lambda_scale = torch.nn.Parameter(
            torch.tensor(raw_lambda, dtype=torch.float32),
            requires_grad=lambda_learnable,
        )
        self._bin_coordinate_cache_key = None
        self._bin_coordinate_cache = None

    @property
    def lambda_scale(self) -> torch.Tensor:
        """Positive, differentiable lambda scale."""
        return F.softplus(self.raw_lambda_scale) + self.lambda_epsilon

    def set_lambda_learnable(self, enabled: bool) -> None:
        if enabled and not self.raw_lambda_scale.requires_grad:
            self._bin_coordinate_cache_key = None
            self._bin_coordinate_cache = None
        self.raw_lambda_scale.requires_grad_(enabled)

    def _scaled_log_rect(self, t: float) -> float:
        """Paper-style log-rectilinear normalized radial mapping."""
        linear = t
        nonlinear = (math.exp(t**self.exponent) - 1.0) / (math.e - 1.0)
        return max(linear, nonlinear)

    def _log_rect_offsets(self, du: torch.Tensor, buffer_half: float, crop_half: float) -> torch.Tensor:
        ad = du.abs()
        lam = self.lambda_scale * float(self.pattern_size) / (math.e - 1.0)
        exp_term = lam * (torch.exp((ad / buffer_half) ** self.exponent) - 1.0)
        return torch.maximum(ad, exp_term) * du.sign()

    @staticmethod
    def _make_strict_integer_edges(raw_edges: torch.Tensor, total_size: int) -> torch.Tensor:
        """Clamp OpenCL grid boundaries into a stable monotone SAT edge array."""
        edges = raw_edges.detach().round().clamp(0, total_size).to(torch.int64).cpu().tolist()
        edges[0] = 0
        edges[-1] = total_size
        for index in range(1, len(edges)):
            edges[index] = max(edges[index], edges[index - 1] + 1)
        edges[-1] = total_size
        for index in range(len(edges) - 2, -1, -1):
            edges[index] = min(edges[index], edges[index + 1] - 1)
        return raw_edges.new_tensor(edges, dtype=raw_edges.dtype)

    def _build_full_bin_edges_tensor(self) -> torch.Tensor:
        """Build one log-rectilinear source-space edge per packed buffer bin.

        The original OpenCL implementation stores rectangle boundaries halfway
        between neighboring mapped reduced-buffer samples. The STT tokenizer's
        packed buffer has ``axis_bins * token_size`` bins per axis, so we build
        all of those boundaries before grouping bins into tokens.
        """
        packed_size = self.axis_bins * self.token_size
        if packed_size <= 0:
            raise ValueError("[LogRectilinearFoveator]: invalid axis_bins/token_size.")

        device = self.raw_lambda_scale.device
        dtype = self.raw_lambda_scale.dtype
        buffer_half = packed_size / 2.0
        boundary_index = torch.arange(packed_size + 1, device=device, dtype=dtype)
        left_du = boundary_index - 1.0 - buffer_half
        right_du = boundary_index - buffer_half
        left_delta = self._log_rect_offsets(left_du, buffer_half, self.pattern_size / 2.0)
        right_delta = self._log_rect_offsets(right_du, buffer_half, self.pattern_size / 2.0)
        grid_offsets = torch.floor((left_delta + right_delta) / 2.0)
        raw_edges = self.pattern_size / 2.0 + grid_offsets
        return self._make_strict_integer_edges(raw_edges, self.pattern_size)

    def _build_axis_edges_tensor(self) -> torch.Tensor:
        """Build macro token edges from full half-offset log-rectilinear bins.

        The previous Zaratan implementation interpolated almost linearly across
        the 1280-pixel crop, producing a weak foveation. Here we follow the
        log-rectilinear buffer-to-crop mapping from Li et al.: a small
        axis-aligned buffer of ``axis_bins * token_size`` pixels is mapped to
        the full prompt-centered crop with an identity foveal band and
        exponentially larger peripheral bins. Rectangle boundaries are offset
        halfway between neighboring reduced-buffer samples, matching the
        original OpenCL grid construction.
        """
        full_edges = self._build_full_bin_edges_tensor()
        return full_edges[:: self.token_size]

    def _build_axis_edges(self) -> list[int]:
        """Integer edge helper retained for visualization/debug scripts."""
        return [int(round(value)) for value in self._build_axis_edges_tensor().detach().cpu().tolist()]

    def get_token_boxes(self) -> torch.Tensor:
        edges = self._build_axis_edges_tensor()
        y0, x0 = torch.meshgrid(edges[:-1], edges[:-1], indexing="ij")
        y1, x1 = torch.meshgrid(edges[1:], edges[1:], indexing="ij")
        return torch.stack([x0, y0, x1, y1], dim=-1).reshape(-1, 4)

    def _compute_bin_coordinates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edges = self._build_full_bin_edges_tensor()
        axis_index = torch.arange(self.axis_bins, device=edges.device)
        y_index = axis_index.repeat_interleave(self.axis_bins)
        x_index = axis_index.repeat(self.axis_bins)
        local = torch.arange(self.token_size, device=edges.device)
        x_edges = x_index[:, None] * self.token_size + local[None, :]
        y_edges = y_index[:, None] * self.token_size + local[None, :]
        x_lower = edges[x_edges][:, None, :].expand(-1, self.token_size, -1)
        x_upper = edges[x_edges + 1][:, None, :].expand(-1, self.token_size, -1)
        y_lower = edges[y_edges][:, :, None].expand(-1, -1, self.token_size)
        y_upper = edges[y_edges + 1][:, :, None].expand(-1, -1, self.token_size)
        lower = torch.stack([x_lower, y_lower], dim=-1)
        upper = torch.stack([x_upper, y_upper], dim=-1)
        area = ((upper - lower).prod(dim=-1)).clamp_min(1e-6)
        return lower, upper, area

    def get_bin_coordinates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.raw_lambda_scale.requires_grad:
            return self._compute_bin_coordinates()
        key = (
            self.raw_lambda_scale.device.type,
            self.raw_lambda_scale.device.index,
            self.raw_lambda_scale.dtype,
            int(self.raw_lambda_scale._version),
        )
        if self._bin_coordinate_cache_key != key or self._bin_coordinate_cache is None:
            self._bin_coordinate_cache = self._compute_bin_coordinates()
            self._bin_coordinate_cache_key = key
        return self._bin_coordinate_cache

    @staticmethod
    def _sample_integral(integral: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        height, width = integral.shape[-2:]
        normalized = coords.clone()
        normalized[..., 0] = normalized[..., 0] * (2.0 / (width - 1)) - 1.0
        normalized[..., 1] = normalized[..., 1] * (2.0 / (height - 1)) - 1.0
        flat_grid = normalized.reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            integral.unsqueeze(0),
            flat_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(0).squeeze(-1).reshape(integral.shape[0], *coords.shape[:-1])

    @staticmethod
    def _sample_integral_batch(integral: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        height, width = integral.shape[-2:]
        normalized = coords.to(device=integral.device, dtype=integral.dtype).clone()
        normalized[..., 0] = normalized[..., 0] * (2.0 / (width - 1)) - 1.0
        normalized[..., 1] = normalized[..., 1] * (2.0 / (height - 1)) - 1.0
        batch = integral.shape[0]
        flat_grid = normalized.reshape(1, -1, 1, 2).expand(batch, -1, -1, -1)
        sampled = F.grid_sample(
            integral,
            flat_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).reshape(batch, integral.shape[1], *coords.shape[:-1])

    def get_pattern_bounds_size(self) -> int:
        return self.pattern_size

    def get_num_tokens(self) -> int:
        return self.axis_bins * self.axis_bins

    def extract_foveated_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 3:
            raise ValueError("[LogRectilinearFoveator.extract_foveated_image]: Expected 3D input Tensor.")
        if images.shape[-2] != self.pattern_size or images.shape[-1] != self.pattern_size:
            raise ValueError(
                f"[LogRectilinearFoveator.extract_foveated_image]: Expected square image of size {self.pattern_size}"
            )
        if images.shape[-3] != 3:
            raise ValueError("[LogRectilinearFoveator.extract_foveated_image]: Expected 3-channel image.")
        if images.dtype != torch.uint8:
            raise ValueError("[LogRectilinearFoveator.extract_foveated_image]: Expected byte images.")

        integral_image = F.pad(images.float(), (1, 0, 1, 0), mode="constant", value=0.0)
        integral_image = integral_image.cumsum(dim=2).cumsum(dim=1)
        lower, upper, area = self.get_bin_coordinates()
        top_left = self._sample_integral(integral_image, lower)
        top_right_coords = torch.stack([upper[..., 0], lower[..., 1]], dim=-1)
        bottom_left_coords = torch.stack([lower[..., 0], upper[..., 1]], dim=-1)
        summed = (
            self._sample_integral(integral_image, upper)
            - self._sample_integral(integral_image, top_right_coords)
            - self._sample_integral(integral_image, bottom_left_coords)
            + top_left
        )
        return (summed / area.unsqueeze(0)).clamp(0.0, 255.0).permute(1, 0, 2, 3)

    def extract_foveated_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("[LogRectilinearFoveator.extract_foveated_images]: Expected 4D input Tensor.")
        if images.shape[-2] != self.pattern_size or images.shape[-1] != self.pattern_size:
            raise ValueError(
                f"[LogRectilinearFoveator.extract_foveated_images]: Expected square images of size {self.pattern_size}"
            )
        if images.shape[-3] != 3:
            raise ValueError("[LogRectilinearFoveator.extract_foveated_images]: Expected 3-channel images.")
        if images.dtype != torch.uint8:
            raise ValueError("[LogRectilinearFoveator.extract_foveated_images]: Expected byte images.")

        integral_image = F.pad(images.float(), (1, 0, 1, 0), mode="constant", value=0.0)
        integral_image = integral_image.cumsum(dim=3).cumsum(dim=2)
        lower, upper, area = self.get_bin_coordinates()
        top_left = self._sample_integral_batch(integral_image, lower)
        top_right_coords = torch.stack([upper[..., 0], lower[..., 1]], dim=-1)
        bottom_left_coords = torch.stack([lower[..., 0], upper[..., 1]], dim=-1)
        summed = (
            self._sample_integral_batch(integral_image, upper)
            - self._sample_integral_batch(integral_image, top_right_coords)
            - self._sample_integral_batch(integral_image, bottom_left_coords)
            + top_left
        )
        return (summed / area.view(1, 1, *area.shape)).clamp(0.0, 255.0).permute(0, 2, 1, 3, 4)

    def get_in_bounds_tokens(
        self,
        image_size: torch.Tensor,
        crop_bounds: torch.Tensor,
        in_bounds_threshold: float = 0.0,
    ) -> torch.Tensor:
        box_coords = self.get_token_boxes().to(crop_bounds.device)
        lower = crop_bounds[0] + box_coords[:, :2]
        upper = crop_bounds[0] + box_coords[:, 2:]
        bounded_lower = lower.clamp(min=0)
        bounded_upper = torch.minimum(upper, image_size)
        in_bounds_area = ((bounded_upper - bounded_lower).clamp(min=0).prod(dim=-1)).float()
        total_area = ((box_coords[:, 2:] - box_coords[:, :2]).prod(dim=-1)).float()
        return (in_bounds_area / total_area.clamp(min=1.0)) > in_bounds_threshold

    def generate_foveated_visualization(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(
                "[LogRectilinearFoveator.generate_foveated_visualization]: Expected 4D input Tensor (N, C, H, W)."
            )
        if tokens.shape[0] != self.get_num_tokens():
            raise ValueError(
                f"[LogRectilinearFoveator.generate_foveated_visualization]: Expected {self.get_num_tokens()} tokens"
            )
        if tokens.shape[-2] != self.token_size or tokens.shape[-1] != self.token_size:
            raise ValueError(
                f"[LogRectilinearFoveator.generate_foveated_visualization]: Expected square tokens of size {self.token_size}"
            )

        device = tokens.device
        packed_size = self.axis_bins * self.token_size
        packed = (
            tokens.reshape(self.axis_bins, self.axis_bins, tokens.shape[1], self.token_size, self.token_size)
            .permute(2, 0, 3, 1, 4)
            .reshape(tokens.shape[1], packed_size, packed_size)
        )
        edges = self._build_full_bin_edges_tensor().to(device=device)
        coords = torch.arange(self.pattern_size, device=device, dtype=edges.dtype) + 0.5
        bin_index = torch.bucketize(coords, edges[1:-1]).clamp(0, packed_size - 1)
        return packed[:, bin_index[:, None], bin_index[None, :]]

    def generate_smooth_foveated_visualization(self, tokens: torch.Tensor) -> torch.Tensor:
        """Diagnostic inverse-logrect preview using bilinear lookup in packed-token space.

        This mirrors the original decoder's smooth display behavior for visual
        inspection only. Training targets and segmentation metric reconstruction
        intentionally continue to use ``generate_foveated_visualization``.
        """
        if tokens.ndim != 4:
            raise ValueError(
                "[LogRectilinearFoveator.generate_smooth_foveated_visualization]: Expected 4D input Tensor (N, C, H, W)."
            )
        if tokens.shape[0] != self.get_num_tokens():
            raise ValueError(
                f"[LogRectilinearFoveator.generate_smooth_foveated_visualization]: Expected {self.get_num_tokens()} tokens"
            )
        if tokens.shape[-2] != self.token_size or tokens.shape[-1] != self.token_size:
            raise ValueError(
                f"[LogRectilinearFoveator.generate_smooth_foveated_visualization]: Expected square tokens of size {self.token_size}"
            )

        device = tokens.device
        dtype = tokens.dtype if tokens.is_floating_point() else torch.float32
        packed_size = self.axis_bins * self.token_size
        packed = (
            tokens.to(dtype)
            .reshape(self.axis_bins, self.axis_bins, tokens.shape[1], self.token_size, self.token_size)
            .permute(2, 0, 3, 1, 4)
            .reshape(1, tokens.shape[1], packed_size, packed_size)
        )

        edges = self._build_full_bin_edges_tensor().to(device=device, dtype=dtype)
        coords = torch.arange(self.pattern_size, device=device, dtype=dtype) + 0.5
        bin_index = torch.bucketize(coords, edges[1:-1]).clamp(0, packed_size - 1)
        left = edges[bin_index]
        right = edges[bin_index + 1]
        frac = ((coords - left) / (right - left).clamp_min(1e-6)).clamp(0.0, 1.0)
        packed_coord = bin_index.to(dtype) + frac
        normalized = packed_coord * (2.0 / float(packed_size - 1)) - 1.0
        grid_y, grid_x = torch.meshgrid(normalized, normalized, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return F.grid_sample(
            packed,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
