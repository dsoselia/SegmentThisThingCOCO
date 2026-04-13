from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import islice
from typing import List

import torch


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
    ) -> None:
        super().__init__()
        if axis_bins < 3 or axis_bins % 2 == 0:
            raise ValueError("[LogRectilinearFoveator]: axis_bins must be an odd integer >= 3.")
        if pattern_size <= 0 or pattern_size % 2 != 0:
            raise ValueError("[LogRectilinearFoveator]: pattern_size must be a positive even integer.")

        self.token_size = token_size
        self.pattern_size = pattern_size
        self.axis_bins = axis_bins
        self.exponent = exponent
        self.center_width = center_width or token_size

        edges = self._build_axis_edges()
        token_boxes = []
        for y0, y1 in zip(edges[:-1], edges[1:]):
            for x0, x1 in zip(edges[:-1], edges[1:]):
                token_boxes.append([x0, y0, x1, y1])

        token_boxes_tensor = torch.tensor(token_boxes, dtype=torch.int64)
        bin_lower, bin_upper = _subdivide_boxes(token_boxes_tensor, token_size)
        bin_area = ((bin_upper - bin_lower).prod(dim=-1)).clamp(min=1)

        self.register_buffer("token_boxes", token_boxes_tensor, persistent=False)
        self.register_buffer("bin_lower_pixel_coords", bin_lower, persistent=False)
        self.register_buffer("bin_upper_pixel_coords", bin_upper, persistent=False)
        self.register_buffer("bin_area", bin_area, persistent=False)

    def _scaled_log_rect(self, t: float) -> float:
        linear = t
        nonlinear = (math.exp(t**self.exponent) - 1.0) / (math.e - 1.0)
        return max(linear, nonlinear)

    def _build_axis_edges(self) -> list[int]:
        radius = self.pattern_size // 2
        center = radius
        side_bins = (self.axis_bins - 1) // 2
        half_center = self.center_width // 2

        if half_center < 1 or half_center >= radius:
            raise ValueError("[LogRectilinearFoveator]: center_width must be in [2, pattern_size).")

        distances = [half_center]
        for index in range(1, side_bins):
            t = index / side_bins
            offset = half_center + (radius - half_center) * self._scaled_log_rect(t)
            distances.append(offset)
        distances.append(radius)
        distances = _strictly_increasing_ints(distances, half_center, radius)

        edges = [0]
        edges.extend(center - distance for distance in reversed(distances[:-1]))
        edges.append(center + distances[0])
        edges.extend(center + distance for distance in distances[1:-1])
        edges.append(self.pattern_size)
        return _strictly_increasing_ints(edges, 0, self.pattern_size)

    def get_pattern_bounds_size(self) -> int:
        return self.pattern_size

    def get_num_tokens(self) -> int:
        return len(self.token_boxes)

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

        integral_image = compute_integral_image(images)
        return _compute_rect_integral_mean(
            integral_image,
            self.bin_lower_pixel_coords.to(images.device),
            self.bin_upper_pixel_coords.to(images.device),
            self.bin_area.to(images.device),
        )

    def get_in_bounds_tokens(
        self,
        image_size: torch.Tensor,
        crop_bounds: torch.Tensor,
        in_bounds_threshold: float = 0.0,
    ) -> torch.Tensor:
        box_coords = self.token_boxes.to(crop_bounds.device)
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

        output = torch.zeros((tokens.shape[1], self.pattern_size, self.pattern_size), dtype=tokens.dtype)
        lower = self.bin_lower_pixel_coords.cpu()
        upper = self.bin_upper_pixel_coords.cpu()
        source = tokens.cpu()

        for token_index in range(source.shape[0]):
            for row in range(self.token_size):
                for col in range(self.token_size):
                    x0 = lower[token_index, row, col, 0].item()
                    y0 = lower[token_index, row, col, 1].item()
                    x1 = upper[token_index, row, col, 0].item()
                    y1 = upper[token_index, row, col, 1].item()
                    output[:, y0:y1, x0:x1] = source[token_index, :, row, col].view(-1, 1, 1)

        return output
