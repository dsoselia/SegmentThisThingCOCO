from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn

from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_model(size: str, image_size: int = 1024, patch_size: int = 16) -> nn.Module:
    key = size.lower()
    if key == "b":
        key = "vit_b"
    if key == "l":
        key = "vit_l"
    if key == "h":
        key = "vit_h"
    encoder_configs = {
        "vit_b": {
            "encoder_embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "encoder_global_attn_indexes": [2, 5, 8, 11],
        },
        "vit_l": {
            "encoder_embed_dim": 1024,
            "encoder_depth": 24,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [5, 11, 17, 23],
        },
        "vit_h": {
            "encoder_embed_dim": 1280,
            "encoder_depth": 32,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [7, 15, 23, 31],
        },
    }
    encoder_config = encoder_configs[key]
    prompt_embed_dim = 256
    image_embedding_size = image_size // patch_size
    model = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_config["encoder_depth"],
            embed_dim=encoder_config["encoder_embed_dim"],
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_config["encoder_num_heads"],
            patch_size=patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_config["encoder_global_attn_indexes"],
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    model.train()
    return model


def normalize_image_batch(images: torch.Tensor, sam_model: nn.Module) -> torch.Tensor:
    pixel_mean = sam_model.pixel_mean.view(1, -1, 1, 1).to(images.device)
    pixel_std = sam_model.pixel_std.view(1, -1, 1, 1).to(images.device)
    return (images - pixel_mean) / pixel_std


def load_backbone_into_sam(sam_model: nn.Module, backbone_path: str) -> dict[str, Any]:
    state = torch.load(backbone_path, map_location="cpu", weights_only=False)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = sam_model.image_encoder.load_state_dict(model_state, strict=False)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def load_sam_checkpoint(sam_model: nn.Module, checkpoint_path: str) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    filtered_state = {
        key: value for key, value in model_state.items() if key not in {"pixel_mean", "pixel_std"}
    }
    missing, unexpected = sam_model.load_state_dict(filtered_state, strict=False)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


class SamMaeBackbone(nn.Module):
    def __init__(self, image_encoder: nn.Module):
        super().__init__()
        self.patch_embed = image_encoder.patch_embed
        self.pos_embed = image_encoder.pos_embed
        self.blocks = image_encoder.blocks


class SamBackboneMAE(nn.Module):
    def __init__(self, sam_model: nn.Module):
        super().__init__()
        self.backbone = SamMaeBackbone(sam_model.image_encoder)
        self.register_buffer("pixel_mean", sam_model.pixel_mean.detach().clone(), persistent=True)
        self.register_buffer("pixel_std", sam_model.pixel_std.detach().clone(), persistent=True)
        embed_dim = self.backbone.patch_embed.proj.out_channels
        patch_size = self.backbone.patch_embed.proj.kernel_size[0]
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.patch_dim),
        )
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        batch, channels, height, width = images.shape
        h = height // patch
        w = width // patch
        x = images.reshape(batch, channels, h, patch, w, patch)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch, h * w, patch * patch * channels)
        return x

    def _random_mask(self, batch: int, grid_h: int, grid_w: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        num_tokens = grid_h * grid_w
        num_mask = max(1, int(round(mask_ratio * num_tokens)))
        noise = torch.rand(batch, num_tokens, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(batch, num_tokens, device=device, dtype=torch.bool)
        mask.scatter_(1, ids[:, :num_mask], True)
        return mask.view(batch, grid_h, grid_w)

    def forward(self, images: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, dict[str, float]]:
        normalized = normalize_image_batch(images, self)
        x = self.backbone.patch_embed(normalized)
        batch, grid_h, grid_w, embed_dim = x.shape
        mask = self._random_mask(batch, grid_h, grid_w, mask_ratio, x.device)
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(batch, grid_h, grid_w, embed_dim), x)
        for block in self.backbone.blocks:
            x = block(x)
        pred = self.reconstruction_head(x.reshape(batch, grid_h * grid_w, embed_dim))
        target = self.patchify(images / 255.0)
        mask_flat = mask.reshape(batch, grid_h * grid_w, 1).float()
        loss = ((pred - target) ** 2 * mask_flat).sum() / (mask_flat.sum() * target.shape[-1]).clamp(min=1.0)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "mask_ratio": float(mask_ratio),
            "masked_fraction": float(mask_flat.mean().detach().cpu()),
        }
        return loss, metrics


class SamSegmentationModel(nn.Module):
    def __init__(self, sam_model: nn.Module):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.register_buffer("pixel_mean", sam_model.pixel_mean.detach().clone(), persistent=True)
        self.register_buffer("pixel_std", sam_model.pixel_std.detach().clone(), persistent=True)
        self._freeze_unused_prompt_paths()

    def _freeze_unused_prompt_paths(self) -> None:
        # This baseline trains only with a single positive point prompt.
        # Box embeddings and mask-input downscaling are never exercised in that regime,
        # so leaving them trainable causes DDP unused-parameter failures.
        for embedding in self.prompt_encoder.point_embeddings[2:]:
            embedding.weight.requires_grad_(False)
        for parameter in self.prompt_encoder.mask_downscaling.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.image_encoder(normalize_image_batch(images, self))
        dense_pe = self.prompt_encoder.get_dense_pe()
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        pred_masks, pred_ious = self._predict_masks_one_prompt_per_image(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        return pred_masks[:, 1:, :, :], pred_ious[:, 1:]

    def _predict_masks_one_prompt_per_image(
        self,
        *,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder = self.mask_decoder
        batch_size = image_embeddings.shape[0]
        output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe.expand(batch_size, -1, -1, -1)
        batch_size, channels, height, width = src.shape

        hs, src = decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + decoder.num_mask_tokens), :]

        src = src.transpose(1, 2).view(batch_size, channels, height, width)
        upscaled_embedding = decoder.output_upscaling(src)
        hyper_in = torch.stack(
            [mlp(mask_tokens_out[:, idx, :]) for idx, mlp in enumerate(decoder.output_hypernetworks_mlps)],
            dim=1,
        )
        batch_size, channels, height, width = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(batch_size, channels, height * width)).view(
            batch_size, -1, height, width
        )
        iou_pred = decoder.iou_prediction_head(iou_token_out)
        return masks, iou_pred
