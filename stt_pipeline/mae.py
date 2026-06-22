from __future__ import annotations

import torch


class FoveatedMAE(torch.nn.Module):
    def __init__(self, image_encoder: torch.nn.Module, feature_dim: int, token_size: int, foveator: torch.nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.foveator = foveator
        self.mask_token = torch.nn.Parameter(torch.zeros(3, token_size, token_size))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim, 3 * token_size * token_size),
        )
        self.token_size = token_size

    def forward(self, tokens: torch.Tensor, valid_token_mask: torch.Tensor | None, mask_ratio: float) -> tuple[torch.Tensor, dict]:
        batch, num_tokens = tokens.shape[:2]
        device = tokens.device
        token_mask = torch.rand((batch, num_tokens), device=device) < mask_ratio
        if valid_token_mask is not None:
            token_mask = token_mask & valid_token_mask

        masked_tokens = tokens.clone()
        masked_tokens[token_mask] = self.mask_token.view(1, 1, *self.mask_token.shape).expand(batch, num_tokens, -1, -1, -1)[token_mask]
        image_features, _ = self.image_encoder(masked_tokens, valid_token_mask)
        recon = self.decoder(image_features).view(batch, num_tokens, 3, self.token_size, self.token_size)

        # The resampling layout is allowed to learn through the visible input
        # tokens, but the reconstruction target must remain a stop-gradient
        # target. Otherwise lambda can reduce loss by moving both prediction
        # inputs and target values together.
        target = tokens.detach().float() / 255.0
        loss_map = (recon - target).pow(2).mean(dim=(2, 3, 4))
        loss = loss_map[token_mask].mean()
        return loss, {
            "loss": float(loss.detach().cpu()),
            "mask_fraction": float(token_mask.float().mean().detach().cpu()),
        }
