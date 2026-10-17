"""Visualization decoder mapping latent representations back to pixel frames.

Trained strictly with stop-gradient on the representation space, ensuring
zero feedback/leakage from pixel reconstruction into the learned JEPA representations.
Used exclusively for:
1. Qualitative GIFs of imaginary and planned rollouts
2. Video reconstruction evaluation metrics (PSNR, LPIPS, SSIM)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import TransformerBlock, _init_weights, unpatchify


class VisualDecoder(nn.Module):
    """Decodes representation tokens z [B, N, D] into pixel frames [B, 3, H, W]."""

    def __init__(
        self,
        image_size: int = 64,
        patch: int = 8,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch = patch
        self.num_patches = (image_size // patch) ** 2
        self.pos_space = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch * out_channels)
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, N, D] (or [B, T, N, D]) -> frames: [B, 3, H, W] (or [B, T, 3, H, W])."""
        has_time = z.ndim == 4
        if has_time:
            B, T, N, D = z.shape
            z = z.reshape(B * T, N, D)
        else:
            B, N, D = z.shape
            T = 1

        h = z + self.pos_space
        for blk in self.blocks:
            h = blk(h, causal=False)
        h = self.norm(h)
        pixels = unpatchify(self.head(h).unsqueeze(1), self.patch).squeeze(1)  # [B*T, 3, H, W]

        if has_time:
            pixels = pixels.reshape(B, T, 3, self.image_size, self.image_size)
        return pixels

    def loss(self, pred_pixels: torch.Tensor, target_pixels: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_pixels.float(), target_pixels.float())
