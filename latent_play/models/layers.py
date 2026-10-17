"""Building blocks: patch embedding and a spatio-temporal (ST) transformer.

The ST block (Genie, Bruce et al. 2024) factorises attention into
*spatial* attention within each frame and *causal temporal* attention across
frames for each patch location, which keeps memory linear in ``T``.
Tensors are laid out as ``[B, T, N, D]`` (batch, time, patches, channels).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, T, C, H, W] -> [B, T, N, patch*patch*C]."""
    B, T, C, H, W = x.shape
    x = x.reshape(B, T, C, H // patch, patch, W // patch, patch)
    return x.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, T, (H // patch) * (W // patch), patch * patch * C)


def unpatchify(x: torch.Tensor, patch: int, channels: int = 3) -> torch.Tensor:
    """[B, T, N, patch*patch*C] -> [B, T, C, H, W] (square images)."""
    B, T, N, _ = x.shape
    g = int(N**0.5)
    x = x.reshape(B, T, g, g, patch, patch, channels)
    return x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, channels, g * patch, g * patch)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch: int, dim: int, max_frames: int, channels: int = 3) -> None:
        super().__init__()
        self.patch = patch
        self.num_patches = (image_size // patch) ** 2
        self.proj = nn.Linear(patch * patch * channels, dim)
        self.pos_space = nn.Parameter(torch.zeros(1, 1, self.num_patches, dim))
        self.pos_time = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        nn.init.trunc_normal_(self.pos_time, std=0.02)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[1]
        return self.proj(patchify(frames, self.patch)) + self.pos_space + self.pos_time[:, :T]


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        L, D = x.shape[-2:]
        q, k, v = self.qkv(x).reshape(-1, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)  # FlashAttention when available
        return self.out(y.transpose(1, 2).reshape(*x.shape))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, int(dim * ratio)), nn.GELU(), nn.Linear(int(dim * ratio), dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with SDPA FlashAttention."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.n1(x), causal=causal)
        return x + self.mlp(self.n2(x))


class STBlock(nn.Module):
    """Pre-norm block: spatial attn -> causal temporal attn -> MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.spatial = Attention(dim, heads)
        self.temporal = Attention(dim, heads)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x = x + self.spatial(self.n1(x).reshape(B * T, N, D)).reshape(B, T, N, D)
        xt = self.n2(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = x + self.temporal(xt, causal=True).reshape(B, N, T, D).permute(0, 2, 1, 3)
        return x + self.mlp(self.n3(x))


class STTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([STBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
