"""Latent Action Model (LAM): infer discrete actions from action-free video.

Encoder (inverse dynamics, IDM)
    frames ``x_0..x_H`` -> ST-transformer with causal time -> for each transition
    ``t -> t+1`` pool the tokens of frame ``t+1`` (which, by causality, see only
    ``x_<=t+1``) -> project to ``code_dim`` -> VQ with ``K`` codes.

Decoder (forward dynamics, FDM)
    past frames ``x_0..x_{H-1}`` + quantised actions -> ST-transformer with causal
    time -> predict ``x_{t+1}`` from frame-``t`` tokens (residual over ``x_t``).

The tiny bottleneck (``K`` codes, ~3 bits) forces the code to carry *what changed
because of the agent* rather than the whole next frame (Genie; LAPO).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import PatchEmbed, STTransformer, unpatchify
from .vq import VectorQuantizer


@dataclass
class LAMConfig:
    image_size: int = 64
    patch: int = 8
    context: int = 2  # H past frames -> windows of H+1 frames, H transitions
    dim: int = 256
    enc_depth: int = 4
    dec_depth: int = 4
    heads: int = 8
    num_codes: int = 8
    code_dim: int = 32
    beta: float = 0.25
    vq_weight: float = 1.0

    @property
    def num_frames(self) -> int:
        return self.context + 1


class LatentActionModel(nn.Module):
    def __init__(self, cfg: LAMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        T = cfg.num_frames
        self.enc_embed = PatchEmbed(cfg.image_size, cfg.patch, cfg.dim, T)
        self.encoder = STTransformer(cfg.dim, cfg.enc_depth, cfg.heads)
        self.to_action = nn.Linear(cfg.dim, cfg.code_dim)
        self.vq = VectorQuantizer(cfg.num_codes, cfg.code_dim, beta=cfg.beta)

        self.dec_embed = PatchEmbed(cfg.image_size, cfg.patch, cfg.dim, T - 1)
        self.action_proj = nn.Linear(cfg.code_dim, cfg.dim)
        self.decoder = STTransformer(cfg.dim, cfg.dec_depth, cfg.heads)
        self.head = nn.Linear(cfg.dim, cfg.patch * cfg.patch * 3)
        nn.init.zeros_(self.head.weight)  # start as "copy the last frame"
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------ inference
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B, T, 3, H, W] in [-1, 1] -> continuous actions [B, T-1, code_dim]."""
        B, T, C, H, W = frames.shape
        max_f = self.cfg.num_frames
        if T <= max_f:
            h = self.encoder(self.enc_embed(frames))
            return self.to_action(h[:, 1:].mean(dim=2))

        # For longer sequences (e.g. T=8 or 16 in dynamics training), slide a window of length max_f
        pad_len = max_f - 2
        if pad_len > 0:
            padded = torch.cat([frames[:, :1].expand(B, pad_len, C, H, W), frames], dim=1)
        else:
            padded = frames
        # Build (T - 1) windows of length max_f: [B, T-1, max_f, C, H, W]
        windows = torch.stack([padded[:, i : i + max_f] for i in range(T - 1)], dim=1)
        flat_win = windows.reshape(B * (T - 1), max_f, C, H, W)
        h = self.encoder(self.enc_embed(flat_win))  # [B*(T-1), max_f, N, D]
        out = self.to_action(h[:, -1].mean(dim=1))  # [B*(T-1), code_dim]
        return out.reshape(B, T - 1, -1)

    def infer_actions(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.vq(self.encode(frames))

    def decode(self, past: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """past [B, T', 3, H, W], actions [B, T', code_dim] -> next frames [B, T', 3, H, W]."""
        h = self.dec_embed(past) + self.action_proj(actions)[:, :, None, :]
        delta = unpatchify(self.head(self.decoder(h)), self.cfg.patch)
        return past + delta

    def decode_indices(self, past: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.decode(past, self.vq.lookup(indices).to(past.dtype))

    # ------------------------------------------------------------- training
    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        vq = self.infer_actions(frames)
        pred = self.decode(frames[:, :-1], vq["z_q"])
        target = frames[:, 1:]
        recon = F.mse_loss(pred.float(), target.float())
        return {
            "loss": recon + self.cfg.vq_weight * vq["loss"],
            "recon": recon,
            "vq_loss": vq["loss"],
            "perplexity": vq["perplexity"],
            "indices": vq["indices"],
            "pred": pred,
        }

    def config_dict(self) -> dict:
        return asdict(self.cfg)
