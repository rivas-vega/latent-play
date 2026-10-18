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
    vq_weight: float = 0.05
    motion_focus: float = 15.0  # upweights pixels that changed between t and t+1
    state_context: bool = False  # condition latent action VQ on initial frame affordances

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
        self.spatial_attn = nn.Linear(cfg.dim, 1)

        in_dim = cfg.dim * 2 if cfg.state_context else cfg.dim
        if cfg.state_context:
            num_patches = (cfg.image_size // cfg.patch) ** 2
            self.flow_conv = nn.Sequential(
                nn.Conv2d(cfg.dim * 2, cfg.dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.flow_pos = nn.Parameter(torch.zeros(1, 1, num_patches, cfg.dim))
            nn.init.trunc_normal_(self.flow_pos, std=0.02)
        self.action_ln = nn.LayerNorm(in_dim)
        self.to_action = nn.Linear(in_dim, cfg.code_dim)
        self.vq = VectorQuantizer(cfg.num_codes, cfg.code_dim, beta=cfg.beta)

        self.dec_embed = PatchEmbed(cfg.image_size, cfg.patch, cfg.dim, T - 1)
        self.action_proj = nn.Linear(cfg.code_dim, cfg.dim)
        self.decoder = STTransformer(cfg.dim, cfg.dec_depth, cfg.heads)
        self.head = nn.Linear(cfg.dim, cfg.patch * cfg.patch * 3)
        nn.init.zeros_(self.head.weight)  # start as "copy the last frame"
        nn.init.zeros_(self.head.bias)

    def _pool_transitions(self, h: torch.Tensor) -> torch.Tensor:
        """Pools temporal feature differences dh = h[:, 1:] - h[:, :-1] across spatial patches via attention."""
        dh = h[:, 1:] - h[:, :-1]  # [B, T-1, N, D] — cancels static episode background
        if self.cfg.state_context:
            B, T_m1, N, D = dh.shape
            g = int(N**0.5)
            # Combine initial state h_t and temporal difference dh over 2D patch neighborhood (3x3)
            # so each patch sees directional displacements (up/down/left/right) and adjacent obstacles
            cat_2d = torch.cat([h[:, :-1], dh], dim=-1).permute(0, 1, 3, 2).reshape(B * T_m1, D * 2, g, g)
            flow_feat = self.flow_conv(cat_2d).reshape(B, T_m1, D, N).permute(0, 1, 3, 2) + self.flow_pos
            weights = F.softmax(self.spatial_attn(flow_feat), dim=2)  # [B, T-1, N, 1]
            pooled_dh = (dh * weights).sum(dim=2)
            pooled_flow = (flow_feat * weights).sum(dim=2)
            pooled = torch.cat([pooled_dh, pooled_flow], dim=-1)
        else:
            weights = F.softmax(self.spatial_attn(dh), dim=2)  # [B, T-1, N, 1]
            pooled = (dh * weights).sum(dim=2)  # [B, T-1, D]
        return self.to_action(self.action_ln(pooled))

    # ------------------------------------------------------------ inference
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B, T, 3, H, W] in [-1, 1] -> continuous actions [B, T-1, code_dim]."""
        B, T, C, H, W = frames.shape
        max_f = self.cfg.num_frames
        if T <= max_f:
            h = self.encoder(self.enc_embed(frames))
            return self._pool_transitions(h)

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
        out = self._pool_transitions(h)[:, -1]  # [B*(T-1), code_dim]
        return out.reshape(B, T - 1, -1)

    def infer_actions(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.vq(self.encode(frames))

    def decode(self, past: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """past [B, T', 3, H, W], actions [B, T', code_dim] -> next frames [B, T', 3, H, W]."""
        B, T_p, C, H, W = past.shape
        # Decode each transition Markovianly (x_t + a_t -> x_{t+1}) without leaking velocity from x_{t-1}
        past_flat = past.reshape(B * T_p, 1, C, H, W)
        act_flat = self.action_proj(actions).reshape(B * T_p, 1, 1, -1)
        h = self.dec_embed(past_flat) + act_flat
        delta = unpatchify(self.head(self.decoder(h)), self.cfg.patch).reshape(B, T_p, C, H, W)
        return past + delta

    def decode_indices(self, past: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.decode(past, self.vq.lookup(indices).to(past.dtype))

    # ------------------------------------------------------------- training
    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        vq = self.infer_actions(frames)
        past = frames[:, :-1]
        target = frames[:, 1:]
        pred = self.decode(past, vq["z_q"])
        # Motion-focused MSE: upweight pixels that actually changed between past and target
        sq_err = (pred.float() - target.float()).pow(2)
        if self.cfg.motion_focus > 0.0:
            motion_weight = 1.0 + self.cfg.motion_focus * (target.float() - past.float()).abs().detach()
            recon = (sq_err * motion_weight).mean()
        else:
            recon = sq_err.mean()
        return {
            "loss": recon + self.cfg.vq_weight * vq["loss"],
            "recon": sq_err.mean(),
            "vq_loss": vq["loss"],
            "perplexity": vq["perplexity"],
            "indices": vq["indices"],
            "pred": pred,
        }

    def config_dict(self) -> dict:
        return asdict(self.cfg)
