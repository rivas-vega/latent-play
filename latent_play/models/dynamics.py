r"""Action-conditioned JEPA Dynamics Model (V-JEPA 2 / I-JEPA style).

Architecture:
1. Context Encoder:
   ViT-based vision encoder mapping each frame x_t -> spatial tokens z_t [B, N, D]
   or global representation z_t [B, D].
2. Target Encoder:
   Exact structural copy of Context Encoder, parameterized by EMA weights
   (never updated via backprop).
3. Action Conditioner:
   Embeds discrete actions (true actions or LAM discrete codes) or continuous vectors.
4. Causal Predictor:
   Transformer that operates autoregressively on the sequence of representations
   and actions: (z_1, a_1, ..., z_t, a_t) -> predicts \hat{z}_{t+1}.
   Supports multi-step autoregressive rollout in representation space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import Attention, MLP, PatchEmbed, TransformerBlock, _init_weights


@dataclass
class DynamicsConfig:
    image_size: int = 64
    patch: int = 8
    dim: int = 256
    enc_depth: int = 6
    enc_heads: int = 8
    pred_depth: int = 6
    pred_heads: int = 8
    mlp_ratio: float = 4.0
    num_actions: int = 17  # Crafter default (or K=8 for latent actions)
    action_dim: int = 64
    max_context: int = 16  # maximum temporal rollout window
    pool_tokens: bool = False  # if True, pool tokens to [B, D]; else keep [B, N, D]


class VisionEncoder(nn.Module):
    """ViT frame encoder."""

    def __init__(self, image_size: int, patch: int, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.patch = patch
        self.num_patches = (image_size // patch) ** 2
        self.patch_embed = PatchEmbed(image_size, patch, dim, max_frames=1)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] -> tokens: [B, T, N, D]"""
        B, T, C, H, W = frames.shape
        # Flatten temporal dimension for spatial ViT encoding
        x = frames.reshape(B * T, 1, C, H, W)
        tokens = self.patch_embed(x)[:, 0]  # [B*T, N, D]
        for blk in self.blocks:
            tokens = blk(tokens, causal=False)
        tokens = self.norm(tokens)
        return tokens.reshape(B, T, self.num_patches, -1)


class JEPADynamics(nn.Module):
    def __init__(self, cfg: DynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VisionEncoder(
            cfg.image_size, cfg.patch, cfg.dim, cfg.enc_depth, cfg.enc_heads, cfg.mlp_ratio
        )
        self.action_embed = nn.Embedding(cfg.num_actions, cfg.dim)
        self.time_embed = nn.Parameter(torch.zeros(1, cfg.max_context, 1, cfg.dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        # Predictor transformer: operates over (tokens_t + action_t)
        self.pred_blocks = nn.ModuleList(
            [TransformerBlock(cfg.dim, cfg.pred_heads, cfg.mlp_ratio) for _ in range(cfg.pred_depth)]
        )
        self.pred_norm = nn.LayerNorm(cfg.dim)
        self.pred_head = nn.Linear(cfg.dim, cfg.dim)
        self.apply(_init_weights)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encodes frames [B, T, 3, H, W] into representations [B, T, N, D] or [B, T, D]."""
        z = self.encoder(frames)
        if self.cfg.pool_tokens:
            return z.mean(dim=2)  # [B, T, D]
        return z

    def step_predict(self, z_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predicts the next representation z_{t+1} given history z_{1..t} and actions a_{1..t}.

        Args:
            z_seq: [B, T, N, D] (or [B, T, D] if pool_tokens)
            actions: [B, T] integer action ids

        Returns:
            pred_next: [B, N, D] (or [B, D]) predicted representation for step T+1
        """
        B, T = z_seq.shape[:2]
        act_tokens = self.action_embed(actions)  # [B, T, D]

        if self.cfg.pool_tokens:
            # Sequence: [z_1 + a_1, z_2 + a_2, ..., z_T + a_T] -> predict z_{T+1}
            h = z_seq + act_tokens + self.time_embed[:, :T, 0, :]
            for blk in self.pred_blocks:
                h = blk(h, causal=True)
            h = self.pred_norm(h)
            return self.pred_head(h[:, -1])  # [B, D]
        else:
            # Add action embedding and temporal embedding broadcast across all spatial patch tokens
            N = z_seq.shape[2]
            h = z_seq + act_tokens[:, :, None, :] + self.time_embed[:, :T]  # [B, T, N, D]
            # Flatten to [B, T*N, D] with causal spatial-temporal ordering
            flat_h = h.reshape(B, T * N, -1)
            for blk in self.pred_blocks:
                flat_h = blk(flat_h, causal=True)
            flat_h = self.pred_norm(flat_h)
            # Last frame patch tokens
            pred_tokens = self.pred_head(flat_h.reshape(B, T, N, -1)[:, -1])  # [B, N, D]
            return pred_tokens

    def forward(self, z_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        r"""Batch sequence prediction for teacher-forcing training.

        z_seq: [B, T, ...] context representations for t=0..T-1
        actions: [B, T-1] actions taken at t=0..T-2
        Returns predicted representations \hat{z}_{1..T-1}
        """
        B, T = actions.shape[:2]
        preds = []
        for t in range(1, T + 1):
            pred_t = self.step_predict(z_seq[:, :t], actions[:, :t])
            preds.append(pred_t)
        return torch.stack(preds, dim=1)  # [B, T, ...]

    @torch.no_grad()
    def rollout(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
        context_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive multi-step rollout in latent space.

        z_init: [B, T_init, ...] (or [B, ...]) initial context frame representation(s)
        actions: [B, H] sequence of H actions to apply sequentially
        context_actions: [B, T_init - 1] actions corresponding to transitions inside z_init (optional)
        Returns:
            rollout_z: [B, H, ...] predicted latent trajectories
        """
        if z_init.ndim == 3:  # [B, N, D] or [B, D] without temporal dim -> add T=1
            z_init = z_init.unsqueeze(1)
        B, T_init = z_init.shape[:2]
        H = actions.shape[1]

        if context_actions is None:
            curr_actions = torch.zeros((B, max(0, T_init - 1)), dtype=actions.dtype, device=actions.device)
        else:
            curr_actions = context_actions

        curr_z = z_init
        traj = []
        for step in range(H):
            act_step = actions[:, step : step + 1]
            curr_actions = torch.cat([curr_actions, act_step], dim=1)
            z_next = self.step_predict(curr_z, curr_actions)
            traj.append(z_next)
            # Append predicted z to context for subsequent steps
            curr_z = torch.cat([curr_z, z_next.unsqueeze(1)], dim=1)
            # Maintain sliding context window if exceeding max_context
            if curr_z.shape[1] > self.cfg.max_context:
                curr_z = curr_z[:, -self.cfg.max_context :]
                curr_actions = curr_actions[:, -self.cfg.max_context :]
        return torch.stack(traj, dim=1)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
