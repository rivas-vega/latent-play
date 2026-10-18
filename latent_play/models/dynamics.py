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

from .layers import Attention, MLP, PatchEmbed, STBlock, TransformerBlock, _init_weights


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
    action_modulation: str = "additive"  # "additive" or "film"


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
        # Extra index (cfg.num_actions) reserved for unlabelled transitions (action == -1)
        self.action_embed = nn.Embedding(cfg.num_actions + 1, cfg.dim)
        self.time_embed = nn.Parameter(torch.zeros(1, cfg.max_context, 1, cfg.dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        if cfg.action_modulation == "film":
            self.action_film = nn.Sequential(
                nn.Linear(cfg.dim, cfg.dim),
                nn.SiLU(),
                nn.Linear(cfg.dim, cfg.dim * 2),
            )
            nn.init.zeros_(self.action_film[-1].weight)
            nn.init.zeros_(self.action_film[-1].bias)

        # Predictor transformer: STBlock (spatial bidirectional + temporal causal) when keeping patch tokens
        if cfg.pool_tokens:
            self.pred_blocks = nn.ModuleList(
                [TransformerBlock(cfg.dim, cfg.pred_heads, cfg.mlp_ratio) for _ in range(cfg.pred_depth)]
            )
        else:
            self.pred_blocks = nn.ModuleList(
                [STBlock(cfg.dim, cfg.pred_heads, cfg.mlp_ratio) for _ in range(cfg.pred_depth)]
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

    def forward(self, z_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        r"""Parallel causal sequence prediction for teacher-forcing training.

        Args:
            z_seq: [B, T, N, D] (or [B, T, D]) context representations for t=0..T-1
            actions: [B, T] integer action ids in {0..num_actions-1} (or -1 for unlabelled)

        Returns:
            pred_seq: [B, T, N, D] (or [B, T, D]) predicted next-step representations \hat{z}_{1..T}
        """
        B, T = actions.shape[:2]
        z_ctx = z_seq[:, :T]
        safe_actions = torch.where(actions < 0, self.cfg.num_actions, actions)
        act_tokens = self.action_embed(safe_actions)  # [B, T, D]
        t_emb = self.time_embed[:, :T]  # [1, T, 1, D]

        if self.cfg.action_modulation == "film":
            cond = act_tokens + (t_emb[:, :, 0, :] if t_emb.ndim == 4 else t_emb)
            gamma, beta = self.action_film(cond).chunk(2, dim=-1)
            if self.cfg.pool_tokens:
                h = z_ctx * (1.0 + gamma) + beta
            else:
                h = z_ctx * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]
        else:
            if self.cfg.pool_tokens:
                h = z_ctx + act_tokens + t_emb[:, :, 0, :]
            else:
                h = z_ctx + act_tokens[:, :, None, :] + t_emb  # [B, T, N, D]

        for blk in self.pred_blocks:
            h = blk(h, causal=True) if self.cfg.pool_tokens else blk(h)
        h = self.pred_norm(h)
        return z_ctx + self.pred_head(h)

    def step_predict(self, z_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predicts the next representation z_{t+1} given history z_{1..t} and actions a_{1..t}.

        Args:
            z_seq: [B, T, N, D] (or [B, T, D] if pool_tokens)
            actions: [B, T] integer action ids

        Returns:
            pred_next: [B, N, D] (or [B, D]) predicted representation for step T+1
        """
        return self.forward(z_seq, actions)[:, -1]

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
        if self.cfg.pool_tokens:
            if z_init.ndim == 2:  # [B, D]
                z_init = z_init.unsqueeze(1)
        else:
            if z_init.ndim == 3:  # [B, N, D]
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
