"""Alignment module: maps discrete latent actions to real environment actions.

Trained strictly on the sparse labelled subset (controlled by ``label_fraction``:
0.001, 0.01, 0.1, 1.0).

Key properties:
- Linear probe or light MLP classifier: P(real_action | latent_code, context)
- Class-balanced loss to handle infrequent game actions
- Used by the MPC planner: plan in latent action space -> map best plan to real actions
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ActionAlignment(nn.Module):
    """Maps latent code indices k in {0..K-1} (optionally with state representation z) to real actions."""

    def __init__(
        self,
        num_codes: int,
        num_actions: int,
        dim: int = 128,
        use_state_context: bool = False,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim is not None:
            dim = embed_dim
        self.num_codes = num_codes
        self.num_actions = num_actions
        self.dim = dim
        self.use_state_context = use_state_context

        if use_state_context:
            self.net = nn.Sequential(
                nn.Linear(num_codes + dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        else:
            # Direct matrix lookup: K codes -> logits over real actions
            self.classifier = nn.Embedding(num_codes, num_actions)
            nn.init.zeros_(self.classifier.weight)

        # Inverse lookup: real action a in {0..A-1} -> logits over K latent codes
        self.inv_classifier = nn.Embedding(num_actions, num_codes)
        nn.init.zeros_(self.inv_classifier.weight)
        self.register_buffer("action_prior", torch.ones(num_actions) / max(1, num_actions))

    def forward(self, codes: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """codes: [B, ...] int64 code indices -> logits: [B, ..., num_actions]."""
        if self.use_state_context:
            assert z is not None, "z representation context must be provided when use_state_context=True"
            # Pool spatial patch dimensions so that z matches codes shape up to feature dim
            if z.shape[: codes.ndim] == codes.shape:
                while z.ndim > codes.ndim + 1:
                    z = z.mean(dim=-2)
            else:
                z = z.reshape(*codes.shape, -1, z.shape[-1]).mean(dim=-2)
            one_hot_codes = F.one_hot(codes, self.num_codes).float()
            inp = torch.cat([one_hot_codes, z], dim=-1)
            return self.net(inp)
        return self.classifier(codes)

    def forward_inverse(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [B, ...] int64 real action indices -> logits: [B, ..., num_codes]."""
        return self.inv_classifier(actions.clamp_min(0))

    def predict_action(
        self,
        codes: torch.Tensor,
        z: torch.Tensor | None = None,
        prior_correction: float = 0.0,
    ) -> torch.Tensor:
        """Returns argmax predicted real action index."""
        logits = self.forward(codes, z)
        if prior_correction > 0.0:
            logits = logits - prior_correction * torch.log(self.action_prior.clamp_min(1e-6))
        return logits.argmax(dim=-1)

    def predict_code(self, actions: torch.Tensor) -> torch.Tensor:
        """Returns argmax predicted latent code index for a given real action."""
        return self.forward_inverse(actions).argmax(dim=-1)

    def loss(
        self,
        logits: torch.Tensor,
        target_actions: torch.Tensor,
        codes: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes cross-entropy loss only on labelled transitions (where target_actions >= 0)."""
        valid = target_actions >= 0
        if not valid.any():
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits[valid], target_actions[valid], weight=class_weights)
        if codes is not None:
            inv_logits = self.forward_inverse(target_actions[valid])
            ce = ce + 0.1 * F.cross_entropy(inv_logits, codes[valid])
        return ce
