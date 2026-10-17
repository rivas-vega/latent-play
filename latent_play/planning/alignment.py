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
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.num_actions = num_actions
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

    def forward(self, codes: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """codes: [B, ...] int64 code indices -> logits: [B, ..., num_actions]."""
        if self.use_state_context:
            assert z is not None
            one_hot_codes = F.one_hot(codes, self.num_codes).float()
            inp = torch.cat([one_hot_codes, z], dim=-1)
            return self.net(inp)
        return self.classifier(codes)

    def predict_action(self, codes: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Returns argmax predicted real action index."""
        logits = self.forward(codes, z)
        return logits.argmax(dim=-1)

    def loss(self, logits: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Computes cross-entropy loss only on labelled transitions (where target_actions >= 0)."""
        valid = target_actions >= 0
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(logits[valid], target_actions[valid])
