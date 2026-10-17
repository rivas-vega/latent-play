"""Vector quantiser with cosine-similarity codes and dead-code restarts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    """VQ-VAE bottleneck (van den Oord et al. 2017) with two standard fixes:

    * **l2-normalised codes** (cosine lookup, Yu et al. 2022) — better codebook usage;
    * **dead-code restarts** — codes whose running usage drops below
      ``dead_threshold / K`` are re-initialised to random encoder outputs.
    """

    def __init__(
        self,
        num_codes: int,
        dim: int,
        beta: float = 0.25,
        l2norm: bool = True,
        dead_threshold: float = 0.1,
        restart_every: int = 200,
        usage_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_codes, self.dim, self.beta, self.l2norm = num_codes, dim, beta, l2norm
        self.dead_threshold, self.restart_every, self.usage_decay = dead_threshold, restart_every, usage_decay
        self.codebook = nn.Parameter(torch.randn(num_codes, dim) * 0.1)
        self.register_buffer("usage", torch.full((num_codes,), 1.0 / num_codes))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) if self.l2norm else x

    def codes(self) -> torch.Tensor:
        return self._norm(self.codebook)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codes()[indices]

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = z.shape
        flat = self._norm(z.reshape(-1, self.dim).float())
        cb = self.codes().float()
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb.pow(2).sum(1)[None]
        idx = dist.argmin(1)
        q = cb[idx]
        loss = F.mse_loss(q, flat.detach()) + self.beta * F.mse_loss(flat, q.detach())
        q_st = flat + (q - flat).detach()  # straight-through estimator

        counts = torch.bincount(idx, minlength=self.num_codes).float()
        probs = counts / counts.sum().clamp_min(1)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        if self.training:
            self._update_usage(probs, flat.detach())
        return {
            "z_q": q_st.reshape(shape).to(z.dtype),
            "indices": idx.reshape(shape[:-1]),
            "loss": loss,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def _update_usage(self, probs: torch.Tensor, flat: torch.Tensor) -> None:
        self.usage.mul_(self.usage_decay).add_(probs, alpha=1 - self.usage_decay)
        self.steps += 1
        if self.restart_every <= 0 or int(self.steps) % self.restart_every != 0:
            return
        dead = self.usage < self.dead_threshold / self.num_codes
        n_dead = int(dead.sum())
        if n_dead == 0 or flat.shape[0] == 0:
            return
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        noise = 0.01 * torch.randn(n_dead, self.dim, device=flat.device)
        self.codebook.data[dead] = (flat[pick] + noise).to(self.codebook.dtype)
        self.usage[dead] = 1.0 / self.num_codes
