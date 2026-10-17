"""Vector quantiser with cosine-similarity codes and dead-code restarts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    """VQ-VAE bottleneck (van den Oord et al. 2017) with anti-collapse guarantees:

    * **Batch-centered l2-normalised codes** (cosine lookup, Yu et al. 2022) — prevents constant-bias collapse;
    * **Entropy diversity regularization** (MaskGIT / Genie style) — encourages uniform marginal usage across K codes;
    * **Dead-code restarts** — codes whose running usage drops below ``dead_threshold / K`` are re-seeded.
    """

    def __init__(
        self,
        num_codes: int,
        dim: int,
        beta: float = 0.25,
        l2norm: bool = True,
        dead_threshold: float = 0.25,
        restart_every: int = 50,
        usage_decay: float = 0.95,
        entropy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_codes, self.dim, self.beta, self.l2norm = num_codes, dim, beta, l2norm
        self.dead_threshold, self.restart_every, self.usage_decay = dead_threshold, restart_every, usage_decay
        self.entropy_weight = entropy_weight
        self.codebook = nn.Parameter(torch.randn(num_codes, dim))
        self.register_buffer("usage", torch.full((num_codes,), 1.0 / num_codes))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("running_mean", torch.zeros(1, dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-6) if self.l2norm else x

    def codes(self) -> torch.Tensor:
        return self._norm(self.codebook)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codes()[indices]

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = z.shape
        raw_flat = z.reshape(-1, self.dim).float()
        if self.training and raw_flat.shape[0] > 1:
            batch_mean = raw_flat.mean(dim=0, keepdim=True)
            with torch.no_grad():
                self.running_mean.mul_(0.95).add_(batch_mean.detach(), alpha=0.05)
            centered = raw_flat - batch_mean
        else:
            centered = raw_flat - self.running_mean

        flat = self._norm(centered)
        cb = self.codes().float()
        # Squared Euclidean distance on unit sphere: 2 - 2 * cos(flat, cb)
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb.pow(2).sum(1)[None]
        idx = dist.argmin(1)
        q = cb[idx]
        vq_loss = F.mse_loss(q, flat.detach()) + self.beta * F.mse_loss(flat, q.detach())

        # Soft assignment entropy regularization: maximize marginal entropy H(mean(p)), minimize conditional H(p)
        if self.entropy_weight > 0.0 and raw_flat.shape[0] > 1:
            soft_p = F.softmax(-dist / 0.2, dim=-1)  # [M, K]
            avg_p = soft_p.mean(dim=0)  # [K]
            marginal_ent = -(avg_p * (avg_p + 1e-8).log()).sum()
            cond_ent = -(soft_p * (soft_p + 1e-8).log()).sum(dim=-1).mean()
            # Target: high marginal_ent (up to log(K)) and low cond_ent
            ent_loss = (torch.log(torch.tensor(float(self.num_codes), device=z.device)) - marginal_ent) + 0.2 * cond_ent
            loss = vq_loss + self.entropy_weight * ent_loss
        else:
            loss = vq_loss

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
        noise = 0.1 * torch.randn(n_dead, self.dim, device=flat.device)
        self.codebook.data[dead] = self._norm(flat[pick] + noise).to(self.codebook.dtype)
        self.usage[dead] = 1.0 / self.num_codes
