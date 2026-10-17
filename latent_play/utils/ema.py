"""Exponential moving average target network (JEPA / BYOL style)."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


class EMA:
    """Keeps ``target = tau * target + (1 - tau) * online`` with a cosine tau schedule.

    ``tau`` goes from ``tau_start`` to ``tau_end`` over ``total_steps`` (as in
    I-JEPA / V-JEPA). The target never receives gradients.
    """

    def __init__(self, model: nn.Module, total_steps: int, tau_start: float = 0.996, tau_end: float = 1.0) -> None:
        self.target = copy.deepcopy(model).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.total_steps = max(1, total_steps)
        self.tau_start, self.tau_end = tau_start, tau_end

    def tau(self, step: int) -> float:
        progress = min(1.0, step / self.total_steps)
        return self.tau_end - (self.tau_end - self.tau_start) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> float:
        tau = self.tau(step)
        for pt, po in zip(self.target.parameters(), model.parameters()):
            pt.lerp_(po.detach(), 1.0 - tau)
        for bt, bo in zip(self.target.buffers(), model.buffers()):
            bt.copy_(bo)
        return tau

    def state_dict(self) -> dict:
        return self.target.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.target.load_state_dict(state)
