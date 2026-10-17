r"""Violation-of-Expectation (VoE) Physics Scorer adapter for latent world models.

Directly bridges `latent-play` with your `vjepa2-physics-scorer-poc` repository:
Evaluates whether the dynamics model has internalized physical common sense and
conservation laws:
1. Presents the model with physically plausible trajectories vs. expectation-violating trajectories:
   - Teleportation (non-inertial position jump)
   - Object/agent disappearance
   - Action contradiction (agent moves opposite to commanded velocity)
2. Measures Latent Prediction Error (Surprise):
   Surprise(t) = || \hat{z}_{t+1} - z_{t+1}^{observed} ||_1
3. Computes the VoE Discrimination Ratio:
   Ratio = Surprise(Violation) / Surprise(Plausible)
   An expressive physical world model exhibits Ratio >> 1.0.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from latent_play.models import JEPADynamics


def inject_teleport_violation(frames: torch.Tensor, roll_pixels: int = 24) -> torch.Tensor:
    """Injects an abrupt non-inertial teleport into the second half of a video clip."""
    violated = frames.clone()
    T = frames.shape[1]
    mid = T // 2
    # Spatial roll abruptly teleports all entities
    violated[:, mid:] = torch.roll(violated[:, mid:], shifts=roll_pixels, dims=-1)
    return violated


def inject_disappearance_violation(frames: torch.Tensor) -> torch.Tensor:
    """Simulates sudden entity disappearance by zeroing out foreground contrast."""
    violated = frames.clone()
    T = frames.shape[1]
    mid = T // 2
    # Abrupt blanking/dropout violating object permanence
    violated[:, mid:] = violated[:, mid:].mean(dim=(-2, -1), keepdim=True)
    return violated


def inject_action_contradiction_violation(frames: torch.Tensor) -> torch.Tensor:
    """Simulates an action-contradicting transition by reversing temporal motion in the second half."""
    violated = frames.clone()
    T = frames.shape[1]
    mid = T // 2
    violated[:, mid:] = torch.flip(violated[:, : T - mid], dims=[1])
    return violated


@torch.no_grad()
def compute_voe_score(
    model: JEPADynamics,
    frames: torch.Tensor,
    actions: torch.Tensor,
    violation_type: str = "teleport",
) -> dict[str, float]:
    """Computes VoE surprise metrics for plausible vs. violated video clips at the violation onset.

    Args:
        model: JEPADynamics model
        frames: [B, T, 3, H, W] clean video frames in [-1, 1]
        actions: [B, T-1] action sequence
        violation_type: "teleport", "disappearance", or "contradiction"

    Returns:
        dict with plausible surprise, violated surprise, and VoE ratio
    """
    model.eval()
    B, T = frames.shape[:2]
    mid = max(1, T // 2)
    onset = min(mid - 1, actions.shape[1] - 1)

    # Clean plausible trajectory
    z_clean = model.encode(frames)
    pred_clean = model(z_clean[:, :-1], actions)
    target_clean = z_clean[:, 1:]
    surprise_plausible = F.l1_loss(pred_clean[:, onset], target_clean[:, onset]).item()

    # Violated trajectory
    if violation_type == "teleport":
        v_frames = inject_teleport_violation(frames)
    elif violation_type == "disappearance":
        v_frames = inject_disappearance_violation(frames)
    elif violation_type == "contradiction":
        v_frames = inject_action_contradiction_violation(frames)
    else:
        raise ValueError(f"Unknown violation type: {violation_type}")

    z_violated = model.encode(v_frames)
    pred_violated = model(z_violated[:, :-1], actions)
    target_violated = z_violated[:, 1:]
    surprise_violated = F.l1_loss(pred_violated[:, onset], target_violated[:, onset]).item()

    ratio = surprise_violated / max(1e-6, surprise_plausible)

    return {
        "surprise_plausible": surprise_plausible,
        "surprise_violated": surprise_violated,
        "voe_ratio": ratio,
        "violation_type": violation_type,
    }
