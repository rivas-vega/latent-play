"""Anti-collapse losses and JEPA training objectives.

Connects directly to the mathematical foundations of representation collapse
and non-contrastive self-supervised learning:
- Pure EMA (target network stop-gradient, Grill et al. 2020 / Assran et al. 2023)
- VICReg variance regularization (Bardes et al. 2022) to enforce dimension spread
- Effective rank diagnostic (Roy & Vetterli 2007) to quantify representation collapse
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_rank(z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes the effective rank of representations z: exp(entropy of singular value spectrum).

    z: [N, D] tensor of representations.
    Returns scalar in [1, D]. An effective rank close to 1 indicates dimensional collapse.
    """
    if z.shape[0] < 2:
        return torch.tensor(1.0, device=z.device)
    z_centered = z - z.mean(dim=0, keepdim=True)
    # Fast SVD or covariance eigendecomposition
    try:
        _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
        p = s / (s.sum() + eps)
        entropy = -(p * torch.log(p + eps)).sum()
        return torch.exp(entropy)
    except Exception:
        return torch.tensor(1.0, device=z.device)


def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """VICReg variance penalty: max(0, gamma - std(z, dim=0)).

    Forces each representation dimension to have standard deviation >= gamma.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance decorrelation penalty: off-diagonal elements of cov(z) -> 0."""
    N, D = z.shape
    if N < 2:
        return torch.tensor(0.0, device=z.device)
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.t() @ z_centered) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / D


def jepa_loss(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    loss_type: str = "l1",
    vicreg_var_weight: float = 0.0,
    vicreg_cov_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JEPA prediction loss against target encoder representation (stop-gradient on target).

    Args:
        pred_z: [B, ..., D] predicted representations
        target_z: [B, ..., D] target representations (already detached)
        loss_type: "l1", "smooth_l1", or "cosine"
        vicreg_var_weight: weight for variance anti-collapse loss on online representations
        vicreg_cov_weight: weight for covariance decorrelation loss

    Returns:
        dict with total loss, pred_loss, var_loss, cov_loss, eff_rank
    """
    flat_pred = pred_z.reshape(-1, pred_z.shape[-1])
    flat_target = target_z.reshape(-1, target_z.shape[-1]).detach()

    if loss_type == "l1":
        pred_err = F.l1_loss(flat_pred, flat_target)
    elif loss_type == "smooth_l1":
        pred_err = F.smooth_l1_loss(flat_pred, flat_target, beta=0.05)
    elif loss_type == "cosine":
        pred_err = 1.0 - F.cosine_similarity(flat_pred, flat_target, dim=-1).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    var_err = vicreg_variance_loss(flat_pred) if vicreg_var_weight > 0 else torch.tensor(0.0, device=pred_z.device)
    cov_err = vicreg_covariance_loss(flat_pred) if vicreg_cov_weight > 0 else torch.tensor(0.0, device=pred_z.device)

    total = pred_err + vicreg_var_weight * var_err + vicreg_cov_weight * cov_err
    erank = effective_rank(flat_pred)

    return {
        "loss": total,
        "pred_loss": pred_err,
        "var_loss": var_err,
        "cov_loss": cov_err,
        "eff_rank": erank,
    }
