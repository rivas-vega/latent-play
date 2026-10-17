"""Anti-collapse losses and JEPA training objectives.

Connects directly to the mathematical foundations of representation collapse
and non-contrastive self-supervised learning:
- Pure EMA (target network stop-gradient, Grill et al. 2020 / Assran et al. 2023)
- VICReg variance & covariance regularization (Bardes et al. 2022)
- Statistical Invariant Geometric Regularizer (SIGReg) via Cramér-Wold & Epps-Pulley test (Balestriero, LeCun et al. 2025/2026)
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


def sigreg_loss(z: torch.Tensor, num_slices: int = 64) -> torch.Tensor:
    r"""Statistical Invariant Geometric Regularizer (SIGReg).

    Based on the Cramér-Wold theorem and the differentiable Epps-Pulley test statistic (1983):
        T_{EP}(h) = 1 + \frac{\sqrt{2}}{n^2} \sum_{j,k} e^{-(h_j - h_k)^2 / 4} - \frac{4}{\sqrt{3} n} \sum_j e^{-h_j^2 / 6}

    Projecting representations z against random unit vectors u ~ S^{D-1} and testing
    each 1D projection against N(0, 1) guarantees by Cramér-Wold that the full
    multivariate representation converges to an isotropic standard Gaussian N(0, I_D).
    """
    N, D = z.shape
    if N < 2:
        return torch.tensor(0.0, device=z.device)

    u = torch.randn(D, num_slices, device=z.device, dtype=z.dtype)
    u = F.normalize(u, dim=0)  # Random unit vectors on S^{D-1}
    h = z @ u  # [N, num_slices]

    diff = h.unsqueeze(1) - h.unsqueeze(0)  # [N, N, num_slices]
    sqrt2 = 1.4142135623730951
    sqrt3 = 1.7320508075688772

    t2 = (sqrt2 / (N * N)) * torch.exp(-diff.pow(2) / 4.0).sum(dim=(0, 1))
    t3 = (4.0 / (sqrt3 * N)) * torch.exp(-h.pow(2) / 6.0).sum(dim=0)
    tep = 1.0 + t2 - t3  # [num_slices]
    return tep.mean()


def jepa_loss(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    loss_type: str = "l1",
    vicreg_var_weight: float = 0.0,
    vicreg_cov_weight: float = 0.0,
    sigreg_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JEPA prediction loss against target encoder representation (stop-gradient on target).

    Args:
        pred_z: [B, ..., D] predicted representations
        target_z: [B, ..., D] target representations (already detached)
        loss_type: "l1", "smooth_l1", or "cosine"
        vicreg_var_weight: weight for variance anti-collapse loss on online representations
        vicreg_cov_weight: weight for covariance decorrelation loss
        sigreg_weight: weight for SIGReg Cramér-Wold isotropic Gaussian regularization

    Returns:
        dict with total loss, pred_loss, var_loss, cov_loss, sigreg_loss, eff_rank
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
    sig_err = sigreg_loss(flat_pred) if sigreg_weight > 0 else torch.tensor(0.0, device=pred_z.device)

    total = pred_err + vicreg_var_weight * var_err + vicreg_cov_weight * cov_err + sigreg_weight * sig_err
    erank = effective_rank(flat_pred)

    return {
        "loss": total,
        "pred_loss": pred_err,
        "var_loss": var_err,
        "cov_loss": cov_err,
        "sigreg_loss": sig_err,
        "eff_rank": erank,
    }
