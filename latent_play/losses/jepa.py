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


def _subsample_rows(z: torch.Tensor, max_samples: int = 512) -> torch.Tensor:
    """Flattens z to [M, D] and randomly subsamples at most max_samples rows if M > max_samples."""
    flat = z.reshape(-1, z.shape[-1])
    if flat.shape[0] > max_samples:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:max_samples]
        flat = flat[idx]
    return flat


def effective_rank(z: torch.Tensor, eps: float = 1e-7, max_samples: int = 1024) -> torch.Tensor:
    """Computes the effective rank of representations z: exp(entropy of singular value spectrum).

    z: [..., D] tensor of representations.
    Returns scalar in [1, D]. An effective rank close to 1 indicates dimensional collapse.
    """
    flat = _subsample_rows(z.detach().float(), max_samples=max_samples)
    if flat.shape[0] < 2:
        return torch.tensor(1.0, device=z.device)
    z_centered = flat - flat.mean(dim=0, keepdim=True)
    try:
        _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
        p = s / (s.sum() + eps)
        entropy = -(p * torch.log(p + eps)).sum()
        return torch.exp(entropy)
    except Exception:
        return torch.tensor(1.0, device=z.device)


def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """VICReg variance penalty: max(0, gamma - std(z, dim=0)).

    Forces each representation dimension to have standard deviation >= gamma across the batch/tokens.
    """
    flat = z.reshape(-1, z.shape[-1]).float()
    if flat.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    std = torch.sqrt(flat.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance decorrelation penalty: off-diagonal elements of cov(z) -> 0."""
    flat = z.reshape(-1, z.shape[-1]).float()
    N, D = flat.shape
    if N < 2:
        return torch.tensor(0.0, device=z.device)
    z_centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (z_centered.t() @ z_centered) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / D


def sigreg_loss(z: torch.Tensor, num_slices: int = 64, max_samples: int = 512) -> torch.Tensor:
    r"""Statistical Invariant Geometric Regularizer (SIGReg).

    Based on the Cramér-Wold theorem and the differentiable Epps-Pulley test statistic (1983):
        T_{EP}(h) = 1 + \frac{\sqrt{2}}{n^2} \sum_{j,k} e^{-(h_j - h_k)^2 / 4} - \frac{4}{\sqrt{3} n} \sum_j e^{-h_j^2 / 6}

    Projecting representations z against random unit vectors u ~ S^{D-1} and testing
    each 1D projection against N(0, 1) guarantees by Cramér-Wold that the full
    multivariate representation converges to an isotropic standard Gaussian N(0, I_D).

    Subsamples at most ``max_samples`` tokens per call to keep the O(n^2) pairwise
    Gaussian kernel fast and memory-bounded (~64 MB instead of O((BTN)^2)).
    """
    flat = _subsample_rows(z.float(), max_samples=max_samples)
    N, D = flat.shape
    if N < 2:
        return torch.tensor(0.0, device=z.device)

    u = torch.randn(D, num_slices, device=flat.device, dtype=flat.dtype)
    u = F.normalize(u, dim=0)  # Random unit vectors on S^{D-1}
    h = flat @ u  # [N, num_slices]

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
    online_z: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    loss_type: str = "l1",
    vicreg_var_weight: float = 0.0,
    vicreg_cov_weight: float = 0.0,
    sigreg_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JEPA prediction loss against EMA target encoder + anti-collapse regularization on online_z.

    Args:
        pred_z: [B, T, ..., D] predicted next-frame representations
        target_z: [B, T, ..., D] target representations from EMA encoder (detached)
        online_z: [B, T, ..., D] online encoder representations to regularize against collapse.
                  If None, falls back to regularizing pred_z.
        valid_mask: optional boolean tensor [B, T] masking which transitions have valid actions
                    (used when training supervised baselines with partial action labels).
        loss_type: "l1", "smooth_l1", or "cosine"
        vicreg_var_weight: weight for VICReg variance hinge loss
        vicreg_cov_weight: weight for VICReg covariance decorrelation loss
        sigreg_weight: weight for SIGReg Cramér-Wold isotropic Gaussian regularization

    Returns:
        dict with total loss, pred_loss, var_loss, cov_loss, sigreg_loss, eff_rank
    """
    target_z = target_z.detach()
    reg_z = online_z if online_z is not None else pred_z

    if valid_mask is not None:
        if valid_mask.any():
            pred_valid = pred_z[valid_mask]
            target_valid = target_z[valid_mask]
        else:
            pred_valid = pred_z[:0]
            target_valid = target_z[:0]
    else:
        pred_valid = pred_z
        target_valid = target_z

    if pred_valid.numel() == 0:
        pred_err = pred_z.sum() * 0.0
    else:
        flat_pred = pred_valid.reshape(-1, pred_z.shape[-1])
        flat_target = target_valid.reshape(-1, target_z.shape[-1])
        if loss_type == "l1":
            pred_err = F.l1_loss(flat_pred, flat_target)
        elif loss_type == "smooth_l1":
            pred_err = F.smooth_l1_loss(flat_pred, flat_target, beta=0.05)
        elif loss_type == "cosine":
            pred_err = 1.0 - F.cosine_similarity(flat_pred, flat_target, dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    var_err = vicreg_variance_loss(reg_z) if vicreg_var_weight > 0 else torch.tensor(0.0, device=pred_z.device)
    cov_err = vicreg_covariance_loss(reg_z) if vicreg_cov_weight > 0 else torch.tensor(0.0, device=pred_z.device)
    sig_err = sigreg_loss(reg_z) if sigreg_weight > 0 else torch.tensor(0.0, device=pred_z.device)

    total = pred_err + vicreg_var_weight * var_err + vicreg_cov_weight * cov_err + sigreg_weight * sig_err
    erank = effective_rank(reg_z)

    return {
        "loss": total,
        "pred_loss": pred_err,
        "var_loss": var_err,
        "cov_loss": cov_err,
        "sigreg_loss": sig_err,
        "eff_rank": erank,
    }
