import torch

from latent_play.losses import (
    effective_rank,
    jepa_loss,
    sigreg_loss,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)


def test_effective_rank():
    # 1D line: singular values [s1, 0, 0, 0] -> rank = exp(0) = 1
    z_line = torch.zeros(100, 10)
    z_line[:, 0] = torch.linspace(-1, 1, 100)
    erank_line = effective_rank(z_line).item()
    assert abs(erank_line - 1.0) < 0.1

    # Full isotropic Gaussian: full rank
    torch.manual_seed(42)
    z_iso = torch.randn(200, 16)
    erank_iso = effective_rank(z_iso).item()
    assert erank_iso > 12.0  # high dimensional spread


def test_vicreg_variance_loss():
    # Collapsed representation (zero variance)
    z_collapsed = torch.zeros(32, 16)
    loss = vicreg_variance_loss(z_collapsed, gamma=1.0)
    assert abs(loss.item() - 1.0) < 1e-2

    # High variance representation (std >= 1.0)
    z_spread = torch.randn(1000, 16) * 1.5
    loss_spread = vicreg_variance_loss(z_spread, gamma=1.0)
    assert loss_spread.item() < 0.05


def test_vicreg_covariance_loss():
    # Identity covariance: zero off-diagonal
    torch.manual_seed(0)
    z = torch.randn(500, 10)
    loss = vicreg_covariance_loss(z)
    assert loss.item() < 0.2


def test_jepa_loss_backward():
    pred_z = torch.randn(4, 8, 32, requires_grad=True)
    target_z = torch.randn(4, 8, 32)
    out = jepa_loss(pred_z, target_z, loss_type="l1", vicreg_var_weight=0.1, sigreg_weight=0.05)
    assert "loss" in out and "eff_rank" in out and "sigreg_loss" in out
    out["loss"].backward()
    assert pred_z.grad is not None
    assert pred_z.grad.abs().sum() > 0


def test_sigreg_loss():
    torch.manual_seed(0)
    z_gauss = torch.randn(200, 16)
    loss_gauss = sigreg_loss(z_gauss, num_slices=32)

    z_collapsed = torch.zeros(200, 16)
    loss_collapsed = sigreg_loss(z_collapsed, num_slices=32)

    assert loss_gauss.item() < loss_collapsed.item()
    assert loss_gauss.item() < 0.05
    assert loss_collapsed.item() > 0.08


def test_jepa_loss_4d_online_z_and_valid_mask():
    # Simulate 4D spatial-temporal tensor [B=8, T=8, N=64, D=32] -> 4096 tokens
    online_z = torch.randn(8, 8, 64, 32, requires_grad=True)
    pred_z = torch.randn(8, 8, 64, 32, requires_grad=True)
    target_z = torch.randn(8, 8, 64, 32)
    valid_mask = torch.zeros(8, 8, dtype=torch.bool)
    valid_mask[:, :2] = True

    out = jepa_loss(
        pred_z,
        target_z,
        online_z=online_z,
        valid_mask=valid_mask,
        vicreg_var_weight=0.05,
        vicreg_cov_weight=0.05,
        sigreg_weight=0.05,
    )
    out["loss"].backward()
    assert online_z.grad is not None and online_z.grad.abs().sum() > 0
    assert pred_z.grad is not None and pred_z.grad[:, :2].abs().sum() > 0
    assert pred_z.grad[:, 2:].abs().sum() == 0
