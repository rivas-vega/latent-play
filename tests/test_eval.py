import torch

from latent_play.eval import (
    compute_voe_score,
    inject_disappearance_violation,
    inject_teleport_violation,
)
from latent_play.models import DynamicsConfig, JEPADynamics

TINY_DYN = DynamicsConfig(
    image_size=32,
    patch=8,
    dim=32,
    enc_depth=1,
    enc_heads=2,
    pred_depth=1,
    pred_heads=2,
    num_actions=5,
    max_context=8,
    pool_tokens=False,
)


def test_voe_injections():
    frames = torch.randn(2, 4, 3, 32, 32)
    tele = inject_teleport_violation(frames)
    assert tele.shape == frames.shape
    assert not torch.equal(tele[:, 2:], frames[:, 2:])
    assert torch.equal(tele[:, :2], frames[:, :2])

    disap = inject_disappearance_violation(frames)
    assert disap.shape == frames.shape
    assert not torch.equal(disap[:, 2:], frames[:, 2:])


def test_compute_voe_score():
    torch.manual_seed(0)
    model = JEPADynamics(TINY_DYN)
    frames = torch.randn(2, 4, 3, 32, 32)
    actions = torch.randint(0, 5, (2, 3))

    res = compute_voe_score(model, frames, actions, violation_type="teleport")
    assert "surprise_plausible" in res
    assert "surprise_violated" in res
    assert "voe_ratio" in res
    assert res["voe_ratio"] >= 0.0
