import torch

from latent_play.models import DynamicsConfig, JEPADynamics
from latent_play.planning import ActionAlignment, CEMPlanner

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


def test_action_alignment():
    align = ActionAlignment(num_codes=4, num_actions=5)
    codes = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([2, 3, 0, 1])

    opt = torch.optim.Adam(align.parameters(), lr=0.1)
    for _ in range(50):
        logits = align(codes)
        loss = align.loss(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    preds = align.predict_action(codes)
    torch.testing.assert_close(preds, targets)


def test_cem_planner():
    torch.manual_seed(0)
    dynamics = JEPADynamics(TINY_DYN).eval()
    planner = CEMPlanner(
        dynamics=dynamics,
        horizon=4,
        num_samples=16,
        num_elites=4,
        iterations=2,
    )

    z_curr = torch.randn(1, 16, 32)
    z_goal = torch.randn(1, 16, 32)

    action, seq = planner.plan(z_curr, z_goal)
    assert 0 <= action < 5
    assert seq.shape == (4,)
