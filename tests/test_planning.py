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


def test_action_alignment_with_context():
    dim = 32
    align = ActionAlignment(num_codes=4, num_actions=5, dim=dim, use_state_context=True)
    codes = torch.tensor([0, 1, 2, 3])
    # Distinct context vectors for each sample
    z = torch.randn(4, 16, dim)
    targets = torch.tensor([1, 2, 3, 4])

    opt = torch.optim.Adam(align.parameters(), lr=0.05)
    for _ in range(60):
        logits = align(codes, z=z)
        loss = align.loss(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    preds = align.predict_action(codes, z=z)
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


def test_cem_planner_with_context_alignment():
    torch.manual_seed(0)
    dynamics = JEPADynamics(TINY_DYN).eval()
    alignment = ActionAlignment(num_codes=5, num_actions=5, dim=32, use_state_context=True)
    planner = CEMPlanner(
        dynamics=dynamics,
        horizon=4,
        num_samples=16,
        num_elites=4,
        iterations=2,
        alignment=alignment,
    )

    z_curr = torch.randn(1, 16, 32)
    z_goal = torch.randn(1, 16, 32)

    action, seq = planner.plan(z_curr, z_goal)
    assert 0 <= action < 5
    assert seq.shape == (4,)


def test_cem_planner_with_stagnation_penalty():
    torch.manual_seed(0)
    dynamics = JEPADynamics(TINY_DYN).eval()
    planner = CEMPlanner(
        dynamics=dynamics,
        horizon=4,
        num_samples=16,
        num_elites=4,
        iterations=2,
        stagnation_penalty=0.5,
        stagnation_threshold=0.01,
    )

    z_curr = torch.randn(1, 16, 32)
    z_goal = torch.randn(1, 16, 32)

    action, seq = planner.plan(z_curr, z_goal)
    assert 0 <= action < 5
    assert seq.shape == (4,)


def test_cem_planner_pooled():
    torch.manual_seed(0)
    cfg = DynamicsConfig(
        image_size=32,
        patch=8,
        dim=32,
        enc_depth=1,
        enc_heads=2,
        pred_depth=1,
        pred_heads=2,
        num_actions=5,
        max_context=8,
        pool_tokens=True,
    )
    dynamics = JEPADynamics(cfg).eval()
    planner = CEMPlanner(
        dynamics=dynamics,
        horizon=4,
        num_samples=16,
        num_elites=4,
        iterations=2,
    )

    z_curr = torch.randn(1, 32)
    z_goal = torch.randn(1, 32)

    action, seq = planner.plan(z_curr, z_goal)
    assert 0 <= action < 5
    assert seq.shape == (4,)


def test_fit_alignment_probe():
    from scripts.sweep_labels import fit_alignment_probe

    torch.manual_seed(42)
    num_codes = 4
    num_actions = 4
    N = 200
    # Perfect correspondence: code k corresponds to action k
    codes = torch.randint(0, num_codes, (N,))
    actions = codes.clone()

    align = fit_alignment_probe(
        codes,
        actions,
        z=None,
        num_codes=num_codes,
        num_actions=num_actions,
        dim=32,
        device=torch.device("cpu"),
        epochs=10,
        lr=0.05,
    )

    preds = align.predict_action(codes)
    acc = (preds == actions).float().mean().item()
    assert acc > 0.95


def test_cem_planner_momentum():
    torch.manual_seed(0)
    dynamics = JEPADynamics(TINY_DYN).eval()
    planner = CEMPlanner(
        dynamics=dynamics,
        horizon=4,
        num_samples=16,
        num_elites=4,
        iterations=3,
        alpha=0.6,
    )
    assert planner.alpha == 0.6
    z_curr = torch.randn(1, 16, 32)
    z_goal = torch.randn(1, 16, 32)
    action, seq = planner.plan(z_curr, z_goal)
    assert 0 <= action < 5
    assert seq.shape == (4,)



