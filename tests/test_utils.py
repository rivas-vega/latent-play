import torch
from torch import nn

from latent_play.utils import EMA, latest_checkpoint, load_checkpoint, save_checkpoint
from latent_play.utils.ckpt import list_checkpoints


def test_ema_schedule_and_update():
    model = nn.Linear(3, 3)
    ema = EMA(model, total_steps=100, tau_start=0.9, tau_end=1.0)
    assert abs(ema.tau(0) - 0.9) < 1e-9 and abs(ema.tau(100) - 1.0) < 1e-9
    assert all(not p.requires_grad for p in ema.target.parameters())
    with torch.no_grad():
        model.weight.add_(1.0)
    before = ema.target.weight.clone()
    ema.update(model, step=0)  # tau = 0.9 -> moves 10% of the way
    torch.testing.assert_close(ema.target.weight, before + 0.1)
    ema.update(model, step=100)  # tau = 1 -> frozen
    torch.testing.assert_close(ema.target.weight, before + 0.1)


def test_checkpoint_roundtrip_and_rotation(tmp_path):
    model = nn.Linear(2, 2)
    for step in (10, 20, 30, 40):
        save_checkpoint(tmp_path, {"model": model.state_dict()}, step=step, keep=2)
    assert [p.name for p in list_checkpoints(tmp_path)] == ["ckpt_000000030.pt", "ckpt_000000040.pt"]
    state = load_checkpoint(latest_checkpoint(tmp_path))
    assert state["step"] == 40
    torch.testing.assert_close(state["model"]["weight"], model.weight)
    assert not list(tmp_path.glob("*.tmp"))


def test_latest_checkpoint_empty(tmp_path):
    assert latest_checkpoint(tmp_path / "missing") is None
