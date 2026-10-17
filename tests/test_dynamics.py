import torch

from latent_play.models import DynamicsConfig, JEPADynamics, VisualDecoder

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


def test_jepa_dynamics_shapes_and_backward():
    torch.manual_seed(0)
    model = JEPADynamics(TINY_DYN)
    frames = torch.rand(2, 3, 3, 32, 32) * 2 - 1  # B=2, T=3
    actions = torch.randint(0, 5, (2, 2))  # actions for t=0, 1

    z = model.encode(frames)  # [2, 3, 16, 32]
    assert z.shape == (2, 3, 16, 32)

    pred = model(z[:, :-1], actions)  # predict z for t=1, 2
    assert pred.shape == (2, 2, 16, 32)

    loss = pred.sum()
    loss.backward()
    assert model.pred_head.weight.grad is not None
    assert model.encoder.patch_embed.proj.weight.grad is not None


def test_jepa_dynamics_rollout():
    torch.manual_seed(0)
    model = JEPADynamics(TINY_DYN).eval()
    frames = torch.rand(2, 2, 3, 32, 32)  # 2 context frames
    actions = torch.randint(0, 5, (2, 4))  # rollout horizon H=4

    z_init = model.encode(frames)
    rollout_z = model.rollout(z_init, actions)
    assert rollout_z.shape == (2, 4, 16, 32)


def test_visual_decoder():
    torch.manual_seed(0)
    decoder = VisualDecoder(image_size=32, patch=8, dim=32, depth=1, heads=2)
    z = torch.randn(2, 16, 32)
    reconstructed = decoder(z)
    assert reconstructed.shape == (2, 3, 32, 32)

    z_seq = torch.randn(2, 4, 16, 32)
    reconstructed_seq = decoder(z_seq)
    assert reconstructed_seq.shape == (2, 4, 3, 32, 32)
