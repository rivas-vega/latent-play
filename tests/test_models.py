import numpy as np
import torch

from latent_play.eval import code_action_stats, visible_actions
from latent_play.models import LAMConfig, LatentActionModel, VectorQuantizer
from latent_play.models.layers import patchify, unpatchify

TINY = LAMConfig(image_size=32, patch=8, context=2, dim=32, enc_depth=1, dec_depth=1, heads=2, num_codes=4, code_dim=8)


def test_patchify_roundtrip():
    x = torch.randn(2, 3, 3, 32, 32)
    torch.testing.assert_close(unpatchify(patchify(x, 8), 8), x)


def test_lam_shapes_and_backward():
    torch.manual_seed(0)
    model = LatentActionModel(TINY)
    frames = torch.rand(2, 3, 3, 32, 32) * 2 - 1
    out = model(frames)
    assert out["pred"].shape == (2, 2, 3, 32, 32)
    assert out["indices"].shape == (2, 2)
    out["loss"].backward()
    assert model.to_action.weight.grad is not None, "gradient must reach the encoder through the VQ"
    assert model.to_action.weight.grad.abs().sum() > 0


def test_decoder_starts_as_copy_last_frame():
    model = LatentActionModel(TINY)
    frames = torch.rand(1, 3, 3, 32, 32)
    torch.testing.assert_close(model(frames)["pred"], frames[:, :-1])


def test_encoder_is_causal():
    """The action for transition t -> t+1 must not depend on frames after t+1."""
    torch.manual_seed(0)
    model = LatentActionModel(TINY).eval()
    a = torch.rand(1, 3, 3, 32, 32)
    b = a.clone()
    b[:, 2] = torch.rand_like(b[:, 2])  # change the last frame only
    za, zb = model.encode(a), model.encode(b)
    torch.testing.assert_close(za[:, 0], zb[:, 0])  # transition 0->1 unaffected
    assert not torch.allclose(za[:, 1], zb[:, 1])


def test_decoder_is_causal():
    torch.manual_seed(0)
    model = LatentActionModel(TINY).eval()
    torch.nn.init.normal_(model.head.weight, std=0.02)
    past = torch.rand(1, 2, 3, 32, 32)
    acts = torch.randn(1, 2, TINY.code_dim)
    past2 = past.clone()
    past2[:, 1] = torch.rand_like(past2[:, 1])
    torch.testing.assert_close(model.decode(past, acts)[:, 0], model.decode(past2, acts)[:, 0])


def test_vq_perplexity_and_restart():
    torch.manual_seed(0)
    vq = VectorQuantizer(num_codes=4, dim=8, restart_every=1, dead_threshold=0.5)
    vq.codebook.data[1:] = vq.codebook.data[0]  # collapse all codes onto one
    vq.usage.data[1:] = 0.0
    vq.train()
    z = torch.randn(256, 8)
    out = vq(z)
    assert out["perplexity"].item() < 1.01
    assert vq.usage[1:].min() > 0  # dead codes were restarted to batch samples
    out2 = vq(z)
    assert out2["perplexity"].item() > 1.5


def test_code_action_stats():
    actions = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    perfect = code_action_stats(actions[::-1].copy() % 4, actions[::-1].copy(), 4, 4)
    assert perfect["purity"] == 1.0 and abs(perfect["nmi"] - 1.0) < 1e-9
    const = code_action_stats(np.zeros(8, int), actions, 4, 4)
    assert const["purity"] == 0.25 and const["nmi"] == 0.0
    ignored = code_action_stats(np.array([0, 1]), np.array([0, -1]), 2, 2)
    assert ignored["n"] == 1


def test_visible_actions():
    a = np.array([1, 2, 0])
    p0 = np.array([[0, 0], [3, 3], [1, 1]])
    p1 = np.array([[0, 0], [4, 3], [1, 1]])
    np.testing.assert_array_equal(visible_actions(a, p0, p1), [0, 2, 0])


def test_lam_with_state_context():
    torch.manual_seed(0)
    cfg = LAMConfig(
        image_size=32,
        patch=8,
        context=2,
        dim=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        num_codes=4,
        code_dim=8,
        state_context=True,
    )
    model = LatentActionModel(cfg)
    frames = torch.rand(2, 3, 3, 32, 32) * 2 - 1
    out = model(frames)
    assert out["pred"].shape == (2, 2, 3, 32, 32)
    assert out["indices"].shape == (2, 2)
    out["loss"].backward()
    assert model.to_action.weight.grad is not None

