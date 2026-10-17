import numpy as np
import pytest

from latent_play.envs import ToyGridEnv, make_env


def test_shapes_and_dtype():
    env = ToyGridEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    assert info["pos"].shape == (2,)


def test_actions_move_agent_and_clip():
    env = ToyGridEnv(seed=1, distractor=False)
    env.reset()
    env._agent = np.array([3, 3])
    expected = {0: (3, 3), 1: (2, 3), 2: (4, 3), 3: (3, 2), 4: (3, 4)}
    for a, pos in expected.items():
        env._agent = np.array([3, 3])
        _, _, _, info = env.step(a)
        assert tuple(info["pos"]) == pos
    env._agent = np.array([0, 0])
    _, _, _, info = env.step(1)  # up at the border -> stays
    assert tuple(info["pos"]) == (0, 0)


def test_determinism():
    a = ToyGridEnv(seed=7)
    b = ToyGridEnv(seed=7)
    oa, _ = a.reset()
    ob, _ = b.reset()
    np.testing.assert_array_equal(oa, ob)
    for act in [1, 2, 3, 4, 0, 1]:
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
        np.testing.assert_array_equal(oa, ob)


def test_episode_terminates():
    env = ToyGridEnv(seed=0, max_steps=5)
    env.reset()
    dones = [env.step(0)[2] for _ in range(5)]
    assert dones == [False] * 4 + [True]


def test_agent_pixels_follow_position():
    env = ToyGridEnv(seed=3, distractor=False)
    obs, info = env.reset()
    r, c = info["pos"] * env.cell + env.cell // 2
    np.testing.assert_array_equal(obs[r, c], [230, 60, 50])


def test_crafter_wrapper():
    pytest.importorskip("crafter")
    env = make_env("crafter", seed=0, size=64)
    obs, info = env.reset()
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    state = env.clone_state()
    obs_step, r, done, info_step = env.step(1)
    assert info_step["pos"].shape == (2,) and isinstance(done, bool)
    obs_restored, info_restored = env.restore_state(state)
    np.testing.assert_array_equal(obs, obs_restored)
    np.testing.assert_array_equal(info["pos"], info_restored["pos"])


def test_toy_clone_and_restore_state():
    env = ToyGridEnv(seed=11)
    obs0, info0 = env.reset()
    state = env.clone_state()
    for a in [1, 2, 3, 4]:
        env.step(a)
    obs_r, info_r = env.restore_state(state)
    np.testing.assert_array_equal(obs0, obs_r)
    np.testing.assert_array_equal(info0["pos"], info_r["pos"])
