import json

import numpy as np
import pytest
import torch

from latent_play.data import VideoWindows, frames_to_float
from latent_play.data.collect import CollectConfig, collect
from latent_play.envs.toy_grid import _DELTAS


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    cfg = CollectConfig(env="toy", out=str(root), total_steps=3000, workers=2, seed=0,
                        env_kwargs={"max_steps": 40, "distractor": False})
    collect(cfg)
    return root


def _load(root, name):
    meta = json.loads((root / "meta.json").read_text())
    return np.concatenate([np.load(root / s / f"{name}.npy") for s in meta["shards"]])


def test_meta_and_sizes(toy_root):
    meta = json.loads((toy_root / "meta.json").read_text())
    assert meta["total_steps"] == 3000 and len(meta["shards"]) == 2
    assert len(_load(toy_root, "obs")) == 3000


def test_actions_align_with_position_changes(toy_root):
    """action[i] must explain pos[i] -> pos[i+1] for every within-episode transition."""
    a, pos, ep = _load(toy_root, "actions"), _load(toy_root, "pos"), _load(toy_root, "episode")
    within = (ep[:-1] == ep[1:])
    assert np.all(a[:-1][within] >= 0), "within-episode frames must have an action"
    expected = np.clip(pos[:-1] + _DELTAS[np.maximum(a[:-1], 0)], 0, 7)
    np.testing.assert_array_equal(expected[within], pos[1:][within])


def test_terminal_frames_have_no_action(toy_root):
    a, ep = _load(toy_root, "actions"), _load(toy_root, "episode")
    boundary = np.r_[ep[:-1] != ep[1:], True]
    assert np.all(a[boundary] == -1)
    # episodes of max_steps=40 contain 41 frames (incl. terminal), except truncated ones at shard ends
    _, lengths = np.unique(ep, return_counts=True)
    assert np.all(lengths <= 41) and np.median(lengths) == 41


def test_windows_never_cross_episodes(toy_root):
    ds = VideoWindows(toy_root, num_frames=8, split="all", return_true_actions=True)
    assert len(ds) > 0
    for i in np.random.default_rng(0).choice(len(ds), 200, replace=False):
        item = ds[int(i)]
        assert item["frames"].shape == (8, 3, 64, 64) and item["frames"].dtype == torch.uint8
        assert torch.all(item["action_true"] >= 0)


def test_train_val_split_is_disjoint(toy_root):
    tr = VideoWindows(toy_root, num_frames=4, split="train", val_fraction=0.2)
    va = VideoWindows(toy_root, num_frames=4, split="val", val_fraction=0.2)
    al = VideoWindows(toy_root, num_frames=4, split="all")
    assert len(tr) + len(va) == len(al) and len(va) > 0
    ep = _load(toy_root, "episode")
    # map global windows back to episodes via shard offsets
    def episodes(ds):
        out = set()
        for k, s in zip(ds._index_shard, ds._index_start):
            out.add(int(np.load(toy_root / ds.meta["shards"][k] / "episode.npy", mmap_mode="r")[s]))
        return out
    assert episodes(tr).isdisjoint(episodes(va))
    assert len(ep) == 3000


def test_label_masking_no_leak_and_nested(toy_root):
    fracs = [0.0, 0.01, 0.1, 1.0]
    sets = []
    for f in fracs:
        ds = VideoWindows(toy_root, num_frames=2, split="all", label_fraction=f, return_true_actions=True)
        labeled = np.array([bool(ds[i]["labeled"][0]) for i in range(len(ds))])
        acts = np.array([int(ds[i]["action"][0]) for i in range(len(ds))])
        true = np.array([int(ds[i]["action_true"][0]) for i in range(len(ds))])
        assert np.all(acts[~labeled] == -1), "unlabelled transitions must hide the action"
        np.testing.assert_array_equal(acts[labeled], true[labeled])
        sets.append(labeled)
        if f in (0.0, 1.0):
            assert labeled.mean() == f
    for small, big in zip(sets, sets[1:]):
        assert np.all(big[small]), "label subsets must be nested"
    assert abs(sets[2].mean() - 0.1) < 0.03


def test_default_dataset_hides_true_actions(toy_root):
    item = VideoWindows(toy_root, num_frames=3, label_fraction=0.0)[0]
    assert "action_true" not in item and torch.all(item["action"] == -1)


def test_frames_to_float_range():
    x = torch.tensor([0, 255], dtype=torch.uint8)
    np.testing.assert_allclose(frames_to_float(x).numpy(), [-1.0, 1.0])


def test_refuses_to_overwrite(toy_root):
    with pytest.raises(FileExistsError):
        collect(CollectConfig(env="toy", out=str(toy_root), total_steps=10, workers=1))
