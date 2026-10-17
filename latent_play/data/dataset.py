"""Windowed video dataset with controlled access to action labels.

Key design points:

* **No label leakage by construction.** ``batch["action"]`` is ``-1`` wherever the
  transition is *unlabelled* under the requested ``label_fraction``. Ground-truth
  actions are only returned when ``return_true_actions=True`` (evaluation).
* **Nested label subsets.** Every transition gets a fixed uniform draw ``u``;
  it is labelled iff ``u < label_fraction``. Hence the 0.1 % subset is contained
  in the 1 % subset, which is contained in the 10 % subset, ... — the label
  sweep differs only in the amount of supervision, never in *which* data.
* **Episode-level train/val split**, deterministic and independent of shard order.
* Arrays are opened lazily with ``mmap_mode="r"`` (safe with DataLoader workers).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_ARRAYS = ("obs", "actions", "episode", "timestep", "pos", "rewards")


def episode_in_val(episode_ids: np.ndarray, val_fraction: float) -> np.ndarray:
    """Deterministic hash-based split (Knuth multiplicative hash)."""
    h = (episode_ids.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return (h.astype(np.float64) / 2**32) < val_fraction


def label_uniforms(num_steps: int, shard_index: int, label_seed: int) -> np.ndarray:
    return np.random.default_rng([label_seed, shard_index]).random(num_steps)


class VideoWindows(Dataset):
    """Windows of ``num_frames`` consecutive frames inside a single episode.

    Each item is a dict:

    - ``frames``  uint8 [T, 3, H, W]
    - ``action``  int64 [T-1]  ground-truth action if labelled, else -1
    - ``labeled`` bool  [T-1]
    - ``pos``     int64 [T, 2]  (always returned; for evaluation / probes)
    - ``action_true`` int64 [T-1]  only if ``return_true_actions``
    """

    def __init__(
        self,
        root: str | Path,
        num_frames: int,
        split: str = "train",
        val_fraction: float = 0.05,
        label_fraction: float = 1.0,
        label_seed: int = 0,
        stride: int = 1,
        return_true_actions: bool = False,
    ) -> None:
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train/val/all, got {split}")
        if not 0.0 <= label_fraction <= 1.0:
            raise ValueError("label_fraction must be in [0, 1]")
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.num_frames = num_frames
        self.label_fraction = label_fraction
        self.return_true_actions = return_true_actions
        self.num_actions = self.meta["num_actions"]
        self._shard_dirs = [self.root / s for s in self.meta["shards"]]
        self._arrays: list[dict[str, np.ndarray]] | None = None  # opened lazily

        index_shard, index_start, self._label_u = [], [], []
        for k, d in enumerate(self._shard_dirs):
            ep = np.load(d / "episode.npy", mmap_mode="r")
            n = len(ep)
            self._label_u.append(label_uniforms(n, k, label_seed))
            if n < num_frames:
                continue
            ep = np.asarray(ep)
            starts = np.arange(0, n - num_frames + 1, stride)
            same_episode = ep[starts] == ep[starts + num_frames - 1]  # episodes are contiguous runs
            starts = starts[same_episode]
            if split != "all":
                is_val = episode_in_val(ep[starts], val_fraction)
                starts = starts[is_val] if split == "val" else starts[~is_val]
            index_shard.append(np.full(len(starts), k, dtype=np.int32))
            index_start.append(starts.astype(np.int64))
        self._index_shard = np.concatenate(index_shard) if index_shard else np.zeros(0, np.int32)
        self._index_start = np.concatenate(index_start) if index_start else np.zeros(0, np.int64)

    def _open(self) -> list[dict[str, np.ndarray]]:
        if self._arrays is None:
            self._arrays = [
                {name: np.load(d / f"{name}.npy", mmap_mode="r") for name in _ARRAYS} for d in self._shard_dirs
            ]
        return self._arrays

    def __len__(self) -> int:
        return len(self._index_start)

    def labeled_fraction_actual(self) -> float:
        """Fraction of all stored steps that fall in the labelled subset (for logging)."""
        u = np.concatenate(self._label_u)
        return float((u < self.label_fraction).mean()) if len(u) else 0.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        k, s = int(self._index_shard[idx]), int(self._index_start[idx])
        T = self.num_frames
        arr = self._open()[k]
        frames = torch.from_numpy(np.array(arr["obs"][s : s + T])).permute(0, 3, 1, 2)
        true_a = np.asarray(arr["actions"][s : s + T - 1], dtype=np.int64)
        labeled = self._label_u[k][s : s + T - 1] < self.label_fraction
        item = {
            "frames": frames,
            "action": torch.from_numpy(np.where(labeled, true_a, -1)),
            "labeled": torch.from_numpy(labeled),
            "pos": torch.from_numpy(np.asarray(arr["pos"][s : s + T], dtype=np.int64)),
        }
        if self.return_true_actions:
            item["action_true"] = torch.from_numpy(true_a)
        return item


def frames_to_float(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [.., 3, H, W] in [0, 255] -> float in [-1, 1]. Do this on the GPU."""
    return frames.float().div_(127.5).sub_(1.0)
