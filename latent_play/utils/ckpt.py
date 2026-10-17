"""Crash-safe checkpointing for preemptible Colab/Kaggle sessions.

Writes are atomic (temp file + ``os.replace``), so a disconnect mid-save never
corrupts the latest checkpoint. Only the newest ``keep`` checkpoints are kept.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

_PATTERN = re.compile(r"^ckpt_(\d{9})\.pt$")


def save_checkpoint(ckpt_dir: str | Path, state: dict[str, Any], step: int, keep: int = 3) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{step:09d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({**state, "step": step}, tmp)
    os.replace(tmp, path)
    for old in list_checkpoints(ckpt_dir)[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)
    return path


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []
    found = [p for p in ckpt_dir.iterdir() if _PATTERN.match(p.name)]
    return sorted(found, key=lambda p: int(_PATTERN.match(p.name).group(1)))


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
