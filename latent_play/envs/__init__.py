"""Environment registry."""

from __future__ import annotations

from typing import Any

from .toy_grid import ToyGridEnv


def make_env(name: str, seed: int | None = None, **kwargs: Any):
    """Build an environment by name (``"toy"`` or ``"crafter"``)."""
    if name == "toy":
        return ToyGridEnv(seed=seed, **kwargs)
    if name == "crafter":
        from .crafter_env import CrafterEnv

        return CrafterEnv(seed=seed, **kwargs)
    raise ValueError(f"unknown env '{name}'")


__all__ = ["make_env", "ToyGridEnv"]
