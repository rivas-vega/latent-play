"""Crafter wrapper exposing the same ``reset``/``step`` API as :class:`ToyGridEnv`."""

from __future__ import annotations

import copy
import numpy as np

CRAFTER_ACTIONS = (
    "noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
)


class CrafterEnv:
    """Thin wrapper around ``crafter.Env``.

    ``info["pos"]`` is the player's world position ``(x, y)``. Crafter seeds each
    episode deterministically from ``(seed, episode_index)``, so every worker
    with a distinct ``seed`` produces distinct worlds.
    """

    action_names = CRAFTER_ACTIONS
    num_actions = len(CRAFTER_ACTIONS)
    movement_actions = (1, 2, 3, 4)

    def __init__(self, size: int = 64, max_steps: int = 1000, seed: int | None = None) -> None:
        try:
            import crafter
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError("Crafter is not installed: `pip install -e .[crafter]`") from e
        self._env = crafter.Env(size=(size, size), length=max_steps, seed=seed)
        self.size = size

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        # Crafter has no per-reset seed; episode seeds derive from the constructor seed.
        obs = self._env.reset()
        return obs, {"pos": np.asarray(self._env._player.pos, dtype=np.int64).copy(), "t": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self._env.step(int(action))
        out = {"pos": np.asarray(info["player_pos"], dtype=np.int64).copy(), "t": self._env._step}
        return obs, float(reward), bool(done), out

    def clone_state(self):
        """Deep-copies the internal Crafter world state so goal rollouts can be rewound to t=0."""
        return copy.deepcopy(self._env)

    def restore_state(self, state) -> tuple[np.ndarray, dict]:
        """Restores a previously cloned Crafter world state and returns (obs, info)."""
        self._env = copy.deepcopy(state)
        obs = self._env.render()
        return obs, {"pos": np.asarray(self._env._player.pos, dtype=np.int64).copy(), "t": self._env._step}
