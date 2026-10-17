"""Toy grid-world rendered in pixels, used to debug the full pipeline in minutes.

An agent (red square) moves on an ``grid x grid`` lattice. A distractor (blue
square) performs an action-independent random walk, and the background texture
is resampled every episode. A correct latent action model must therefore encode
*the agent's* motion and ignore the distractor and the background.
"""

from __future__ import annotations

import numpy as np

# (dy, dx) per action. Moves into the border leave the agent in place.
_DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)

AGENT_COLOR = np.array([230, 60, 50], dtype=np.uint8)
DISTRACTOR_COLOR = np.array([60, 110, 230], dtype=np.uint8)


class ToyGridEnv:
    """Minimal pixel grid-world with a uniform ``reset``/``step`` API.

    Observations are ``uint8`` arrays of shape ``(size, size, 3)``.
    ``info["pos"]`` holds the agent cell ``(row, col)``.
    """

    action_names = ("noop", "up", "down", "left", "right")
    num_actions = 5
    movement_actions = (1, 2, 3, 4)

    def __init__(
        self,
        size: int = 64,
        grid: int = 8,
        max_steps: int = 64,
        distractor: bool = True,
        seed: int | None = None,
    ) -> None:
        if size % grid != 0:
            raise ValueError(f"size ({size}) must be divisible by grid ({grid})")
        self.size = size
        self.grid = grid
        self.cell = size // grid
        self.max_steps = max_steps
        self.use_distractor = distractor
        self._rng = np.random.default_rng(seed)
        self._background: np.ndarray | None = None
        self._agent = np.zeros(2, dtype=np.int64)
        self._distractor = np.zeros(2, dtype=np.int64)
        self._t = 0

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        base = self._rng.integers(20, 90, size=3)
        noise = self._rng.integers(-12, 13, size=(self.size, self.size, 1))
        self._background = np.clip(base + noise, 0, 255).astype(np.uint8)
        self._agent = self._rng.integers(0, self.grid, size=2)
        self._distractor = self._rng.integers(0, self.grid, size=2)
        self._t = 0
        return self._render(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if not 0 <= int(action) < self.num_actions:
            raise ValueError(f"invalid action {action}")
        self._agent = np.clip(self._agent + _DELTAS[int(action)], 0, self.grid - 1)
        if self.use_distractor:
            move = _DELTAS[self._rng.integers(0, self.num_actions)]
            self._distractor = np.clip(self._distractor + move, 0, self.grid - 1)
        self._t += 1
        done = self._t >= self.max_steps
        return self._render(), 0.0, done, self._info()

    def clone_state(self) -> dict:
        """Captures the exact environment state (background, agent, distractor, RNG) for reproducible resets."""
        assert self._background is not None, "call reset() first"
        return {
            "background": self._background.copy(),
            "agent": self._agent.copy(),
            "distractor": self._distractor.copy(),
            "t": self._t,
            "rng_state": dict(self._rng.bit_generator.state),
        }

    def restore_state(self, state: dict) -> tuple[np.ndarray, dict]:
        """Restores a previously cloned environment state and returns (obs, info)."""
        self._background = state["background"].copy()
        self._agent = state["agent"].copy()
        self._distractor = state["distractor"].copy()
        self._t = int(state["t"])
        self._rng.bit_generator.state = state["rng_state"]
        return self._render(), self._info()

    # ------------------------------------------------------------- internals
    def _info(self) -> dict:
        return {"pos": self._agent.copy(), "t": self._t}

    def _paint(self, frame: np.ndarray, cell: np.ndarray, color: np.ndarray) -> None:
        m = max(1, self.cell // 8)  # small margin so adjacent squares stay distinct
        r0, c0 = cell * self.cell
        frame[r0 + m : r0 + self.cell - m, c0 + m : c0 + self.cell - m] = color

    def _render(self) -> np.ndarray:
        assert self._background is not None, "call reset() first"
        frame = self._background.copy()
        if self.use_distractor:
            self._paint(frame, self._distractor, DISTRACTOR_COLOR)
        self._paint(frame, self._agent, AGENT_COLOR)  # agent drawn on top
        return frame
