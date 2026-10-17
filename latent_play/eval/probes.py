"""Probes that measure how well discrete latent actions recover the true actions."""

from __future__ import annotations

import numpy as np


def visible_actions(actions: np.ndarray, pos_t: np.ndarray, pos_tp1: np.ndarray, noop: int = 0) -> np.ndarray:
    """Map movement actions that did not move the agent (blocked by a wall) to ``noop``.

    In the toy env a blocked move is pixel-identical to a noop, so no video model
    can tell them apart; this gives the achievable ceiling for the probe.
    """
    moved = np.any(pos_t != pos_tp1, axis=-1)
    return np.where(moved, actions, noop)


def code_action_stats(codes: np.ndarray, actions: np.ndarray, num_codes: int, num_actions: int) -> dict:
    """Confusion-matrix statistics between latent codes and true actions.

    - ``purity``: accuracy of the best many-to-one map code -> action
      (each code predicts its majority action). 1.0 = codes determine the action.
    - ``inverse_purity``: accuracy of the best map action -> code.
    - ``nmi``: normalised mutual information (arithmetic normalisation).
    """
    codes, actions = np.asarray(codes).ravel(), np.asarray(actions).ravel()
    keep = actions >= 0
    codes, actions = codes[keep], actions[keep]
    n = len(codes)
    if n == 0:
        return {"purity": float("nan"), "inverse_purity": float("nan"), "nmi": float("nan"), "n": 0}
    C = np.zeros((num_codes, num_actions), dtype=np.float64)
    np.add.at(C, (codes, actions), 1)
    P = C / n
    pc, pa = P.sum(1), P.sum(0)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (pc[:, None] * pa[None, :])[nz])).sum())
    h = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())  # noqa: E731
    denom = (h(pc) + h(pa)) / 2
    return {
        "purity": float(C.max(1).sum() / n),
        "inverse_purity": float(C.max(0).sum() / n),
        "nmi": mi / denom if denom > 0 else 0.0,
        "n": int(n),
        "confusion": C.astype(np.int64),
    }
