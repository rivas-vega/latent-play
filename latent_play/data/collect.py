"""Parallel, low-memory data collection.

Each worker writes one *shard* directly to disk through ``np.lib.format.open_memmap``,
so collecting millions of frames never needs more than a few MB of RAM per worker.

Shard layout (``<root>/shard_XXX/``), all arrays aligned on the step index ``i``:

- ``obs.npy``       uint8  [N, H, W, 3]  observation at step i
- ``actions.npy``   int16  [N]           action taken at obs[i]; -1 on the last frame of an episode
- ``rewards.npy``   float32[N]           reward received after actions[i]
- ``episode.npy``   int64  [N]           globally unique episode id (contiguous runs)
- ``timestep.npy``  int32  [N]           step index within the episode
- ``pos.npy``       int16  [N, 2]        agent position (evaluation only)

A transition ``obs[i] -> obs[i+1]`` is valid iff ``episode[i] == episode[i+1]``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from latent_play.envs import make_env

EPISODE_STRIDE = 10**7  # episode id = shard_id * EPISODE_STRIDE + local episode index


@dataclass
class PolicyConfig:
    """Segment policy: pick an action, repeat it for a random number of steps.

    With probability ``p_explore`` a segment is a long, persistent movement
    (better state coverage than pure random walks); otherwise it is a uniformly
    random action repeated ``repeat_min..repeat_max`` times.
    """

    p_explore: float = 0.2
    repeat_min: int = 1
    repeat_max: int = 4
    explore_min: int = 4
    explore_max: int = 12


class SegmentPolicy:
    def __init__(self, num_actions: int, movement_actions: tuple[int, ...], cfg: PolicyConfig, rng: np.random.Generator):
        self.num_actions = num_actions
        self.movement_actions = np.asarray(movement_actions)
        self.cfg = cfg
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self._action = 0
        self._remaining = 0

    def act(self) -> int:
        if self._remaining <= 0:
            c = self.cfg
            if self.rng.random() < c.p_explore:
                self._action = int(self.rng.choice(self.movement_actions))
                self._remaining = int(self.rng.integers(c.explore_min, c.explore_max + 1))
            else:
                self._action = int(self.rng.integers(0, self.num_actions))
                self._remaining = int(self.rng.integers(c.repeat_min, c.repeat_max + 1))
        self._remaining -= 1
        return self._action


@dataclass
class CollectConfig:
    env: str = "toy"
    out: str = "data/toy"
    total_steps: int = 50_000
    workers: int = 4
    seed: int = 0
    env_kwargs: dict = field(default_factory=dict)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def collect_shard(cfg: CollectConfig, shard_id: int, num_steps: int, show_progress: bool = False) -> dict:
    """Collect ``num_steps`` frames into ``<cfg.out>/shard_<id>``. Returns shard metadata."""
    seed = cfg.seed * 1000 + shard_id
    env = make_env(cfg.env, seed=seed, **cfg.env_kwargs)
    rng = np.random.default_rng(seed)
    policy = SegmentPolicy(env.num_actions, env.movement_actions, cfg.policy, rng)

    shard_dir = Path(cfg.out) / f"shard_{shard_id:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    obs, info = env.reset()
    mm = lambda name, dtype, shape: np.lib.format.open_memmap(  # noqa: E731
        shard_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
    )
    obs_buf = mm("obs", np.uint8, (num_steps, *obs.shape))
    act_buf = mm("actions", np.int16, (num_steps,))
    rew_buf = mm("rewards", np.float32, (num_steps,))
    ep_buf = mm("episode", np.int64, (num_steps,))
    t_buf = mm("timestep", np.int32, (num_steps,))
    pos_buf = mm("pos", np.int16, (num_steps, 2))

    episode, t, pending_reset, n_episodes = 0, 0, False, 1
    for i in tqdm(range(num_steps), disable=not show_progress, desc=f"shard {shard_id}"):
        obs_buf[i] = obs
        pos_buf[i] = info["pos"]
        ep_buf[i] = shard_id * EPISODE_STRIDE + episode
        t_buf[i] = t
        rew_buf[i] = 0.0
        if pending_reset or i == num_steps - 1:
            # Terminal frame of an episode (or of the shard): no action follows it.
            act_buf[i] = -1
            if pending_reset and i < num_steps - 1:
                obs, info = env.reset()
                policy.reset()
                episode, t, pending_reset = episode + 1, 0, False
                n_episodes += 1
            continue
        a = policy.act()
        obs, reward, done, info = env.step(a)
        act_buf[i] = a
        rew_buf[i] = reward
        t += 1
        pending_reset = done

    for buf in (obs_buf, act_buf, rew_buf, ep_buf, t_buf, pos_buf):
        buf.flush()
    meta = {"shard_id": shard_id, "num_steps": num_steps, "num_episodes": n_episodes, "seed": seed}
    (shard_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def collect(cfg: CollectConfig) -> dict:
    """Collect ``cfg.total_steps`` frames split over ``cfg.workers`` shards."""
    out = Path(cfg.out)
    if (out / "meta.json").exists():
        raise FileExistsError(f"{out} already contains a dataset; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    base, rem = divmod(cfg.total_steps, cfg.workers)
    sizes = [base + (k < rem) for k in range(cfg.workers)]

    if cfg.workers == 1:
        shards = [collect_shard(cfg, 0, sizes[0], show_progress=True)]
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = [pool.submit(collect_shard, cfg, k, n, k == 0) for k, n in enumerate(sizes)]
            shards = [f.result() for f in futures]

    env = make_env(cfg.env, seed=0, **cfg.env_kwargs)
    meta = {
        "env": cfg.env,
        "num_actions": env.num_actions,
        "action_names": list(env.action_names),
        "movement_actions": list(env.movement_actions),
        "total_steps": cfg.total_steps,
        "num_episodes": sum(s["num_episodes"] for s in shards),
        "shards": [f"shard_{s['shard_id']:03d}" for s in shards],
        "config": asdict(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
