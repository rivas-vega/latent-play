"""Sanity-check a dataset: action histogram, episode stats and a GIF of sample windows.

    python scripts/inspect_data.py --data data/toy --out outputs/toy_samples.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from latent_play.data import VideoWindows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="outputs/samples.gif")
    p.add_argument("--num-windows", type=int, default=8)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--scale", type=int, default=3)
    args = p.parse_args()

    root = Path(args.data)
    meta = json.loads((root / "meta.json").read_text())
    actions = np.concatenate([np.load(root / s / "actions.npy") for s in meta["shards"]])
    episode = np.concatenate([np.load(root / s / "episode.npy") for s in meta["shards"]])
    valid = actions[actions >= 0]
    counts = np.bincount(valid, minlength=meta["num_actions"])
    _, ep_len = np.unique(episode, return_counts=True)

    print(f"env={meta['env']}  frames={len(actions)}  episodes={len(ep_len)}  transitions={len(valid)}")
    print(f"episode length: mean={ep_len.mean():.1f}  median={np.median(ep_len):.0f}  min={ep_len.min()}  max={ep_len.max()}")
    print("action distribution:")
    for name, c in zip(meta["action_names"], counts):
        print(f"  {name:>20s} {c / len(valid):6.1%}")

    ds = VideoWindows(root, num_frames=args.num_frames, split="all", return_true_actions=True)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(ds), size=min(args.num_windows, len(ds)), replace=False)
    clips = [ds[int(i)]["frames"].permute(0, 2, 3, 1).numpy() for i in picks]  # [T, H, W, 3]
    gif = np.concatenate(clips, axis=2)  # side by side
    gif = gif.repeat(args.scale, axis=1).repeat(args.scale, axis=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, list(gif), duration=0.15, loop=0)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
