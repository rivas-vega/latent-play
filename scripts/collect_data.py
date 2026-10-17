"""Collect an action-labelled video dataset (labels are hidden later by the dataset).

Examples:
    python scripts/collect_data.py --config configs/collect_toy.yaml
    python scripts/collect_data.py --config configs/collect_crafter.yaml --total-steps 300000
"""

from __future__ import annotations

import argparse
import json
import time

import yaml

from latent_play.data.collect import CollectConfig, PolicyConfig, collect, default_workers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML config (CLI flags override it)")
    p.add_argument("--env", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--total-steps", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    raw = yaml.safe_load(open(args.config)) if args.config else {}
    for key in ("env", "out", "total_steps", "workers", "seed"):
        if getattr(args, key) is not None:
            raw[key] = getattr(args, key)
    raw.setdefault("workers", default_workers())
    cfg = CollectConfig(**{**raw, "policy": PolicyConfig(**raw.get("policy", {}))})

    t0 = time.time()
    meta = collect(cfg)
    dt = time.time() - t0
    print(json.dumps({k: meta[k] for k in ("env", "total_steps", "num_episodes", "shards")}, indent=2))
    print(f"done in {dt:.1f}s ({cfg.total_steps / dt:.0f} steps/s) -> {cfg.out}")


if __name__ == "__main__":
    main()
