"""Systematic ablation script for CEM planning cost metrics and planning horizons.

Ablation 1 (Cost Metric):
- spatial_conv: 2D Gaussian convolution over patch token grid
- l1: direct token-wise L1 distance
- cosine: token-wise cosine distance (1 - cosine_similarity)
- pooled: spatially-averaged token representation distance

Ablation 2 (Planning Horizon H):
- H in {2, 3, 4, 6}

Outputs:
- JSON summary table at <out-dir>/ablation_results.json
- Formatted markdown table printed to stdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from latent_play.envs import make_env
from latent_play.models import DynamicsConfig, JEPADynamics
from latent_play.planning import ActionAlignment, CEMPlanner
from latent_play.utils import get_device, load_checkpoint, seed_everything
from scripts.eval_planning import run_goal_episode


def evaluate_planner_config(
    env,
    model: JEPADynamics,
    device: torch.device,
    alignment: ActionAlignment | None,
    horizon: int,
    cost_metric: str,
    num_episodes: int = 10,
    target_dist: int = 4,
    max_steps: int = 12,
    num_samples: int = 128,
    num_elites: int = 16,
    iterations: int = 4,
) -> dict:
    planner = CEMPlanner(
        dynamics=model,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        iterations=iterations,
        alignment=alignment,
        cost_metric=cost_metric,
    )
    successes = []
    step_counts = []
    for _ in range(num_episodes):
        res = run_goal_episode(env, model, planner, device, target_dist=target_dist, max_steps=max_steps)
        successes.append(res["success"])
        if res["success"]:
            step_counts.append(res["steps"])

    sr = float(np.mean(successes) * 100.0)
    avg_s = float(np.mean(step_counts)) if step_counts else float("nan")
    return {"success_rate": sr, "avg_steps": avg_s, "num_episodes": num_episodes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alignment-checkpoint", default=None)
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--target-dist", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out-dir", default="outputs/ablations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    cfg = yaml.safe_load(open(args.config))

    ckpt = load_checkpoint(args.checkpoint, device)
    mcfg = DynamicsConfig(**ckpt.get("model_cfg", cfg.get("model", {})))
    model = JEPADynamics(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt and ckpt["ema"] is not None:
        model.encoder.load_state_dict(ckpt["ema"])

    alignment = None
    if args.alignment_checkpoint:
        align_ckpt = load_checkpoint(args.alignment_checkpoint, device)
        alignment = ActionAlignment(
            num_codes=align_ckpt["num_codes"],
            num_actions=align_ckpt["num_actions"],
            use_state_context=align_ckpt.get("use_state_context", False),
            embed_dim=align_ckpt.get("embed_dim", 192),
        ).to(device)
        alignment.load_state_dict(align_ckpt["alignment"])
        if "action_prior" in align_ckpt:
            alignment.action_prior.copy_(align_ckpt["action_prior"].to(device))
        alignment.eval()

    env_name = cfg["data"]["root"].split("/")[-1]
    env_kwargs = {"distractor": False} if env_name == "toy" else {}
    env = make_env(env_name, seed=args.seed, **env_kwargs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {"cost_metric_ablation": {}, "horizon_ablation": {}}

    print("=" * 60)
    print("ABLATION 1: CEM COST METRIC (fixing Horizon H=4)")
    print("=" * 60)
    cost_metrics = ["spatial_conv", "l1", "cosine", "pooled"]
    best_cost_metric = "spatial_conv"
    best_sr = -1.0

    for cm in cost_metrics:
        res = evaluate_planner_config(
            env, model, device, alignment,
            horizon=4,
            cost_metric=cm,
            num_episodes=args.num_episodes,
            target_dist=args.target_dist,
            max_steps=args.max_steps,
        )
        results["cost_metric_ablation"][cm] = res
        print(f"Cost Metric: {cm:<15} | Success: {res['success_rate']:5.1f}% | Avg Steps: {res['avg_steps']:.2f}")
        if res["success_rate"] > best_sr:
            best_sr = res["success_rate"]
            best_cost_metric = cm

    print("\n" + "=" * 60)
    print(f"ABLATION 2: PLANNING HORIZON H (using best cost metric: {best_cost_metric})")
    print("=" * 60)
    horizons = [2, 3, 4, 6]
    for h in horizons:
        res = evaluate_planner_config(
            env, model, device, alignment,
            horizon=h,
            cost_metric=best_cost_metric,
            num_episodes=args.num_episodes,
            target_dist=args.target_dist,
            max_steps=args.max_steps,
        )
        results["horizon_ablation"][str(h)] = res
        print(f"Horizon H: {h:<16} | Success: {res['success_rate']:5.1f}% | Avg Steps: {res['avg_steps']:.2f}")

    json_path = out / "ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved ablation results to {json_path}")


if __name__ == "__main__":
    main()
