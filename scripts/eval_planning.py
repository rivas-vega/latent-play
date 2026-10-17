"""Closed-loop MPC planning evaluation on goal-conditioned tasks.

Evaluates how effectively the learned JEPA world model plans to reach visual goals
in the environment:
1. Initialize environment at x_0
2. Roll out environment for N_target steps to acquire a physically guaranteed reachable goal x_goal
3. Reset environment to x_0
4. Execute receding-horizon CEM planning in representation space
5. Measure success rate (% of goals reached within budget), steps to goal, and generate GIF

Examples:
    python scripts/eval_planning.py --config configs/dyn_toy.yaml --checkpoint checkpoints/dyn_toy/dyn_final.pt --num-episodes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from latent_play.data.dataset import frames_to_float
from latent_play.envs import make_env
from latent_play.models import DynamicsConfig, JEPADynamics
from latent_play.planning import ActionAlignment, CEMPlanner
from latent_play.utils import get_device, load_checkpoint, seed_everything


def run_goal_episode(
    env,
    model: JEPADynamics,
    planner: CEMPlanner,
    device: torch.device,
    target_dist: int = 6,
    max_steps: int = 15,
) -> dict:
    """Executes a single goal-conditioned evaluation episode."""
    obs_0, info_0 = env.reset()
    start_pos = info_0["pos"].copy()
    init_state = env.clone_state()

    # Generate reachable goal by taking movement actions from current state
    goal_obs = obs_0
    goal_pos = start_pos.copy()
    actions_taken = []
    for _ in range(target_dist):
        a = int(np.random.choice(env.movement_actions))
        goal_obs, _, _, g_info = env.step(a)
        goal_pos = g_info["pos"].copy()
        actions_taken.append(a)

    # Ensure goal is not identical to start position
    extra_tries = 0
    while np.array_equal(goal_pos, start_pos) and extra_tries < 10:
        a = int(np.random.choice(env.movement_actions))
        goal_obs, _, _, g_info = env.step(a)
        goal_pos = g_info["pos"].copy()
        actions_taken.append(a)
        extra_tries += 1

    # Encode goal into representation space: [1, N, D]
    t_goal = frames_to_float(torch.from_numpy(goal_obs).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)).to(device)
    z_goal = model.encode(t_goal)[:, 0]

    # Restore environment back to the exact starting state x_0 (same world, background, and entities)
    obs, info = env.restore_state(init_state)

    frames_history = [obs]
    success = False
    step_count = 0

    for step in range(max_steps):
        t_curr = frames_to_float(torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)).to(device)
        z_curr = model.encode(t_curr)[:, 0]

        # Check if already at goal position
        if np.array_equal(info["pos"], goal_pos):
            success = True
            break

        # Plan next action via CEM in representation space
        action, _ = planner.plan(z_curr, z_goal)
        obs, _, done, info = env.step(action)
        frames_history.append(obs)
        step_count += 1

        if np.array_equal(info["pos"], goal_pos):
            success = True
            break
        if done:
            break

    return {
        "success": success,
        "steps": step_count,
        "target_dist": target_dist,
        "start_pos": start_pos.tolist(),
        "goal_pos": goal_pos.tolist(),
        "frames": frames_history,
        "goal_frame": goal_obs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alignment-checkpoint", default=None)
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--target-dist", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--num-elites", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--out-dir", default="outputs/eval_planning")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    cfg = yaml.safe_load(open(args.config))

    ckpt = load_checkpoint(args.checkpoint, device)
    mcfg = DynamicsConfig(**ckpt.get("model_cfg", cfg.get("model", {})))
    model = JEPADynamics(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded dynamics from {args.checkpoint} (num_actions={mcfg.num_actions})")

    alignment = None
    if args.alignment_checkpoint:
        align_ckpt = load_checkpoint(args.alignment_checkpoint, device)
        alignment = ActionAlignment(
            num_codes=align_ckpt["num_codes"],
            num_actions=align_ckpt["num_actions"],
        ).to(device)
        alignment.load_state_dict(align_ckpt["alignment"])
        alignment.eval()

    planner = CEMPlanner(
        dynamics=model,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        iterations=args.iterations,
        alignment=alignment,
    )

    env = make_env(cfg["data"]["root"].split("/")[-1], seed=args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    successes = []
    step_counts = []

    for ep in range(args.num_episodes):
        res = run_goal_episode(
            env, model, planner, device, target_dist=args.target_dist, max_steps=args.max_steps
        )
        successes.append(res["success"])
        if res["success"]:
            step_counts.append(res["steps"])
        print(f"Episode {ep+1}/{args.num_episodes}: success={res['success']}, steps={res['steps']}")

        # Save GIF of first episode
        if ep == 0:
            # Render video with goal banner side-by-side
            goal_frame = res["goal_frame"]
            combined_frames = []
            for f in res["frames"]:
                # Stack current frame and goal frame side-by-side
                row = np.concatenate([f, goal_frame], axis=1)
                # Scale up for clean viewing
                row = row.repeat(3, axis=0).repeat(3, axis=1)
                combined_frames.append(row)
            imageio.mimsave(out / "planned_trajectory.gif", combined_frames, duration=0.25)
            print(f"saved trajectory animation to {out / 'planned_trajectory.gif'}")

    sr = np.mean(successes)
    avg_steps = np.mean(step_counts) if step_counts else float("nan")
    print(f"\nResults over {args.num_episodes} episodes:")
    print(f"  Goal Success Rate: {sr:6.1%}")
    print(f"  Avg Steps (Success): {avg_steps:.2f}")


if __name__ == "__main__":
    main()
