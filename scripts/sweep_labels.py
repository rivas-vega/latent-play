"""Automated label-fraction sweep for the headline figure of the research paper.

Evaluates goal-planning success rate vs. label supervision fraction:
  Fractions: [0.001, 0.01, 0.05, 0.1, 1.0] (0.1%, 1%, 5%, 10%, 100%)
Compares:
  (A) latent-play (Ours): Unsupervised LAM + JEPA Dynamics + Action Alignment
  (B) Supervised Baseline: Trained solely on the sparse labelled transitions
  (C) Oracle Upper Bound: Dynamics trained with 100% true ground-truth actions

Outputs:
  - JSON results table with mean and std error across seeds
  - High-resolution SVG / PNG publication plot (Matplotlib)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from latent_play.data import VideoWindows
from latent_play.envs import make_env
from latent_play.eval import visible_actions
from latent_play.models import DynamicsConfig, JEPADynamics, LAMConfig, LatentActionModel
from latent_play.planning import ActionAlignment, CEMPlanner
from latent_play.utils import get_device, load_checkpoint, seed_everything
from scripts.eval_planning import run_goal_episode


@torch.no_grad()
def precompute_dataset_features(
    lam: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    dyn_model: JEPADynamics | None = None,
    use_state_context: bool = False,
    only_valid_labels: bool = False,
    collapse_blocked: bool = False,
) -> dict[str, torch.Tensor]:
    """Precomputes and caches discrete LAM codes, actions, and visual embeddings once."""
    lam.eval()
    if dyn_model is not None:
        dyn_model.eval()
    use_ctx = use_state_context and dyn_model is not None

    all_codes = []
    all_actions = []
    all_z = []

    for batch in loader:
        target_actions = batch["action_true"] if "action_true" in batch else batch["action"]
        if collapse_blocked and "pos" in batch:
            a_np = target_actions.numpy()
            pos_np = batch["pos"].numpy()
            vis_np = visible_actions(a_np, pos_np[:, :-1], pos_np[:, 1:])
            target_actions = torch.from_numpy(vis_np)

        if only_valid_labels:
            valid = target_actions >= 0
            if not valid.any():
                continue
        else:
            valid = None

        frames = batch["frames"].float().div_(127.5).sub_(1.0).to(device)
        codes = lam.infer_actions(frames)["indices"].cpu()
        actions = target_actions.cpu()

        if use_ctx:
            z = dyn_model.encode(frames[:, :-1]).cpu()
            B, T_m1 = codes.shape
            z_flat = z.reshape(B * T_m1, *z.shape[2:])

        if valid is not None:
            v_flat = valid.cpu().flatten()
            all_codes.append(codes.flatten()[v_flat])
            all_actions.append(actions.flatten()[v_flat])
            if use_ctx:
                all_z.append(z_flat[v_flat])
        else:
            all_codes.append(codes.flatten())
            all_actions.append(actions.flatten())
            if use_ctx:
                all_z.append(z_flat)

    res = {
        "codes": torch.cat(all_codes, dim=0) if all_codes else torch.zeros(0, dtype=torch.long),
        "actions": torch.cat(all_actions, dim=0) if all_actions else torch.zeros(0, dtype=torch.long),
    }
    if use_ctx and all_z:
        res["z"] = torch.cat(all_z, dim=0)
    return res


def fit_alignment_probe(
    codes: torch.Tensor,
    actions: torch.Tensor,
    z: torch.Tensor | None,
    num_codes: int,
    num_actions: int,
    dim: int,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-2,
    batch_size: int = 256,
    use_state_context: bool = False,
) -> ActionAlignment:
    """Trains the action alignment classifier on pre-cached tensors using standard minibatches."""
    align = ActionAlignment(
        num_codes=num_codes,
        num_actions=num_actions,
        dim=dim,
        use_state_context=use_state_context,
    ).to(device)
    if len(actions) == 0:
        return align

    opt = torch.optim.Adam(align.parameters(), lr=lr)
    codes = codes.to(device)
    actions = actions.to(device)
    z = z.to(device) if z is not None else None

    counts = torch.bincount(actions, minlength=num_actions).float()
    prior = (counts + 1.0) / (counts.sum() + num_actions)
    align.action_prior.copy_(prior)

    dataset = (
        torch.utils.data.TensorDataset(codes, actions)
        if z is None
        else torch.utils.data.TensorDataset(codes, actions, z)
    )
    effective_bs = min(batch_size, max(16, len(actions)))
    loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True)

    align.train()
    for _ in range(epochs):
        for b_data in loader:
            b_codes, b_actions = b_data[0], b_data[1]
            b_z = b_data[2] if z is not None else None
            logits = align(b_codes, z=b_z)
            loss = align.loss(logits, b_actions, codes=b_codes)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return align.eval()


def train_alignment_probe(
    lam: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    dyn_model: JEPADynamics | None = None,
    epochs: int = 10,
    lr: float = 1e-2,
    batch_size: int = 256,
    use_state_context: bool = False,
    collapse_blocked: bool = False,
) -> ActionAlignment:
    """Trains the action alignment classifier on only the labelled subset (where action >= 0)."""
    lam.eval()
    if dyn_model is not None:
        dyn_model.eval()
    K = lam.cfg.num_codes
    A = loader.dataset.num_actions

    dim = dyn_model.cfg.dim if dyn_model is not None else 128
    use_ctx = use_state_context and dyn_model is not None

    cached = precompute_dataset_features(
        lam,
        loader,
        device,
        dyn_model=dyn_model,
        use_state_context=use_ctx,
        only_valid_labels=True,
        collapse_blocked=collapse_blocked,
    )
    return fit_alignment_probe(
        cached["codes"],
        cached["actions"],
        cached.get("z", None),
        num_codes=K,
        num_actions=A,
        dim=dim,
        device=device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_state_context=use_ctx,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/toy")
    ap.add_argument("--lam-ckpt", default=None)
    ap.add_argument("--dyn-ckpt", default=None, help="JEPA dynamics checkpoint trained on LAM latent codes")
    ap.add_argument("--oracle-ckpt", default=None, help="JEPA dynamics checkpoint trained on 100%% true actions")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.001, 0.01, 0.05, 0.1, 1.0])
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--target-dist", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out-dir", default="outputs/sweep_results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--use-state-context", action="store_true", default=False, help="Condition alignment probe on visual state context")
    ap.add_argument("--cem-samples", type=int, default=128, help="Number of candidate action sequences sampled per CEM iteration")
    ap.add_argument("--cem-elites", type=int, default=16, help="Number of top elite sequences selected per CEM iteration")
    ap.add_argument("--cem-iters", type=int, default=4, help="Number of CEM optimization iterations")
    ap.add_argument("--cem-horizon", type=int, default=6, help="Planning horizon in steps")
    ap.add_argument("--collapse-blocked", action="store_true", default=None, help="Map blocked movement actions to NOOP (default: True for toy, False otherwise)")
    ap.add_argument("--distractor", action="store_true", default=None, help="Enable distractor in toy env (default: False for planning eval)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    env_name = Path(args.data_root).name

    if args.collapse_blocked is None:
        args.collapse_blocked = (env_name == "toy")
    if args.distractor is None:
        args.distractor = False if env_name == "toy" else True
    env_kwargs = {"distractor": args.distractor} if env_name == "toy" else {}

    print(f"Running label-fraction sweep on fractions: {args.fractions}")
    print(f"Seeds: {args.seeds} | Use State Context: {args.use_state_context} | Collapse Blocked: {args.collapse_blocked}")

    results = {
        "fractions": args.fractions,
        "seeds": args.seeds,
        "alignment_accuracy": {str(f): [] for f in args.fractions},
        "planning_success": {str(f): [] for f in args.fractions},
        "ours": {"mean": [], "std": []},
    }

    dyn_model = None
    if args.dyn_ckpt:
        dyn_data = load_checkpoint(args.dyn_ckpt, device)
        dyn_cfg = DynamicsConfig(**dyn_data["model_cfg"])
        dyn_model = JEPADynamics(dyn_cfg).to(device).eval()
        dyn_model.load_state_dict(dyn_data["model"])

    if args.oracle_ckpt:
        ora_data = load_checkpoint(args.oracle_ckpt, device)
        ora_cfg = DynamicsConfig(**ora_data["model_cfg"])
        ora_model = JEPADynamics(ora_cfg).to(device).eval()
        ora_model.load_state_dict(ora_data["model"])
        ora_planner = CEMPlanner(
            dynamics=ora_model,
            horizon=args.cem_horizon,
            num_samples=args.cem_samples,
            num_elites=args.cem_elites,
            iterations=args.cem_iters,
        )
        env = make_env(env_name, seed=100, **env_kwargs)
        ora_sr = np.mean(
            [
                run_goal_episode(env, ora_model, ora_planner, device, args.target_dist, args.max_steps)["success"]
                for _ in range(args.num_episodes)
            ]
        )
        results["oracle"] = float(ora_sr * 100.0)
        print(f"Oracle (100% True Actions) Goal Planning Success: {results['oracle']:.1f}%")

    # If LAM checkpoint provided, evaluate alignment probe + CEM planning scaling
    if args.lam_ckpt:
        lam_data = load_checkpoint(args.lam_ckpt, device)
        lam_cfg = LAMConfig(**lam_data["model_cfg"])
        lam = LatentActionModel(lam_cfg).to(device).eval()
        lam.load_state_dict(lam_data["model"])

        val_ds = VideoWindows(args.data_root, num_frames=lam_cfg.num_frames, split="val", return_true_actions=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        print("Pre-caching validation representations...")
        val_cached = precompute_dataset_features(
            lam,
            val_loader,
            device,
            dyn_model=dyn_model,
            use_state_context=args.use_state_context,
            collapse_blocked=args.collapse_blocked,
        )
        val_codes_all = val_cached["codes"].to(device)
        val_actions_all = val_cached["actions"].to(device)
        val_z_all = val_cached["z"].to(device) if "z" in val_cached else None

        full_train_ds = VideoWindows(
            args.data_root,
            num_frames=lam_cfg.num_frames,
            split="train",
            label_fraction=1.0,
            return_true_actions=True,
        )
        full_train_loader = DataLoader(full_train_ds, batch_size=64, shuffle=False)
        print("Pre-caching full training representations...")
        train_cached = precompute_dataset_features(
            lam,
            full_train_loader,
            device,
            dyn_model=dyn_model,
            use_state_context=args.use_state_context,
            collapse_blocked=args.collapse_blocked,
        )
        train_codes_all = train_cached["codes"]
        train_actions_all = train_cached["actions"]
        train_z_all = train_cached.get("z", None)
        N_train = len(train_actions_all)
        align_dim = dyn_model.cfg.dim if dyn_model is not None else 128
        use_ctx = args.use_state_context and dyn_model is not None

        for frac in args.fractions:
            frac_srs = []
            for seed in args.seeds:
                seed_everything(seed)
                if frac >= 1.0:
                    sub_codes = train_codes_all
                    sub_actions = train_actions_all
                    sub_z = train_z_all
                else:
                    rng = np.random.default_rng(seed)
                    mask = rng.random(N_train) < frac
                    if not mask.any():
                        mask[rng.integers(0, N_train)] = True
                    sub_codes = train_codes_all[mask]
                    sub_actions = train_actions_all[mask]
                    sub_z = train_z_all[mask] if train_z_all is not None else None

                align = fit_alignment_probe(
                    sub_codes,
                    sub_actions,
                    sub_z,
                    num_codes=lam_cfg.num_codes,
                    num_actions=val_ds.num_actions,
                    dim=align_dim,
                    device=device,
                    use_state_context=use_ctx,
                )
                torch.save(
                    {
                        "alignment": align.state_dict(),
                        "num_codes": lam_cfg.num_codes,
                        "num_actions": val_ds.num_actions,
                        "fraction": frac,
                        "seed": seed,
                        "use_state_context": align.use_state_context,
                        "dim": align.dim,
                        "embed_dim": align.dim,
                        "action_prior": align.action_prior.cpu(),
                    },
                    out / f"alignment_f{frac}_s{seed}.pt",
                )

                # Evaluate accuracy on validation ground-truth actions in one shot
                with torch.no_grad():
                    preds = align.predict_action(val_codes_all, z=val_z_all, prior_correction=0.0)
                acc = (preds == val_actions_all).float().mean().item()
                results["alignment_accuracy"][str(frac)].append(acc)

                if dyn_model is not None:
                    planner = CEMPlanner(
                        dynamics=dyn_model,
                        horizon=args.cem_horizon,
                        num_samples=args.cem_samples,
                        num_elites=args.cem_elites,
                        iterations=args.cem_iters,
                        alignment=align,
                    )
                    env = make_env(env_name, seed=seed + 100, **env_kwargs)
                    ep_succ = [
                        run_goal_episode(env, dyn_model, planner, device, args.target_dist, args.max_steps)["success"]
                        for _ in range(args.num_episodes)
                    ]
                    sr = float(np.mean(ep_succ) * 100.0)
                    frac_srs.append(sr)
                    results["planning_success"][str(frac)].append(sr)
                    print(
                        f"Fraction {frac:6.1%} | Seed {seed} | Probe Acc: {acc:6.1%} | Planning Success: {sr:5.1f}%"
                    )
                else:
                    print(f"Fraction {frac:6.1%} | Seed {seed} | Action Alignment Acc: {acc:6.1%}")

            if frac_srs:
                results["ours"]["mean"].append(float(np.mean(frac_srs)))
                results["ours"]["std"].append(float(np.std(frac_srs)))

    (out / "sweep_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved sweep results to {out / 'sweep_metrics.json'}")


if __name__ == "__main__":
    main()
