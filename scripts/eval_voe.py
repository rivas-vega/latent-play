"""Evaluate Violation-of-Expectation (VoE) physical plausibility scores on a trained JEPA Dynamics model.

Bridges `latent-play` with `vjepa2-physics-scorer-poc`:
Measures surprise ratio R = Surprise(Violation) / Surprise(Plausible) at the violation onset for:
  - teleport (non-inertial position jump)
  - disappearance (sudden object permanence violation)
  - contradiction (temporal motion reversal contradicting actions)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from latent_play.data import VideoWindows, frames_to_float
from latent_play.eval import compute_voe_score
from latent_play.models import DynamicsConfig, JEPADynamics, LAMConfig, LatentActionModel
from latent_play.utils import get_device, load_checkpoint, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lam-checkpoint", default=None)
    ap.add_argument("--num-batches", type=int, default=20)
    ap.add_argument("--out", default="outputs/voe_metrics.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    cfg = yaml.safe_load(open(args.config))

    ckpt = load_checkpoint(args.checkpoint, device)
    mcfg = DynamicsConfig(**ckpt.get("model_cfg", cfg.get("model", {})))
    model = JEPADynamics(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    lam = None
    if args.lam_checkpoint:
        lam_ckpt = load_checkpoint(args.lam_checkpoint, device)
        lam_cfg = LAMConfig(**lam_ckpt["model_cfg"])
        lam = LatentActionModel(lam_cfg).to(device).eval()
        lam.load_state_dict(lam_ckpt["model"])

    val_ds = VideoWindows(
        cfg["data"]["root"],
        num_frames=mcfg.max_context,
        split="val",
        val_fraction=cfg["data"].get("val_fraction", 0.05),
        return_true_actions=True,
    )
    loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    summary = {}
    for vtype in ("teleport", "disappearance", "contradiction"):
        plausible_list, violated_list, ratio_list = [], [], []
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            frames = frames_to_float(batch["frames"].to(device))
            if lam is not None:
                with torch.no_grad():
                    actions = lam.infer_actions(frames)["indices"]
            else:
                actions = batch["action_true"].to(device)
            res = compute_voe_score(model, frames, actions, violation_type=vtype)
            plausible_list.append(res["surprise_plausible"])
            violated_list.append(res["surprise_violated"])
            ratio_list.append(res["voe_ratio"])

        summary[vtype] = {
            "surprise_plausible": float(np.mean(plausible_list)),
            "surprise_violated": float(np.mean(violated_list)),
            "voe_ratio": float(np.mean(ratio_list)),
        }
        print(
            f"VoE [{vtype:13s}] | Plausible: {summary[vtype]['surprise_plausible']:.4f} "
            f"| Violated: {summary[vtype]['surprise_violated']:.4f} "
            f"| Ratio: {summary[vtype]['voe_ratio']:.2f}x"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved VoE metrics to {out_path}")


if __name__ == "__main__":
    main()
