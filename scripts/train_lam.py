"""Train the Latent Action Model on action-free video.

    python scripts/train_lam.py --config configs/lam_toy.yaml
    python scripts/train_lam.py --config configs/lam_crafter.yaml train.batch_size=256 train.compile=true

Any ``section.key=value`` override after the config is parsed as YAML.
The training set is built with ``label_fraction=0``: the LAM never sees actions.
Ground-truth actions are used *only* by the validation probes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from latent_play.data import VideoWindows, frames_to_float
from latent_play.eval import code_action_stats, visible_actions
from latent_play.models import LAMConfig, LatentActionModel
from latent_play.utils import get_device, latest_checkpoint, load_checkpoint, save_checkpoint, seed_everything
from latent_play.utils.train import JsonlLogger, autocast_ctx, cosine_lr, infinite, needs_grad_scaler, psnr

DEFAULT_TRAIN = {
    "batch_size": 64, "lr": 3e-4, "weight_decay": 0.05, "steps": 5000, "warmup": 500, "grad_clip": 1.0,
    "log_every": 50, "eval_every": 500, "eval_batches": 20, "ckpt_every_min": 5.0, "num_workers": 4,
    "amp": "auto", "compile": False, "seed": 0, "out_dir": "checkpoints/lam",
}


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg["train"] = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


@torch.no_grad()
def evaluate(model: LatentActionModel, loader, device, amp, num_batches: int, collapse_blocked: bool) -> tuple[dict, np.ndarray]:
    model.eval()
    K = model.cfg.num_codes
    codes, acts, vis = [], [], []
    p_model, p_copy, p_rand = [], [], []
    sample = None
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        frames = frames_to_float(batch["frames"].to(device, non_blocking=True))
        with autocast_ctx(device, amp):
            vq = model.infer_actions(frames)
            past, target = frames[:, :-1], frames[:, 1:]
            pred = model.decode(past, vq["z_q"])
            rand_idx = torch.randint(0, K, vq["indices"].shape, device=device)
            pred_rand = model.decode_indices(past, rand_idx)
        p_model.append(psnr(pred, target).item())
        p_copy.append(psnr(past, target).item())
        p_rand.append(psnr(pred_rand, target).item())
        idx = vq["indices"].cpu().numpy()
        a = batch["action_true"].numpy()
        pos = batch["pos"].numpy()
        codes.append(idx.ravel())
        acts.append(a.ravel())
        vis.append(visible_actions(a, pos[:, :-1], pos[:, 1:]).ravel())
        if sample is None:
            sample = _sample_grid(past[:8, -1], target[:8, -1], pred[:8, -1], pred_rand[:8, -1])
    model.train()

    codes, acts, vis = map(np.concatenate, (codes, acts, vis))
    A = loader.dataset.num_actions
    stats = code_action_stats(codes, acts, K, A)
    metrics = {
        "val/psnr": float(np.mean(p_model)),
        "val/psnr_copy_last": float(np.mean(p_copy)),
        "val/delta_psnr_control": float(np.mean(p_model) - np.mean(p_rand)),
        "val/purity": stats["purity"],
        "val/nmi": stats["nmi"],
        "val/code_usage": int((np.bincount(codes, minlength=K) > 0).sum()),
    }
    if collapse_blocked:
        sv = code_action_stats(codes, vis, K, A)
        metrics["val/purity_visible"] = sv["purity"]
        metrics["val/nmi_visible"] = sv["nmi"]
    metrics["confusion"] = (sv if collapse_blocked else stats)["confusion"].tolist()
    return metrics, sample


def _sample_grid(*rows: torch.Tensor) -> np.ndarray:
    """Rows: last context frame | true next | predicted next | predicted with random action."""
    rows = [((r.float().clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy() for r in rows]
    return np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="Force training from step 0 by discarding existing checkpoints")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tc, dc = cfg["train"], cfg["data"]
    seed_everything(tc["seed"])
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mcfg = LAMConfig(**cfg.get("model", {}))
    train_ds = VideoWindows(dc["root"], mcfg.num_frames, "train", dc.get("val_fraction", 0.05), label_fraction=0.0)
    val_ds = VideoWindows(dc["root"], mcfg.num_frames, "val", dc.get("val_fraction", 0.05), return_true_actions=True)
    loader_kw = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"], pin_memory=device.type == "cuda",
                     drop_last=True, persistent_workers=tc["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=True, generator=torch.Generator().manual_seed(0), **loader_kw)
    print(f"device={device}  train windows={len(train_ds)}  val windows={len(val_ds)}")

    model = LatentActionModel(mcfg).to(device)
    print(f"LAM params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos_" not in n and "codebook" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "pos_" not in n and "codebook" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tc["lr"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=needs_grad_scaler(device, tc["amp"]))

    out = Path(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    step = 0
    if args.force:
        print(f"--force specified: discarding checkpoints in {out} and training from step 0.")
        for old_pt in out.glob("*.pt"):
            old_pt.unlink(missing_ok=True)
    elif (ck := latest_checkpoint(out)) is not None:
        state = load_checkpoint(ck, device)
        try:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            scaler.load_state_dict(state["scaler"])
            step = state["step"]
            print(f"resumed from {ck} at step {step}")
            if step >= tc["steps"]:
                print(f"Training already completed at step {step} >= {tc['steps']}. Pass --force to retrain from scratch.")
        except RuntimeError as e:
            print(f"Discarding incompatible checkpoint {ck} ({e}); starting fresh from step 0.")
            for old_pt in out.glob("*.pt"):
                old_pt.unlink(missing_ok=True)
            step = 0
    fwd = torch.compile(model) if tc["compile"] else model

    logger = JsonlLogger(out / "log.jsonl")
    last_ckpt = time.time()
    batches = infinite(train_loader)
    t_last, n_last = time.time(), step
    while step < tc["steps"]:
        lr = cosine_lr(step, tc["lr"], tc["warmup"], tc["steps"])
        for g in opt.param_groups:
            g["lr"] = lr
        frames = frames_to_float(next(batches)["frames"].to(device, non_blocking=True))
        with autocast_ctx(device, tc["amp"]):
            res = fwd(frames)
        opt.zero_grad(set_to_none=True)
        scaler.scale(res["loss"]).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % tc["log_every"] == 0:
            dt = time.time() - t_last
            logger.log({"step": step, "loss": res["loss"].item(), "recon": res["recon"].item(),
                        "vq": res["vq_loss"].item(), "ppl": res["perplexity"].item(), "gnorm": gnorm.item(),
                        "lr": lr, "it_s": (step - n_last) / dt})
            t_last, n_last = time.time(), step
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            metrics, grid = evaluate(model, val_loader, device, tc["amp"], tc["eval_batches"], dc.get("collapse_blocked", False))
            confusion = metrics.pop("confusion")
            logger.log({"step": step, **metrics})
            (out / "confusion.json").write_text(json.dumps({"step": step, "code_x_action": confusion}))
            imageio.imwrite(out / f"samples_{step:07d}.png", grid)
        if time.time() - last_ckpt > 60 * tc["ckpt_every_min"] or step == tc["steps"]:
            save_checkpoint(out, {"model": model.state_dict(), "opt": opt.state_dict(),
                                  "scaler": scaler.state_dict(), "model_cfg": model.config_dict()}, step)
            last_ckpt = time.time()

    torch.save({"model": model.state_dict(), "model_cfg": model.config_dict()}, out / "lam_final.pt")
    print(f"saved {out / 'lam_final.pt'}")


if __name__ == "__main__":
    main()
