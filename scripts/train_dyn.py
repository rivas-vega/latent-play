"""Train the Action-Conditioned JEPA Dynamics Model (and Visual Decoder).

Supports:
1. True action conditioning (upper bound benchmark)
2. Latent action conditioning (using frozen pretrained LAM)
3. Anti-collapse regularization: EMA target network + optional VICReg variance loss
4. Autoregressive multi-step rollout evaluation in representation space

Examples:
    # Train on Toy with true actions:
    python scripts/train_dyn.py --config configs/dyn_toy.yaml

    # Train on Crafter with latent actions from trained LAM:
    python scripts/train_dyn.py --config configs/dyn_crafter.yaml lam_checkpoint=checkpoints/lam_crafter/lam_final.pt
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
from latent_play.losses import effective_rank, jepa_loss
from latent_play.models import (
    DynamicsConfig,
    JEPADynamics,
    LAMConfig,
    LatentActionModel,
    VisualDecoder,
)
from latent_play.utils import (
    EMA,
    get_device,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from latent_play.utils.train import (
    JsonlLogger,
    autocast_ctx,
    cosine_lr,
    infinite,
    needs_grad_scaler,
    psnr,
)

DEFAULT_TRAIN = {
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "steps": 10000,
    "warmup": 500,
    "grad_clip": 1.0,
    "log_every": 50,
    "eval_every": 1000,
    "eval_batches": 20,
    "ckpt_every_min": 15.0,
    "num_workers": 4,
    "amp": "auto",
    "compile": False,
    "seed": 0,
    "out_dir": "checkpoints/dyn",
    "tau_start": 0.996,
    "tau_end": 1.0,
    "loss_type": "l1",
    "vicreg_var_weight": 0.0,
    "train_decoder": True,
    "decoder_lr": 3e-4,
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
def evaluate_rollouts(
    model: JEPADynamics,
    ema: EMA,
    decoder: VisualDecoder | None,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    lam: LatentActionModel | None,
    num_batches: int = 10,
    horizons: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[dict, np.ndarray | None]:
    model.eval()
    if decoder:
        decoder.eval()

    errors_by_h = {h: [] for h in horizons}
    sample_gif = None

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        raw_frames = frames_to_float(batch["frames"].to(device, non_blocking=True))  # [B, T, 3, H, W]

        # Determine actions: either latent inferred by LAM or ground truth actions
        if lam is not None:
            with autocast_ctx(device, amp):
                vq = lam.infer_actions(raw_frames)
                actions = vq["indices"]  # [B, T-1]
        else:
            actions = batch["action_true"].to(device)  # [B, T-1]

        with autocast_ctx(device, amp):
            # Target representations computed with EMA target encoder
            z_target = ema.target(raw_frames)  # [B, T, N, D]
            # Online representations
            z_online = model.encode(raw_frames)

            max_h = min(max(horizons), actions.shape[1] - 1)
            if max_h < 1:
                continue

            # Start rollout from frame 0
            z_init = z_online[:, 0:1]
            rollout_actions = actions[:, :max_h]
            z_rollout = model.rollout(z_init, rollout_actions)  # [B, max_h, N, D]

            for h in horizons:
                if h <= max_h:
                    # L1 error between rolled out representation and actual EMA target
                    err = torch.nn.functional.l1_loss(
                        z_rollout[:, h - 1], z_target[:, h]
                    ).item()
                    errors_by_h[h].append(err)

            if sample_gif is None and decoder is not None and max_h >= 4:
                # Decode rolled out imaginary frames and compare to ground truth
                pred_frames = decoder(z_rollout[:4, :4])  # [4, 4, 3, H, W]
                real_frames = raw_frames[:4, 1:5]
                # Construct side-by-side comparison (Real top, Imagined bottom)
                # [B, T, 3, H, W] -> [T, H, W*2, 3]
                B_vis, T_vis = pred_frames.shape[:2]
                real_np = ((real_frames[0].clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
                pred_np = ((pred_frames[0].clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
                combined = np.concatenate([real_np, pred_np], axis=1)  # stacked vertically
                sample_gif = combined

    metrics = {f"val/rollout_err_h{h}": float(np.mean(errors_by_h[h])) for h in horizons if errors_by_h[h]}
    model.train()
    if decoder:
        decoder.train()
    return metrics, sample_gif


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    tc, dc = cfg["train"], cfg["data"]
    seed_everything(tc["seed"])
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mcfg = DynamicsConfig(**cfg.get("model", {}))
    num_frames = mcfg.max_context
    train_ds = VideoWindows(
        dc["root"],
        num_frames=num_frames,
        split="train",
        val_fraction=dc.get("val_fraction", 0.05),
        label_fraction=dc.get("label_fraction", 1.0),
        return_true_actions=True,
    )
    val_ds = VideoWindows(
        dc["root"],
        num_frames=num_frames,
        split="val",
        val_fraction=dc.get("val_fraction", 0.05),
        return_true_actions=True,
    )

    loader_kw = dict(
        batch_size=tc["batch_size"],
        num_workers=tc["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=tc["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    print(f"device={device}  train windows={len(train_ds)}  val windows={len(val_ds)}")

    # Load LAM if latent action mode is active
    lam = None
    if cfg.get("lam_checkpoint"):
        lam_path = cfg["lam_checkpoint"]
        print(f"loading pretrained LAM from {lam_path}...")
        lam_ckpt = load_checkpoint(lam_path, device)
        lam_cfg = LAMConfig(**lam_ckpt["model_cfg"])
        lam = LatentActionModel(lam_cfg).to(device).eval()
        lam.load_state_dict(lam_ckpt["model"])
        for p in lam.parameters():
            p.requires_grad_(False)
        mcfg.num_actions = lam_cfg.num_codes
        print(f"Dynamics conditioned on {mcfg.num_actions} latent actions!")

    model = JEPADynamics(mcfg).to(device)
    ema = EMA(model.encoder, total_steps=tc["steps"], tau_start=tc["tau_start"], tau_end=tc["tau_end"])

    decoder = None
    dec_opt = None
    if tc["train_decoder"]:
        decoder = VisualDecoder(
            image_size=mcfg.image_size,
            patch=mcfg.patch,
            dim=mcfg.dim,
        ).to(device)
        dec_opt = torch.optim.AdamW(decoder.parameters(), lr=tc["decoder_lr"])

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=needs_grad_scaler(device, tc["amp"]))

    out = Path(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    step = 0
    if (ck := latest_checkpoint(out)) is not None:
        state = load_checkpoint(ck, device)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        if decoder and "decoder" in state:
            decoder.load_state_dict(state["decoder"])
        step = state["step"]
        print(f"resumed from {ck} at step {step}")

    logger = JsonlLogger(out / "log.jsonl")
    last_ckpt = time.time()
    batches = infinite(train_loader)
    t_last, n_last = time.time(), step

    while step < tc["steps"]:
        lr = cosine_lr(step, tc["lr"], tc["warmup"], tc["steps"])
        for g in opt.param_groups:
            g["lr"] = lr

        batch = next(batches)
        raw_frames = frames_to_float(batch["frames"].to(device, non_blocking=True))  # [B, T, 3, H, W]

        # Determine actions: discrete code from LAM or dataset action
        if lam is not None:
            with torch.no_grad():
                with autocast_ctx(device, tc["amp"]):
                    actions = lam.infer_actions(raw_frames)["indices"]  # [B, T-1]
        else:
            actions = batch["action_true"].to(device)  # [B, T-1]

        with autocast_ctx(device, tc["amp"]):
            # Target representation computed with EMA target encoder
            with torch.no_grad():
                target_z = ema.target(raw_frames[:, 1:])  # [B, T-1, N, D]

            # Online representation for context
            online_z = model.encode(raw_frames[:, :-1])  # [B, T-1, N, D]
            pred_z = model(online_z, actions)  # [B, T-1, N, D]

            loss_dict = jepa_loss(
                pred_z,
                target_z,
                loss_type=tc["loss_type"],
                vicreg_var_weight=tc["vicreg_var_weight"],
            )
            total_loss = loss_dict["loss"]

            # Train visual decoder with detached representations
            if decoder is not None:
                dec_pred = decoder(online_z.detach())
                dec_loss = decoder.loss(dec_pred, raw_frames[:, :-1])

        opt.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()

        if decoder is not None and dec_opt is not None:
            dec_opt.zero_grad(set_to_none=True)
            dec_loss.backward()
            dec_opt.step()

        # Update EMA target network
        tau = ema.update(model.encoder, step)
        step += 1

        if step % tc["log_every"] == 0:
            dt = time.time() - t_last
            log_record = {
                "step": step,
                "loss": total_loss.item(),
                "pred_loss": loss_dict["pred_loss"].item(),
                "var_loss": loss_dict["var_loss"].item(),
                "eff_rank": loss_dict["eff_rank"].item(),
                "tau": tau,
                "gnorm": gnorm.item(),
                "lr": lr,
                "it_s": (step - n_last) / dt,
            }
            if decoder is not None:
                log_record["dec_loss"] = dec_loss.item()
            logger.log(log_record)
            t_last, n_last = time.time(), step

        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            eval_metrics, sample_vis = evaluate_rollouts(
                model, ema, decoder, val_loader, device, tc["amp"], lam, num_batches=tc["eval_batches"]
            )
            logger.log({"step": step, **eval_metrics})
            if sample_vis is not None:
                imageio.mimsave(out / f"imagined_rollout_{step:07d}.gif", list(sample_vis), duration=0.2)

        if time.time() - last_ckpt > 60 * tc["ckpt_every_min"] or step == tc["steps"]:
            save_payload = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "model_cfg": model.config_dict(),
            }
            if decoder is not None:
                save_payload["decoder"] = decoder.state_dict()
            save_checkpoint(out, save_payload, step)
            last_ckpt = time.time()

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_cfg": model.config_dict(),
            "decoder": decoder.state_dict() if decoder else None,
        },
        out / "dyn_final.pt",
    )
    print(f"saved final dynamics checkpoint to {out / 'dyn_final.pt'}")


if __name__ == "__main__":
    main()
