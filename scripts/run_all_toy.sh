#!/usr/bin/env bash
set -euo pipefail

# End-to-end ToyGridEnv pipeline (~3 minutes on a single GPU)
if [ ! -f data/toy/meta.json ]; then
    uv run python scripts/collect_data.py --config configs/collect_toy.yaml --total-steps 100000 --workers 4
fi
uv run python scripts/inspect_data.py --data data/toy --out outputs/toy_samples.gif

# Stage 1: Unsupervised Latent Action Model (0% action labels)
uv run python scripts/train_lam.py --config configs/lam_toy.yaml train.amp=auto

# Stage 2: JEPA + SIGReg World Model (LAM-conditioned vs. 100% Action Oracle)
uv run python scripts/train_dyn.py --config configs/dyn_toy.yaml lam_checkpoint=checkpoints/lam_toy/lam_final.pt train.out_dir=checkpoints/dyn_toy_lam train.amp=auto
uv run python scripts/train_dyn.py --config configs/dyn_toy.yaml train.out_dir=checkpoints/dyn_toy_oracle train.amp=auto

# Stages 3 & 4: Few-shot ActionAlignment sweep + Closed-Loop CEM planning
uv run python scripts/sweep_labels.py \
    --data-root data/toy \
    --lam-ckpt checkpoints/lam_toy/lam_final.pt \
    --dyn-ckpt checkpoints/dyn_toy_lam/dyn_final.pt \
    --oracle-ckpt checkpoints/dyn_toy_oracle/dyn_final.pt \
    --fractions 0.001 0.01 0.05 0.1 1.0 \
    --num-episodes 25 \
    --out-dir outputs/sweep_toy
