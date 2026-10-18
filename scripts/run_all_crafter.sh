#!/usr/bin/env bash
set -euo pipefail

# End-to-end Crafter pipeline (300k frames, 30k steps per stage)
if [ ! -f data/crafter/meta.json ]; then
    uv run python scripts/collect_data.py --config configs/collect_crafter.yaml --total-steps 300000 --workers 4
fi
uv run python scripts/inspect_data.py --data data/crafter --out outputs/crafter_samples.gif

# Stage 1: Unsupervised Latent Action Model (0% action labels)
uv run python scripts/train_lam.py --config configs/lam_crafter.yaml train.compile=true train.amp=auto

# Stage 2: JEPA + SIGReg World Model (LAM-conditioned vs. 100% Action Oracle)
uv run python scripts/train_dyn.py --config configs/dyn_crafter.yaml lam_checkpoint=checkpoints/lam_crafter/lam_final.pt train.out_dir=checkpoints/dyn_crafter_lam train.amp=auto
uv run python scripts/train_dyn.py --config configs/dyn_crafter.yaml train.out_dir=checkpoints/dyn_crafter_oracle train.amp=auto

# Stages 3 & 4: Few-shot ActionAlignment sweep, Closed-Loop CEM planning, VoE & Figures
uv run python scripts/sweep_labels.py \
    --data-root data/crafter \
    --lam-ckpt checkpoints/lam_crafter/lam_final.pt \
    --dyn-ckpt checkpoints/dyn_crafter_lam/dyn_final.pt \
    --oracle-ckpt checkpoints/dyn_crafter_oracle/dyn_final.pt \
    --fractions 0.001 0.01 0.05 0.1 1.0 \
    --num-episodes 30 \
    --use-state-context \
    --out-dir outputs/sweep_results

uv run python scripts/eval_voe.py \
    --config configs/dyn_crafter.yaml \
    --checkpoint checkpoints/dyn_crafter_lam/dyn_final.pt \
    --lam-checkpoint checkpoints/lam_crafter/lam_final.pt \
    --out outputs/voe_metrics.json

uv run python scripts/eval_planning.py \
    --config configs/dyn_crafter.yaml \
    --checkpoint checkpoints/dyn_crafter_lam/dyn_final.pt \
    --align-ckpt outputs/sweep_results/alignment_f0.01_s1.pt \
    --target-dist 4 \
    --num-episodes 20

uv run python scripts/plot_paper_figures.py --metrics-json outputs/sweep_results/sweep_metrics.json --out-dir outputs/figures
