# latent-play

**A playable world model learned from action-free video.**

> How much action supervision does a world model trained on unlabelled gameplay video need to become controllable and useful for planning?

`latent-play` infers *latent actions* from raw video (Genie / LAPO style), learns an action-conditioned **JEPA dynamics model** in representation space, aligns latent actions to real controls with a tiny labelled subset, and plans with **CEM/MPC** in latent space (V-JEPA 2-AC style).

> 🚧 Work in progress — week 1 of 8 (data pipeline). Results, GIFs and the report will land here.

## Status & Milestones

| Week | Milestone | Implementation | Status |
|---|---|---|---|
| 1 | Toy env, Crafter collection, leak-proof dataset | `envs/`, `data/collect.py`, `data/dataset.py` | ✅ Complete |
| 2 | Latent Action Model (Genie / LAPO style) | `models/lam.py`, `models/vq.py`, `models/layers.py`, `scripts/train_lam.py` | ✅ Complete |
| 3 | JEPA Dynamics (true actions) & anti-collapse losses | `models/dynamics.py`, `losses/jepa.py`, `scripts/train_dyn.py` | ✅ Complete |
| 4 | Dynamics with latent actions + Visual Decoder | `models/decoder.py`, `eval/probes.py` | ✅ Complete |
| 5 | CEM Planner + Goal-conditioned MPC evaluation | `planning/cem.py`, `planning/alignment.py`, `scripts/eval_planning.py` | ✅ Complete |
| 6 | Label-fraction sweep (headline scaling curve) | `scripts/sweep_labels.py` (0.1% to 100% supervision) | ✅ Ready |
| 7 | VoE physics surprise adapter | `eval/voe.py` (integration with `vjepa2-physics-scorer-poc`) | ✅ Ready |
| 8 | Colab Pro A100 notebook & write-up | `notebooks/colab_training_and_demo.ipynb` | ✅ Ready |

All components are covered by **37 passing unit tests** (`uv run pytest`).

## Quickstart

```bash
uv sync --extra crafter --extra dev     # local (CPU torch)
# on Kaggle / Colab:  pip install -e ".[crafter]"

uv run pytest -q
uv run python scripts/collect_data.py --config configs/collect_toy.yaml
uv run python scripts/collect_data.py --config configs/collect_crafter.yaml
uv run python scripts/inspect_data.py --data data/toy --out outputs/toy_samples.gif
```

## Data design

- **Two environments.** A pixel *toy grid* (agent + action-independent distractor + random background) to debug the pipeline in minutes, and **Crafter** (64×64, 17 actions) as the main benchmark.
- **Low-memory collection.** Workers write shards straight to disk via `np.memmap`.
- **No label leakage by construction.** The dataset returns `action = -1` for unlabelled transitions; ground truth is only exposed with `return_true_actions=True` for evaluation.
- **Nested label subsets.** Each transition has a fixed uniform draw, so the 0.1 % ⊂ 1 % ⊂ 10 % ⊂ 100 % subsets differ only in the amount of supervision.
- **Episode-level train/val split.**

## Repository layout

```
latent_play/
  envs/      toy_grid.py, crafter_env.py
  data/      collect.py, dataset.py
  models/    (week 2+) vit, st_transformer, vq, lam, dynamics, decoder
  planning/  (week 5)  cem, mpc_eval
  eval/      (week 3+) controllability, rollout error, probes, VoE
  utils/     ema.py, ckpt.py, misc.py
scripts/     collect_data.py, inspect_data.py
configs/     YAML configs
tests/       pytest suite
```

## References

Bruce et al. 2024 (Genie) · Schmidt & Jiang 2023 (LAPO) · Ye et al. 2024 (LAPA) · Assran et al. 2025 (V-JEPA 2) · Zhou et al. 2024 (DINO-WM) · Kanervisto et al. 2025 (WHAM/Muse) · Alonso et al. 2024 (DIAMOND) · Hafner 2021 (Crafter)

## License

MIT
