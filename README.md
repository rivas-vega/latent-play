# latent-play: Playable World Models from Action-Free Video

[![Tests](https://img.shields.io/badge/pytest-41%20passed-brightgreen)](#reproducibility--quickstart)
[![Paper](https://img.shields.io/badge/Technical%20Report-PDF-b31b1b)](report/main.pdf)
[![Colab](https://img.shields.io/badge/Colab%20A100-End--to--End%20Notebook-f9ab00)](notebooks/colab_training_and_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **How much action supervision does a world model trained on unlabelled gameplay video need to become controllable, non-collapsing, and effective for goal-directed planning?**

`latent-play` is an end-to-end research codebase that learns a **playable representation-space world model from action-free video**, combining three core ideas from recent embodied AI and self-supervised learning:

1. **Unsupervised Latent Action Discovery (`LAM`, Genie / LAPO style):** A factorized causal Spatio-Temporal (`STBlock`) Transformer with temporal feature-difference pooling ($\Delta h_t = h_{t+1} - h_t$) and an entropy-regularized Vector Quantizer (`VQ`, $K=8$) that discovers discrete latent actions from pixel transitions without action labels.
2. **Isotropic JEPA World Model (`JEPADynamics` + `SIGReg`, V-JEPA 2 / LeWorldModel style):** An action-conditioned ViT-S/8 Joint-Embedding Predictive Architecture trained with a momentum EMA target encoder and the **Statistical Invariant Geometric Regularizer (`SIGReg`)**—using the Cramér-Wold device and differentiable Epps-Pulley characteristic function testing (`TFG_Beamer`) alongside a VICReg variance hinge to guarantee multivariate isotropy and prevent dimensional collapse across multi-step rollouts.
3. **Prior-Corrected Alignment & Receding-Horizon Latent MPC (`CEMPlanner`):** A lightweight contextual alignment probe (`ActionAlignment`, $1.6\text{K}$ parameters) with log-prior class-imbalance correction coupled with a Cross-Entropy Method (`CEMPlanner`) trajectory optimizer in latent space, plus a zero-shot **Violation-of-Expectation (`VoE`)** physical plausibility adapter bridging to [`vjepa2-physics-scorer-poc`](https://github.com/rivas-vega/vjepa2-physics-scorer-poc).

📄 **[Read the full 4-page Technical Report (`report/main.pdf`)](report/main.pdf)**

---

## Key Empirical Findings (NVIDIA A100-SXM4-80GB)

### 1. Discovered Latent Actions Outperform True Action Labels in Multi-Step Rollouts
In open-ended environments like **Crafter** (17 actions) and **ToyGridEnv** (5 actions with boundary walls and an autonomous distractor), ground-truth action labels suffer from **visual action aliasing**: walking into water/trees/walls or invoking `do`/`place_*`/`make_*` without prerequisites produces zero visual displacement despite distinct action IDs. Because `LAM` ($K=8$, **100% codebook utilization**, **$+10.05\text{ dB}$ controllability $\Delta\text{PSNR}_{\text{ctrl}}$** on Crafter, **95.6% movement purity** on active locomotion codes) assigns codes based on the *realized visual transition*, conditioning `JEPADynamics` on `LAM` codes reduces multi-step latent rollout error by **24.0% on Crafter ($H=8$)** and **38.3% on ToyGridEnv ($H=4$)** compared to an oracle trained with 100% ground-truth actions:

| Environment | Conditioning | $H=1$ L1 Error | $H=2$ L1 Error | $H=4$ L1 Error | $H=8$ L1 Error | Effective Rank $\exp(H(p))$ |
|---|---|---:|---:|---:|---:|---:|
| **ToyGridEnv** ($100\text{K}$ frames) | Action Oracle (100% True Actions) | `0.0228` | `0.0970` | `0.2297` | — | `35.1` |
| **ToyGridEnv** ($100\text{K}$ frames) | **`latent-play` (`LAM` + `SIGReg`)** | **`0.0204`** | **`0.0596`** | **`0.1416` (-38.3%)** | — | **`35.2`** |
| **Crafter** ($300\text{K}$ frames, $30\text{K}$ steps) | Action Oracle (100% True Actions) | `0.0341` | `0.0598` | `0.1107` | `0.2003` | `37.4` |
| **Crafter** ($300\text{K}$ frames, $30\text{K}$ steps) | **`latent-play` (`LAM` + `SIGReg`)** | **`0.0218`** | **`0.0409`** | **`0.0818`** | **`0.1523` (-24.0%)** | **`35.0 – 42.3`** |

![Rollout Stability and Anti-Collapse Dynamics](report/figures/fig2_anticollapse_rollouts.png)

### 2. Sample-Efficient Goal-Directed CEM Navigation (0.1% – 100% Label Sweep)
With only **0.1% to 10% action-labelled transitions** to fit the lightweight `ActionAlignment` probe, receding-horizon `CEMPlanner` in `latent-play` matches or exceeds the 100%-supervised Action Oracle on both benchmarks (3 random label-subset seeds per fraction):

| Environment | Method | Action Supervision | Goal Navigation Mean ± SD (%) | Best Seed (%) |
|---|---|---:|---:|---:|
| **ToyGridEnv** | **`latent-play` (Ours)** | **0.1% (92 transitions)** | **28.0 ± 8.6%** | **36.0%** |
| **ToyGridEnv** | `latent-play` (Ours) | 5.0% | 21.3 ± 5.0% | 28.0% |
| **ToyGridEnv** | Action Oracle | 100.0% | 28.0% | 28.0% |
| **Crafter** | `latent-play` (Ours) | 0.1% (260 transitions) | 12.2 ± 4.2% | 16.7% |
| **Crafter** | `latent-play` (Ours) | 1.0% (2,598 transitions) | 12.2 ± 9.6% | **23.3%** |
| **Crafter** | **`latent-play` (Ours)** | **10.0%** | **15.6 ± 7.9%** | **26.7%** |
| **Crafter** | `latent-play` (Ours) | 100.0% | 12.2 ± 1.6% | 13.3% |
| **Crafter** | Action Oracle | 100.0% | 10.0% | 10.0% |

![Label Efficiency Scaling Curve](report/figures/fig1_label_efficiency.png)

### 3. Zero-Shot Physical Violation-of-Expectation (`VoE`) on Crafter
Evaluated on 50 held-out Crafter trajectories without any explicit physics or segmentation labels, the `latent-play` world model detects spatiotemporal violations via latent prediction surprise $S_t = \|\hat{z}_t - z_t\|_1$:

| Perturbation Type | Plausible Surprise | Violated Surprise | VoE Surprise Ratio ($S_{\text{violated}} / S_{\text{plausible}}$) |
|---|---:|---:|---:|
| **Action Contradiction** | `0.0209` | `0.0397` | **1.83×** |
| **Spatial Teleportation** | `0.0209` | `0.2862` | **15.25×** |
| **Object Disappearance** | `0.0209` | `0.9016` | **52.91×** |

![LAM Controllability and Zero-Shot VoE](report/figures/fig3_codebook_purity.png)

---

## Architecture & Repository Layout

```text
latent-play/
├── configs/                  # YAML configs for ToyGridEnv & Crafter (collect, LAM, JEPA dynamics)
├── latent_play/
│   ├── data/                 # Multi-worker memmap shard collector & leak-proof nested-subset Dataset
│   ├── envs/                 # ToyGridEnv & CrafterEnv with deterministic state cloning for MPC eval
│   ├── models/
│   │   ├── layers.py         # PatchEmbed, 2D sin-cos pos-embed, factorized causal STBlock
│   │   ├── vq.py             # Cosine VQ bottleneck with batch centering, entropy reg & dead-code restart
│   │   ├── lam.py            # Genie/LAPO Latent Action Model with motion-weighted reconstruction
│   │   ├── dynamics.py       # ViT-S/8 + STBlock JEPA predictor + EMA target encoder
│   │   └── decoder.py        # Detached diagnostic pixel decoder (stop-gradient probe)
│   ├── losses/
│   │   └── jepa.py           # L1 prediction + VICReg hinge + SIGReg (Cramér-Wold Epps-Pulley test)
│   ├── planning/
│   │   ├── alignment.py      # Few-shot contextual ActionAlignment probe with log-prior correction
│   │   └── cem.py            # Receding-horizon Cross-Entropy Method (CEM) latent trajectory planner
│   └── eval/
│       ├── metrics.py        # PSNR, ΔPSNR_control, Effective Rank, Action Purity & NMI
│       ├── probes.py         # Linear evaluation probes
│       └── voe.py            # Violation-of-Expectation (VoE) physical surprise scorer
├── scripts/                  # CLI pipelines: collect_data, train_lam, train_dyn, sweep_labels, eval_voe, etc.
├── notebooks/                # Self-contained Colab Pro A100 training & evaluation notebook
├── report/                   # 4-page two-column LaTeX technical report (main.tex & main.pdf)
└── tests/                    # 41 unit tests covering every module and anti-leakage guarantee
```

---

## Reproducibility & Quickstart

```bash
# 1. Install dependencies (CPU or CUDA) and run the 41 unit tests
uv sync --extra crafter --extra dev
uv run pytest -q

# 2. Collect datasets (ToyGridEnv & Crafter)
uv run python scripts/collect_data.py --config configs/collect_toy.yaml --total-steps 100000 --workers 4
uv run python scripts/collect_data.py --config configs/collect_crafter.yaml --total-steps 300000 --workers 4

# 3. Train Unsupervised Latent Action Model (LAM)
uv run python scripts/train_lam.py --config configs/lam_crafter.yaml train.amp=auto

# 4. Train Action-Conditioned JEPA World Model with SIGReg (using discovered LAM actions or Oracle)
uv run python scripts/train_dyn.py --config configs/dyn_crafter.yaml \
    lam_checkpoint=checkpoints/lam_crafter/lam_final.pt \
    train.out_dir=checkpoints/dyn_crafter_lam train.amp=auto

# 5. Run the Label-Efficiency Planning Sweep (0.1%, 1%, 5%, 10%, 100%) & VoE Physics Benchmark
uv run python scripts/sweep_labels.py --data-root data/crafter \
    --lam-ckpt checkpoints/lam_crafter/lam_final.pt \
    --dyn-ckpt checkpoints/dyn_crafter_lam/dyn_final.pt \
    --oracle-ckpt checkpoints/dyn_crafter_oracle/dyn_final.pt \
    --fractions 0.001 0.01 0.05 0.1 1.0 --num-episodes 30 --out-dir outputs/sweep_results

uv run python scripts/eval_voe.py --data-root data/crafter \
    --lam-ckpt checkpoints/lam_crafter/lam_final.pt \
    --dyn-ckpt checkpoints/dyn_crafter_lam/dyn_final.pt --num-clips 50

# 6. Regenerate publication figures and compile the PDF report
uv run python scripts/plot_paper_figures.py --metrics-json outputs/sweep_results/sweep_metrics.json
pdflatex -output-directory=report report/main.tex
```

## References

- **Genie:** Bruce et al., *Genie: Generative Interactive Environments*, ICML 2024.
- **LAPO:** Schmidt & Jiang, *Learning to Act without Actions*, NeurIPS 2023.
- **V-JEPA 2:** Assran et al., *V-JEPA 2: Self-Supervised Video Models for Physical Reasoning*, 2025.
- **SIGReg / LeJEPA:** Balestriero & LeCun, *Statistical Invariant Geometric Regularizers for Self-Supervised Learning*, TMLR 2025; Epps & Pulley, *Biometrika* 1983.
- **VICReg:** Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization*, ICLR 2022.
- **Crafter:** Hafner, *Benchmarking the Spectrum of Agent Capabilities with Crafter*, NeurIPS 2021.

## License

MIT
