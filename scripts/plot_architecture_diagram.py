"""Generate a didactic, publication-grade 3-panel architecture diagram for latent-play.

Outputs:
  - outputs/figures/fig0_architecture.png
  - outputs/figures/fig0_architecture.pdf
  - report/figures/fig0_architecture.pdf
  - report/figures/fig0_architecture.png
"""
from __future__ import annotations

import shutil
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str = "#f8fafc",
    edgecolor: str = "#334155",
    title_color: str = "#0f172a",
    body_color: str = "#334155",
    lw: float = 1.6,
    fontsize_title: float = 9.3,
    fontsize_body: float = 8.1,
    linestyle: str = "-",
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(box)
    if title and body:
        ax.text(
            x + w / 2,
            y + h * 0.72,
            title,
            ha="center",
            va="center",
            fontsize=fontsize_title,
            fontweight="bold",
            color=title_color,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=fontsize_body,
            color=body_color,
            zorder=3,
        )
    elif title:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=fontsize_title,
            fontweight="bold",
            color=title_color,
            zorder=3,
        )


def draw_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = "",
    color: str = "#334155",
    lw: float = 1.7,
    linestyle: str = "-",
    rad: float = 0.0,
    label_offset: tuple[float, float] = (0.0, 0.15),
    fontsize: float = 8.0,
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=6,head_width=3.8",
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        zorder=4,
    )
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=5,
        )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
        }
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(14.8, 12.2), gridspec_kw={"height_ratios": [1.0, 1.65, 1.28]}
    )
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.22, left=0.02, right=0.98, top=0.94, bottom=0.02)

    fig.suptitle(
        "latent-play Architecture: Embeddings, Unsupervised LAM, JEPA + SIGReg World Model & Dual Action Loops",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    # =========================================================================
    # PANEL 1: STAGE 1 — WHAT IS EMBEDDED & LAM (UNSUPERVISED ACTION DISCOVERY)
    # =========================================================================
    ax1 = axes[0]
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 3.2)
    ax1.axis("off")

    panel1_bg = FancyBboxPatch(
        (0.1, 0.08),
        13.8,
        3.0,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#fffbeb",
        edgecolor="#d97706",
        linewidth=1.5,
        zorder=1,
    )
    ax1.add_patch(panel1_bg)
    ax1.text(
        0.35,
        2.82,
        "STAGE 1: What is Embedded & How the Latent Action Model (LAM) Discovers Discrete Actions (100% Label-Free Video)",
        fontsize=10.5,
        fontweight="bold",
        color="#92400e",
    )

    draw_box(
        ax1,
        0.35,
        0.85,
        2.15,
        1.65,
        "Consecutive Frames\n$x_t, x_{t+1} \\in [0,1]^{3\\times 64\\times 64}$",
        "Raw RGB video clip\n(No action labels $a_t$)",
        facecolor="#fef3c7",
        edgecolor="#d97706",
    )
    draw_arrow(ax1, 2.50, 1.67, 3.05, 1.67, "")

    draw_box(
        ax1,
        3.05,
        0.85,
        2.45,
        1.65,
        "PatchEmbed (8x8) + STBlocks\n(Inverse Dynamics $E_{\\mathrm{IDM}}$)",
        "Conv2d(3->D, k=8, s=8) -> 64 patches\nTemporal diff $\\Delta h_t = h_{t+1} - h_t$\n+ Spatial Motion Attention Pool",
        facecolor="#ffffff",
        edgecolor="#b45309",
    )
    draw_arrow(ax1, 5.50, 1.67, 6.15, 1.67, "$u_t \\in \\mathbb{R}^{16}$")

    draw_box(
        ax1,
        6.15,
        0.85,
        2.45,
        1.65,
        "Cosine Vector Quantizer\nCodebook $C \\in \\mathbb{R}^{8\\times 16}$",
        "$k_t = \\arg\\min_k \\|\\hat{u}_t - \\hat{c}_k\\|_2$\nOutputs discrete code $k_t \\in \\{0..7\\}$\n+ Batch Entropy Reg (8/8 active)",
        facecolor="#f3e8ff",
        edgecolor="#7e22ce",
        title_color="#581c87",
    )
    draw_arrow(ax1, 8.60, 1.67, 9.25, 1.67, "$z_q(k_t) + x_t$")

    draw_box(
        ax1,
        9.25,
        0.85,
        2.30,
        1.65,
        "Forward Decoder ($D_{\\mathrm{FDM}}$)\nResidual Pixel Head",
        "Takes current $x_t$ + code $z_q(k_t)$\nPredicts $\\hat{x}_{t+1} = x_t + \\Delta\\hat{x}_t$",
        facecolor="#ffffff",
        edgecolor="#b45309",
    )
    draw_arrow(ax1, 11.55, 1.67, 11.95, 1.67, "")

    draw_box(
        ax1,
        11.95,
        0.85,
        1.75,
        1.65,
        "Stage 1 Loss\n$\\mathcal{L}_{\\mathrm{LAM}}$",
        "Motion-Weighted MSE\n$w = 1 + 5|x_{t+1}-x_t|$\n$+ \\mathcal{L}_{\\mathrm{VQ}} + 0.1\\mathcal{L}_{\\mathrm{ent}}$",
        facecolor="#ffe4e6",
        edgecolor="#e11d48",
        title_color="#9f1239",
    )

    ax1.text(
        7.0,
        0.35,
        "Learnable in Stage 1: Inverse Encoder $E_{\\mathrm{IDM}}$, Codebook $C$ ($K=8$), Forward Decoder $D_{\\mathrm{FDM}}$ (1.68M params). After Stage 1, LAM is FROZEN to label all video transitions with $k_t \\in \\{0..7\\}$.",
        ha="center",
        va="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#78350f",
    )

    # =========================================================================
    # PANEL 2: STAGE 2 — JEPA WORLD MODEL TRAINING & WHERE SIGREG IS APPLIED
    # Clean non-crossing layout:
    #   Top row:    [SIGReg + VICReg Box (above Online ViT)]   [Action Embed E_a (above Predictor)]
    #   Middle row: [Frame x_t] -> [Online ViT f_theta] -----> [Causal ST-Predictor g_psi] -> [Stage 2 Loss]
    #   Bottom row: [Next Frame x_{t+1}] -> [Target ViT f_theta_bar (EMA, STOP-GRAD)] ------> (into Loss)
    # =========================================================================
    ax2 = axes[1]
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 5.0)
    ax2.axis("off")

    panel2_bg = FancyBboxPatch(
        (0.1, 0.08),
        13.8,
        4.80,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#eff6ff",
        edgecolor="#2563eb",
        linewidth=1.5,
        zorder=1,
    )
    ax2.add_patch(panel2_bg)
    ax2.text(
        0.35,
        4.60,
        "STAGE 2: How the JEPA Predictor ($g_\\psi$) is Trained & Exactly Where SIGReg + VICReg Act to Prevent Collapse",
        fontsize=10.5,
        fontweight="bold",
        color="#1e3a8a",
    )

    # Top-Left: SIGReg + VICReg Box (directly above Online ViT)
    draw_box(
        ax2,
        1.35,
        3.22,
        5.65,
        1.20,
        "★ SIGReg + VICReg ACT HERE (Directly on Online Encoder Output $z_t$)",
        "Pools $\\bar{z}_t \\in \\mathbb{R}^{192}$ across the batch:\n"
        "(1) SIGReg ($\\lambda=0.05$): $M=64$ random 1D projections $u^{(m)}$ + Epps-Pulley test $\\to \\mathcal{N}(0, I_{192})$\n"
        "(2) VICReg Hinge ($\\lambda=0.05$): Forces per-dimension $\\mathrm{Std}(\\bar{z}_t) \\geq 1.0$",
        facecolor="#f3e8ff",
        edgecolor="#7e22ce",
        title_color="#6b21a8",
        lw=2.2,
        fontsize_title=8.9,
        fontsize_body=7.8,
    )

    # Top-Right: Action Embed Table (directly above Predictor)
    draw_box(
        ax2,
        7.35,
        3.22,
        3.95,
        1.20,
        "Action Embedding Table $E_a$ [TRAINABLE $8\\times 192$]",
        "Frozen Stage-1 LAM provides discrete code $k_t \\in \\{0..7\\}$\n"
        "Looks up vector $e(k_t) \\in \\mathbb{R}^{192}$ & adds to all 64 spatial tokens of $z_t$",
        facecolor="#fef3c7",
        edgecolor="#d97706",
        title_color="#92400e",
        fontsize_title=9.0,
        fontsize_body=7.9,
    )

    # Middle Row: Frame x_t -> Online ViT -> Predictor -> Stage 2 Loss
    draw_box(
        ax2,
        0.35,
        1.65,
        2.15,
        1.25,
        "Current Frame $x_t$\nPatchEmbed (8x8)",
        "Splits $64\\times 64$ into 64\npatches ($8\\times 8$) + PosEmbed",
        facecolor="#dbeafe",
        edgecolor="#2563eb",
    )
    draw_arrow(ax2, 2.50, 2.28, 3.05, 2.28, "")

    draw_box(
        ax2,
        3.05,
        1.65,
        2.65,
        1.25,
        "Online ViT-S/8 ($f_\\theta$)\n[TRAINABLE via Backprop]",
        "4 Transformer Blocks\nOutputs $z_t \\in \\mathbb{R}^{64\\times 192}$",
        facecolor="#ffffff",
        edgecolor="#1d4ed8",
        lw=2.0,
    )

    # Arrow UP from Online ViT to SIGReg
    draw_arrow(
        ax2,
        4.38,
        2.90,
        4.38,
        3.22,
        "$\\bar{z}_t$",
        color="#7e22ce",
        lw=2.0,
        label_offset=(0.28, 0.0),
    )

    # Arrow RIGHT from Online ViT to Predictor
    draw_arrow(
        ax2,
        5.70,
        2.28,
        7.35,
        2.28,
        "$z_t \\in \\mathbb{R}^{64\\times 192}$",
        color="#1d4ed8",
        lw=1.9,
        label_offset=(0.0, 0.18),
    )

    # Arrow DOWN from Action Embed to Predictor
    draw_arrow(
        ax2,
        9.32,
        3.22,
        9.32,
        2.90,
        "$e(k_t)$",
        color="#b45309",
        lw=1.9,
        label_offset=(0.38, 0.0),
    )

    # Predictor Box
    draw_box(
        ax2,
        7.35,
        1.65,
        3.95,
        1.25,
        "Causal ST-Predictor $g_\\psi$ [TRAINABLE World Model]",
        "2 Spatio-Temporal Blocks (Spatial Attn + Causal Time Attn)\n"
        "Predicts residual transition: $\\hat{z}_{t+1} = z_t + g_\\psi(z_{0..t}, e(k_{0..t}))$",
        facecolor="#dcfce7",
        edgecolor="#15803d",
        title_color="#14532d",
        lw=2.0,
    )

    # Bottom Row: Next Frame x_{t+1} -> Target ViT (EMA, Stop-Grad)
    draw_box(
        ax2,
        0.35,
        0.22,
        2.15,
        1.10,
        "Next Frame $x_{t+1}$\nPatchEmbed (8x8)",
        "Future observation\nat step $t+1$",
        facecolor="#f1f5f9",
        edgecolor="#64748b",
    )
    draw_arrow(ax2, 2.50, 0.77, 3.05, 0.77, "")

    draw_box(
        ax2,
        3.05,
        0.22,
        5.20,
        1.10,
        "Target ViT-S/8 ($f_{\\bar{\\theta}}$) — [STOP-GRAD] (No Backprop)",
        "Outputs target representation $z_{t+1}^{\\mathrm{tgt}} = \\mathrm{sg}[f_{\\bar{\\theta}}(x_{t+1})] \\in \\mathbb{R}^{64\\times 192}$\n"
        "Weights updated purely via Exponential Moving Average: $\\bar{\\theta} \\leftarrow 0.996\\,\\bar{\\theta} + 0.004\\,\\theta$",
        facecolor="#f1f5f9",
        edgecolor="#475569",
        linestyle="--",
    )

    # Dashed EMA arrow from Online ViT down to Target ViT
    draw_arrow(
        ax2,
        4.38,
        1.65,
        4.38,
        1.32,
        "EMA $\\tau=0.996$",
        color="#475569",
        linestyle="--",
        lw=1.6,
        label_offset=(0.68, 0.0),
        fontsize=7.6,
    )

    # Rightmost Box: Stage 2 Loss
    draw_box(
        ax2,
        11.75,
        0.45,
        1.95,
        2.65,
        "Stage 2 Loss\n$\\mathcal{L}_{\\mathrm{JEPA}}$",
        "Latent $L_1$ Prediction:\n$\\|\\hat{z}_{t+1} - \\mathrm{sg}[z_{t+1}^{\\mathrm{tgt}}]\\|_1$\n\n"
        "$+ 0.05\\,\\mathcal{L}_{\\mathrm{SIGReg}}(\\bar{z}_t)$\n"
        "$+ 0.05\\,\\mathcal{L}_{\\mathrm{var}}(\\bar{z}_t)$\n\n"
        "Backprop updates:\n$f_\\theta,\\; E_a,\\; g_\\psi$",
        facecolor="#ffe4e6",
        edgecolor="#e11d48",
        title_color="#9f1239",
        lw=1.9,
    )

    draw_arrow(ax2, 11.30, 2.28, 11.75, 2.28, "$\\hat{z}_{t+1}$", color="#15803d", label_offset=(0.0, 0.18))
    draw_arrow(
        ax2,
        8.25,
        0.77,
        11.75,
        1.10,
        "$\\mathrm{sg}[z_{t+1}^{\\mathrm{tgt}}]$",
        color="#475569",
        label_offset=(0.0, 0.18),
    )

    # =========================================================================
    # PANEL 3: STAGES 3 & 4 — FEW-SHOT ALIGNMENT & DUAL ACTION LOOPS AT TEST TIME
    # =========================================================================
    ax3 = axes[2]
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 3.9)
    ax3.axis("off")

    panel3_bg = FancyBboxPatch(
        (0.1, 0.08),
        13.8,
        3.70,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#f0fdf4",
        edgecolor="#16a34a",
        linewidth=1.5,
        zorder=1,
    )
    ax3.add_patch(panel3_bg)
    ax3.text(
        0.35,
        3.50,
        "STAGES 3 & 4: Dual Action Loops at Test Time — Inner Mental Loop (CEM in JEPA) vs. Outer Physical Loop (MPC in Game)",
        fontsize=10.5,
        fontweight="bold",
        color="#14532d",
    )

    # Outer Loop Box 1: Environment + Encoder
    draw_box(
        ax3,
        0.35,
        1.18,
        2.35,
        1.95,
        "1. Outer Physical Loop\n(Crafter / Grid Env)",
        "Agent observes live frame $x_t$\nand target visual goal $x_{\\mathrm{goal}}$\n\n"
        "Encodes both via frozen $f_\\theta$:\n"
        "$z_t = f_\\theta(x_t),\\; z_{\\mathrm{goal}} = f_\\theta(x_{\\mathrm{goal}})$",
        facecolor="#dbeafe",
        edgecolor="#2563eb",
    )
    draw_arrow(ax3, 2.70, 2.15, 3.30, 2.15, "$z_t, z_{\\mathrm{goal}}$", label_offset=(0.0, 0.18))

    # Inner Mental Loop Box (Highlighted)
    inner_loop_bg = FancyBboxPatch(
        (3.30, 0.98),
        6.05,
        2.35,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#dcfce7",
        edgecolor="#15803d",
        linewidth=2.2,
        zorder=2,
    )
    ax3.add_patch(inner_loop_bg)
    ax3.text(
        6.32,
        3.11,
        "2. INNER MENTAL LOOP: CEM in Latent Space (4 Iters x 128 Rollouts)",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="#14532d",
        zorder=3,
    )

    draw_box(
        ax3,
        3.48,
        1.12,
        2.72,
        1.70,
        "Sample & Rollout in $g_\\psi$",
        "Samples $M=128$ action sequences\n"
        "$k^{(m)} = (k_0,..,k_5) \\in \\{0..7\\}^6$\n"
        "Unrolls 6 steps in latent space:\n"
        "$\\hat{z}_{h+1}^{(m)} = \\hat{z}_h^{(m)} + g_\\psi(\\hat{z}_h^{(m)}, E_a(k_h^{(m)}))$",
        facecolor="#ffffff",
        edgecolor="#16a34a",
    )
    draw_arrow(ax3, 6.20, 1.97, 6.52, 1.97, "")

    draw_box(
        ax3,
        6.52,
        1.12,
        2.65,
        1.70,
        "Score & Refine Top-16 Elites",
        "Computes latent $L_1$ distance to goal:\n"
        "$\\mathrm{Cost}^{(m)} = \\|\\hat{z}_6^{(m)} - z_{\\mathrm{goal}}\\|_1$\n"
        "Updates $P^{(i+1)}$ from Top-16 elites\n"
        "After 4 iters -> picks best $k_0^* \\in \\{0..7\\}$",
        facecolor="#ffffff",
        edgecolor="#16a34a",
    )

    draw_arrow(ax3, 9.35, 2.15, 10.00, 2.15, "$k_0^* \\in \\{0..7\\}$", color="#b45309", label_offset=(0.0, 0.18))

    # Stage 3 Alignment Probe Box
    draw_box(
        ax3,
        10.00,
        1.18,
        3.65,
        1.95,
        "3. Few-Shot Action Alignment ($P_\\phi$)\n[Trained on 0.1% - 10% Labelled Pairs]",
        "2-Layer MLP takes $(k_0^*, \\bar{z}_t)$ and translates\n"
        "latent code to real game control ($A=17$)\n"
        "with log-prior class-imbalance correction:\n"
        "$\\hat{a}_t = \\arg\\max_a [\\log P_\\phi(a|k_0^*,\\bar{z}_t) - 0.5\\log P(a)]$",
        facecolor="#f3e8ff",
        edgecolor="#7e22ce",
        title_color="#581c87",
    )

    # Outer feedback loop orthogonal path along the bottom margin (y=0.58, completely below all boxes)
    ax3.plot([11.82, 11.82, 1.52], [1.18, 0.58, 0.58], color="#1d4ed8", linewidth=1.9, zorder=4)
    draw_arrow(
        ax3,
        1.52,
        0.58,
        1.52,
        1.18,
        "",
        color="#1d4ed8",
        lw=1.9,
        rad=0.0,
    )
    ax3.text(
        6.67,
        0.32,
        "4. Outer MPC Feedback Loop: Execute $\\hat{a}_t$ in Crafter -> Environment steps to $x_{t+1}$ -> Re-plan next step until $x_t = x_{\\mathrm{goal}}$",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#1d4ed8",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=5,
    )

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig0_architecture.png"
    pdf_path = out_dir / "fig0_architecture.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report_dir = Path("report/figures")
    if report_dir.exists():
        shutil.copy2(png_path, report_dir / "fig0_architecture.png")
        shutil.copy2(pdf_path, report_dir / "fig0_architecture.pdf")
    print(f"Saved architecture diagram to {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
