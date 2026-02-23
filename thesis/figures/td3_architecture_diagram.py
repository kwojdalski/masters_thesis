from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


FIG_DIR = Path(__file__).resolve().parent
OUT_PNG = FIG_DIR / "td3_learning_architecture.png"
OUT_SVG = FIG_DIR / "td3_learning_architecture.svg"


def draw_arrow(ax, start, end, text=None, text_offset=(0.0, 0.0), lw=1.2):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color="black",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=9, ha="center", va="center")


def draw_mlp(
    ax,
    x,
    y,
    w,
    h,
    layers,
    title,
    dashed=True,
    title_y_pad=0.02,
):
    box = Rectangle(
        (x, y),
        w,
        h,
        fill=False,
        linewidth=1.2,
        linestyle=(0, (6, 3)) if dashed else "solid",
        edgecolor="black",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h + title_y_pad, title, fontsize=10, ha="center", va="bottom")

    xs = []
    ys = []
    for idx, n_nodes in enumerate(layers):
        xi = x + w * (0.12 + 0.76 * idx / max(1, len(layers) - 1))
        xs.append(xi)
        if n_nodes == 1:
            ys.append([y + h * 0.5])
        else:
            top = y + h * 0.82
            bottom = y + h * 0.18
            step = (top - bottom) / (n_nodes - 1)
            ys.append([top - i * step for i in range(n_nodes)])

    for li in range(len(layers) - 1):
        for y0 in ys[li]:
            for y1 in ys[li + 1]:
                ax.plot([xs[li], xs[li + 1]], [y0, y1], color="black", linewidth=0.5, alpha=0.7)

    for li, n_nodes in enumerate(layers):
        for yi in ys[li]:
            circ = Circle((xs[li], yi), radius=min(w, h) * 0.045, facecolor="#f2f2f2", edgecolor="black", linewidth=1.0)
            ax.add_patch(circ)

    return {
        "box": (x, y, w, h),
        "input_anchor": (x, y + h / 2),
        "output_anchor": (x + w, y + h / 2),
        "left_mid": (x, y + h / 2),
        "right_mid": (x + w, y + h / 2),
        "bottom_mid": (x + w / 2, y),
        "top_mid": (x + w / 2, y + h),
    }


def draw_label_box(ax, x, y, w, h, title, lines):
    box = Rectangle((x, y), w, h, fill=False, linewidth=1.0, edgecolor="black")
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.03, title, fontsize=10, ha="center", va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.02, y + h - 0.08 - i * 0.05, line, fontsize=9, ha="left", va="top")
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2)}


def build_td3_figure():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    actor = draw_mlp(ax, 0.10, 0.58, 0.22, 0.22, [3, 4, 3], r"Actor network $\mu_\phi(s_t)$")
    critic1 = draw_mlp(ax, 0.43, 0.62, 0.22, 0.18, [4, 4, 3, 1], r"Critic 1 $Q_{\theta_1}(s_t,a_t)$")
    critic2 = draw_mlp(ax, 0.43, 0.38, 0.22, 0.18, [4, 4, 3, 1], r"Critic 2 $Q_{\theta_2}(s_t,a_t)$")

    target_actor = draw_mlp(ax, 0.72, 0.58, 0.18, 0.18, [3, 3, 2], r"Target actor $\mu_{\phi'}(s_{t+1})$")
    target_c1 = draw_mlp(ax, 0.72, 0.39, 0.18, 0.15, [4, 3, 1], r"Target critic $Q_{\theta_1'}$", title_y_pad=0.015)
    target_c2 = draw_mlp(ax, 0.72, 0.18, 0.18, 0.15, [4, 3, 1], r"Target critic $Q_{\theta_2'}$", title_y_pad=0.015)

    replay = draw_label_box(
        ax,
        0.08,
        0.18,
        0.22,
        0.20,
        "Replay Buffer",
        [
            r"$(s_t,a_t,r_t,s_{t+1},d_t)$",
            "sample minibatches",
            "off-policy updates",
        ],
    )
    target_min = draw_label_box(
        ax,
        0.52,
        0.12,
        0.14,
        0.12,
        "TD3 target",
        [
            r"$y_t=r_t+\gamma(1-d_t)\,$",
            r"$\min(Q'_1,Q'_2)$",
        ],
    )
    smooth = draw_label_box(
        ax,
        0.70,
        0.82,
        0.20,
        0.10,
        "Target policy smoothing",
        [
            r"$\tilde a=\mu_{\phi'}(s_{t+1})+\epsilon$",
        ],
    )
    delayed = draw_label_box(
        ax,
        0.32,
        0.84,
        0.30,
        0.10,
        "Delayed actor update",
        [
            r"update actor every $d$ critic steps",
        ],
    )

    # External inputs
    ax.text(0.03, 0.69, r"$s_t$", fontsize=11, ha="left", va="center")
    ax.text(0.03, 0.27, "sampled batch", fontsize=10, ha="left", va="center")
    ax.text(0.93, 0.74, r"$a_t=\mu(s_t)+\eta_t$", fontsize=10, ha="right", va="center")

    # State to actor and critics
    draw_arrow(ax, (0.05, 0.69), actor["left_mid"])
    draw_arrow(ax, (0.05, 0.69), (0.39, 0.71))
    draw_arrow(ax, (0.05, 0.69), (0.39, 0.47))

    # Actor output to critics with exploration annotation
    draw_arrow(ax, actor["right_mid"], (0.39, 0.69), text=r"$\mu_\phi(s_t)$", text_offset=(0.00, 0.03))
    draw_arrow(ax, (0.39, 0.69), (0.43, 0.71), text=r"$+\eta_t$", text_offset=(0.00, 0.03))
    draw_arrow(ax, (0.39, 0.69), (0.43, 0.47))

    # Critic outputs
    draw_arrow(ax, critic1["right_mid"], (0.68, 0.71), text=r"$Q_1(s_t,a_t)$", text_offset=(0.03, 0.03))
    draw_arrow(ax, critic2["right_mid"], (0.68, 0.47), text=r"$Q_2(s_t,a_t)$", text_offset=(0.03, -0.03))

    # Replay buffer links
    draw_arrow(ax, (0.05, 0.27), replay["left"])
    draw_arrow(ax, replay["right"], (0.39, 0.44))
    draw_arrow(ax, replay["right"], (0.39, 0.68))

    # Target branch
    draw_arrow(ax, (0.61, 0.69), (0.72, 0.67), text=r"$s_{t+1}$", text_offset=(0.01, 0.03))
    draw_arrow(ax, target_actor["right_mid"], smooth["left"], text=r"$\mu_{\phi'}(s_{t+1})$", text_offset=(0.0, 0.03))
    draw_arrow(ax, smooth["left"], (0.81, 0.54))
    draw_arrow(ax, (0.81, 0.54), (0.72, 0.46))
    draw_arrow(ax, (0.81, 0.54), (0.72, 0.25))
    ax.text(0.83, 0.50, r"$\tilde a_{t+1}$", fontsize=9, ha="left", va="center")

    draw_arrow(ax, (0.61, 0.69), (0.72, 0.46))
    draw_arrow(ax, (0.61, 0.69), (0.72, 0.25))

    draw_arrow(ax, target_c1["right_mid"], (0.66, 0.195))
    draw_arrow(ax, target_c2["right_mid"], (0.66, 0.165))
    draw_arrow(ax, (0.66, 0.18), target_min["right"])

    # Critic losses from target
    draw_arrow(ax, target_min["left"], (0.43, 0.40), text="critic loss", text_offset=(0.00, -0.03))
    draw_arrow(ax, target_min["left"], (0.43, 0.64))

    # Actor update and delayed policy update note
    draw_arrow(ax, critic1["top_mid"], (0.47, 0.84), text=r"$\nabla_a Q_1$", text_offset=(0.02, 0.03))
    draw_arrow(ax, actor["top_mid"], (0.22, 0.84), text=r"$\nabla_\phi \mu_\phi$", text_offset=(-0.01, 0.03))
    draw_arrow(ax, (0.22, 0.84), delayed["left"])
    draw_arrow(ax, (0.47, 0.84), delayed["left"])

    # Soft updates to targets
    draw_arrow(ax, actor["right_mid"], (0.72, 0.60), text=r"soft update $\tau$", text_offset=(0.05, -0.05))
    draw_arrow(ax, critic1["right_mid"], (0.72, 0.44), text=r"$\tau$", text_offset=(0.03, 0.03))
    draw_arrow(ax, critic2["right_mid"], (0.72, 0.23), text=r"$\tau$", text_offset=(0.03, -0.02))

    # Group labels
    ax.text(0.16, 0.05, "Figure: TD3 learning architecture (actor, twin critics, and target networks).", fontsize=11, ha="left")
    ax.text(0.12, 0.52, "Online networks", fontsize=10, ha="left", va="bottom")
    ax.text(0.72, 0.52, "Target networks", fontsize=10, ha="left", va="bottom")

    fig.tight_layout(pad=0.5)
    return fig


def main():
    fig = build_td3_figure()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_SVG}")


if __name__ == "__main__":
    main()
