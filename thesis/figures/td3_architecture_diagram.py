from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


FIG_DIR = Path(__file__).resolve().parent
OUT_PNG = FIG_DIR / "td3_learning_architecture.png"
OUT_SVG = FIG_DIR / "td3_learning_architecture.svg"


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
    fontsize: int = 9,
    lw: float = 1.2,
    color: str = "black",
    linestyle: str = "solid",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    if text:
        if text_xy is None:
            text_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(text_xy[0], text_xy[1], text, fontsize=fontsize, ha="center", va="center", color=color)


def draw_poly_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
    fontsize: int = 9,
    lw: float = 1.2,
    color: str = "black",
    linestyle: str = "solid",
):
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, linestyle=linestyle)
    draw_arrow(
        ax,
        points[-2],
        points[-1],
        text=text,
        text_xy=text_xy,
        fontsize=fontsize,
        lw=lw,
        color=color,
        linestyle=linestyle,
    )


def draw_mlp(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    layers: list[int],
    title: str,
    *,
    title_y_pad: float = 0.014,
):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linewidth=1.2,
            linestyle=(0, (6, 3)),
            edgecolor="black",
        )
    )
    ax.text(x + w / 2, y + h + title_y_pad, title, fontsize=10, ha="center", va="bottom")

    xs: list[float] = []
    ys: list[list[float]] = []
    for i, n in enumerate(layers):
        xi = x + w * (0.12 + 0.76 * i / max(1, len(layers) - 1))
        xs.append(xi)
        if n == 1:
            ys.append([y + h * 0.5])
        else:
            y_top, y_bot = y + h * 0.82, y + h * 0.18
            step = (y_top - y_bot) / (n - 1)
            ys.append([y_top - k * step for k in range(n)])

    for i in range(len(layers) - 1):
        for y0 in ys[i]:
            for y1 in ys[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y0, y1], color="black", linewidth=0.45, alpha=0.4)

    r = min(w, h) * 0.042
    for col in ys:
        for yy in col:
            ax.add_patch(Circle((xs[ys.index(col)], yy), r, facecolor="#f7f7f7", edgecolor="black", linewidth=1.0))

    return {
        "left_mid": (x, y + h / 2),
        "right_mid": (x + w, y + h / 2),
        "top_mid": (x + w / 2, y + h),
        "bottom_mid": (x + w / 2, y),
        "left_upper": (x, y + h * 0.70),
        "left_lower": (x, y + h * 0.30),
    }


def draw_note_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str | None = None,
):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.0, edgecolor="black"))
    ax.text(x + w / 2, y + h * 0.68, title, fontsize=9.5, ha="center", va="center")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.20, subtitle, fontsize=7.5, ha="center", va="center")
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2), "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}


def build_td3_figure():
    fig, ax = plt.subplots(figsize=(14.5, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Online networks (left / center)
    actor = draw_mlp(ax, 0.08, 0.39, 0.23, 0.19, [3, 4, 3], r"Actor network $\mu_\phi(s_t)$")
    critic1 = draw_mlp(ax, 0.42, 0.49, 0.22, 0.14, [4, 4, 3, 1], r"Critic 1 $Q_{\theta_1}(s_t,a_t)$")
    critic2 = draw_mlp(ax, 0.42, 0.23, 0.22, 0.14, [4, 4, 3, 1], r"Critic 2 $Q_{\theta_2}(s_t,a_t)$")

    # Target networks panel (right)
    panel_x, panel_y, panel_w, panel_h = 0.70, 0.17, 0.26, 0.50
    ax.add_patch(Rectangle((panel_x, panel_y), panel_w, panel_h, fill=False, linewidth=1.0, edgecolor="black"))
    ax.text(panel_x + 0.01, panel_y + 0.01, "Target networks", fontsize=10, ha="left", va="bottom")

    target_actor = draw_mlp(ax, 0.75, 0.52, 0.18, 0.11, [3, 3, 2], r"Target actor $\mu_{\phi'}(s_{t+1})$", title_y_pad=0.008)
    target_c1 = draw_mlp(ax, 0.75, 0.36, 0.18, 0.09, [4, 3, 1], r"Target critic $Q_{\theta_1'}$", title_y_pad=0.008)
    target_c2 = draw_mlp(ax, 0.75, 0.22, 0.18, 0.09, [4, 3, 1], r"Target critic $Q_{\theta_2'}$", title_y_pad=0.008)

    # TD3-specific notes / target box
    delayed = draw_note_box(ax, 0.30, 0.835, 0.32, 0.07, "Delayed actor update", "actor updated every d critic steps")
    smoothing = draw_note_box(ax, 0.66, 0.835, 0.31, 0.07, "Target policy smoothing", r"$\tilde a_{t+1}=\mu_{\phi'}(s_{t+1})+\epsilon$")
    td3_target = draw_note_box(ax, 0.46, 0.05, 0.25, 0.085, "TD3 target", r"$y_t=r_t+\gamma(1-d_t)\min(Q'_1,Q'_2)$")

    ax.text(0.09, 0.33, "Online networks", fontsize=10, ha="left")

    # s_t bus (actor + critics)
    s_y = actor["left_mid"][1]
    state_split_x = 0.36
    ax.text(0.03, s_y, r"$s_t$", fontsize=11, ha="right", va="center")
    draw_poly_arrow(ax, [(0.04, s_y), actor["left_mid"]], lw=1.1)
    ax.plot([actor["left_mid"][0], state_split_x], [s_y, s_y], color="black", linewidth=1.1)
    draw_poly_arrow(ax, [(state_split_x, s_y), (state_split_x, critic1["left_lower"][1]), critic1["left_lower"]], lw=1.0)
    draw_poly_arrow(ax, [(state_split_x, s_y), (state_split_x, critic2["left_lower"][1]), critic2["left_lower"]], lw=1.0)
    ax.text(state_split_x + 0.01, s_y + 0.02, r"$s_t$", fontsize=8, ha="left")

    # Actor action to both online critics
    action_split = (0.35, actor["right_mid"][1])
    draw_poly_arrow(ax, [actor["right_mid"], action_split], text=r"$\mu_\phi(s_t)$", text_xy=(0.33, s_y - 0.03), fontsize=8)
    draw_poly_arrow(ax, [action_split, (0.42, critic1["left_upper"][1]), critic1["left_upper"]], lw=1.0)
    draw_poly_arrow(ax, [action_split, (0.42, critic2["left_upper"][1]), critic2["left_upper"]], lw=1.0)
    ax.text(
        0.475,
        0.575,
        r"$a_t=\mu_\phi(s_t)+\eta_t$",
        fontsize=8,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
    )

    # Online critic outputs (kept short and aligned)
    q1_out = (0.67, critic1["right_mid"][1])
    q2_out = (0.67, critic2["right_mid"][1])
    draw_poly_arrow(ax, [critic1["right_mid"], q1_out], text=r"$Q_1(s_t,a_t)$", text_xy=(0.675, q1_out[1] + 0.03), fontsize=8)
    draw_poly_arrow(ax, [critic2["right_mid"], q2_out], text=r"$Q_2(s_t,a_t)$", text_xy=(0.675, q2_out[1] - 0.03), fontsize=8)

    # s_{t+1} bus entering target panel (single clean vertical bus)
    bus_x = 0.72
    bus_top = target_actor["left_mid"][1]
    bus_bot = target_c2["left_mid"][1]
    ax.plot([bus_x, bus_x], [bus_bot - 0.06, bus_top + 0.06], color="black", linewidth=1.2)
    ax.text(bus_x - 0.02, (bus_top + bus_bot) / 2, r"$s_{t+1}$", fontsize=9, ha="right", va="center")
    draw_poly_arrow(ax, [(bus_x, target_actor["left_mid"][1]), target_actor["left_mid"]], lw=1.0)
    draw_poly_arrow(ax, [(bus_x, target_c1["left_mid"][1]), target_c1["left_mid"]], lw=1.0)
    draw_poly_arrow(ax, [(bus_x, target_c2["left_mid"][1]), target_c2["left_mid"]], lw=1.0)

    # Target actor -> smoothing box -> smoothed action bus
    draw_poly_arrow(ax, [target_actor["top_mid"], smoothing["bottom"]], color="#444444", lw=1.0)
    action_bus_x = 0.952
    ax.plot([action_bus_x, action_bus_x], [target_c2["left_upper"][1], target_actor["right_mid"][1] + 0.02], color="#444444", linewidth=1.0)
    draw_poly_arrow(
        ax,
        [target_actor["right_mid"], (action_bus_x, target_actor["right_mid"][1]), (action_bus_x, target_c1["left_upper"][1]), target_c1["left_upper"]],
        text=r"$\tilde a_{t+1}$",
        text_xy=(0.958, target_actor["right_mid"][1] + 0.028),
        fontsize=8,
        color="#444444",
        lw=1.0,
    )
    draw_poly_arrow(ax, [(action_bus_x, target_c2["left_upper"][1]), target_c2["left_upper"]], color="#444444", lw=1.0)

    # Target critic outputs -> TD3 target (single merge bus)
    qprime_merge_x = 0.975
    qmerge_y = 0.11
    ax.plot([qprime_merge_x, qprime_merge_x], [qmerge_y, target_c1["right_mid"][1]], color="black", linewidth=1.0)
    draw_poly_arrow(ax, [target_c1["right_mid"], (qprime_merge_x, target_c1["right_mid"][1])], lw=1.0)
    draw_poly_arrow(ax, [target_c2["right_mid"], (qprime_merge_x, target_c2["right_mid"][1])], lw=1.0)
    draw_poly_arrow(ax, [(qprime_merge_x, qmerge_y), (td3_target["right"][0], qmerge_y), td3_target["right"]], lw=1.0)
    ax.text(
        0.90,
        qmerge_y + 0.02,
        r"$Q'_1, Q'_2$",
        fontsize=8,
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
    )

    # TD3 target used for critic losses (routed left of the critic boxes to avoid overlap)
    loss_bus_x = 0.40
    loss_bus_y = 0.15
    draw_poly_arrow(ax, [td3_target["top"], (loss_bus_x, loss_bus_y)], lw=1.1)
    draw_poly_arrow(ax, [(loss_bus_x, loss_bus_y), (loss_bus_x, 0.45), (critic1["bottom_mid"][0], 0.45), critic1["bottom_mid"]], lw=1.0)
    draw_poly_arrow(ax, [(loss_bus_x, loss_bus_y), (critic2["bottom_mid"][0], loss_bus_y), critic2["bottom_mid"]], lw=1.0)
    ax.text(
        0.58,
        0.245,
        "critic losses",
        fontsize=8,
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
    )

    # Dashed note arrows (kept outside dense regions)
    draw_poly_arrow(
        ax,
        [critic1["top_mid"], (critic1["top_mid"][0], 0.78), delayed["right"]],
        color="#777777",
        linestyle="dashed",
        lw=0.9,
    )
    draw_poly_arrow(
        ax,
        [delayed["left"], (actor["top_mid"][0], 0.78), actor["top_mid"]],
        color="#777777",
        linestyle="dashed",
        lw=0.9,
    )

    soft_text_xy = (0.78, 0.71)
    ax.text(
        soft_text_xy[0],
        soft_text_xy[1],
        r"soft target updates $\tau$",
        fontsize=9,
        ha="center",
        color="#777777",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
    )
    draw_poly_arrow(ax, [(0.64, 0.69), (0.72, 0.69), (0.75, 0.60)], color="#777777", linestyle="dashed", lw=0.9)

    # Caption
    ax.text(
        0.5,
        0.012,
        "Figure: TD3 architecture (online actor + twin critics, target networks, and TD3 target computation).",
        fontsize=11,
        ha="center",
        va="center",
    )
    fig.tight_layout(pad=0.35)
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
