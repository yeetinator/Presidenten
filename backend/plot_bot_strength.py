"""Draws the README's bot strength chart from results/bot_ladder.json (and
results/human_placement.json when it exists), in a light and a dark version.

python plot_bot_strength.py
"""

import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bot_ladder import FIELDS, RESULTS_PATH
from human_placement import RESULTS_PATH as HUMAN_RESULTS_PATH

OUT = "../public/bot_strength_{mode}.png"
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": {"DMC": "#2a78d6", "Baseline": "#eb6834"},
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#353532",
        "series": {"DMC": "#3987e5", "Baseline": "#d95926"},
    },
}
FIELD_LABELS = {"DMC": "At a table of DMC bots", "Baseline": "At a table of Baseline bots"}
PLAYERS = ["Random", "Baseline", "ISMCTS", "PPO", "DMC"]
HUMAN = "You (human)"


def load_points():
    """{player: {field: (beat rate, half-width, is reference)}}"""
    with open(RESULTS_PATH) as f:
        ladder = json.load(f)
    points = {}
    for player in PLAYERS:
        for field in FIELDS:
            if player == field:
                # Level with copies of itself by definition
                points.setdefault(player, {})[field] = (0.5, 0.0, True)
            elif (entry := ladder.get(f"{player} vs {field}")) is not None:
                points.setdefault(player, {})[field] = (
                    entry["beat_rate"],
                    entry["beat_rate_ci"],
                    False,
                )

    if os.path.exists(HUMAN_RESULTS_PATH):
        with open(HUMAN_RESULTS_PATH) as f:
            vs_dmc = json.load(f)["bots"]["DMC"]
        # DMC is at 50% in its own table, so the human sits their paired
        # difference from DMC away from it
        lo, hi = vs_dmc["diff_ci"]
        points[HUMAN] = {"DMC": (0.5 + vs_dmc["diff"], (hi - lo) / 2, False)}
    return points


def draw(points, mode):
    theme = THEMES[mode]
    rows = sorted(points, key=lambda p: points[p].get("DMC", (0,))[0])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    offsets = {"DMC": 0.12, "Baseline": -0.12}
    for y, player in enumerate(rows):
        for field, (rate, half, reference) in points[player].items():
            color = theme["series"][field]
            yy = y + offsets[field]
            if half > 0:
                ax.plot(
                    [100 * (rate - half), 100 * (rate + half)],
                    [yy, yy],
                    color=color,
                    lw=2,
                    solid_capstyle="round",
                )
            ax.scatter(
                [100 * rate],
                [yy],
                # A ring reads bigger than a dot of the same size
                s=38 if reference else 60,
                zorder=3,
                facecolor=theme["surface"] if reference else color,
                edgecolor=color if reference else theme["surface"],
                linewidth=2,
            )

    ax.axvline(50, color=theme["muted"], lw=1, zorder=0)
    ax.text(
        50.8, len(rows) - 0.45, "level with the table", color=theme["muted"],
        fontsize=8, va="center",
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        rows, color=theme["text"], fontsize=10,
        fontweight="normal",
    )
    for label in ax.get_yticklabels():
        if label.get_text() == HUMAN:
            label.set_fontweight("bold")
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels([f"{t}%" for t in range(0, 101, 25)], color=theme["muted"], fontsize=9)
    ax.set_xlabel(
        "Beat rate: share of opponents finished ahead of per round (95% intervals)",
        color=theme["muted"], fontsize=9,
    )
    ax.grid(axis="x", color=theme["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([], [], marker="o", lw=0, markersize=7, color=theme["series"][f], label=FIELD_LABELS[f])
        for f in FIELDS
    ] + [
        Line2D(
            [], [], marker="o", lw=0, markersize=7, markerfacecolor=theme["surface"],
            markeredgecolor=theme["muted"], markeredgewidth=1.5,
            label="A bot among copies of itself (50% by definition)",
        )
    ]
    legend = ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3,
        frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_color(theme["text"])

    fig.tight_layout()
    fig.savefig(OUT.format(mode=mode), facecolor=theme["surface"])
    plt.close(fig)


def main():
    points = load_points()
    for mode in THEMES:
        draw(points, mode)
        print(f"Wrote {OUT.format(mode=mode)}")


if __name__ == "__main__":
    main()
