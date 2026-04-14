from __future__ import annotations

import math

import matplotlib.pyplot as plt

from run_lnn_experiment import FIGURES, HARD_SCENARIOS, WORLD_SIZE, scenario_obstacles


START = (1.0, 1.0)
GOAL = (8.8, 8.7)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    cols = 3
    rows = math.ceil(len(HARD_SCENARIOS) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, scenario in zip(axes_flat, HARD_SCENARIOS):
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        for obs in scenario_obstacles(scenario):
            ax.add_patch(plt.Circle((obs.x, obs.y), obs.r, color="#555555", alpha=0.85))
        ax.scatter([START[0]], [START[1]], s=55, c="#1b9e77", label="start")
        ax.scatter([GOAL[0]], [GOAL[1]], s=70, marker="*", c="#d95f02", label="goal")
    for ax in axes_flat[len(HARD_SCENARIOS) :]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper left", fontsize=8)
    out_path = FIGURES / "h6_hard_map_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
