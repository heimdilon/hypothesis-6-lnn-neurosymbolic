from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from run_lnn_experiment import (
    ACTION_DELTAS,
    FIGURES,
    ROBOT_RADIUS,
    SAFE_MARGIN,
    WORLD_SIZE,
    FixedPolicy,
    collides,
    expert_scores,
    features,
    make_liquid_policy,
    noisy_sensor,
    obstacle_clearance,
    ray_cast,
    scenario_obstacles,
    scenario_sensor_params,
    supervised_safe_action,
    train_fixed_policy,
    wrap_angle,
)


def simulate_trace(controller_name: str, controller, scenario: str, seed: int, supervisor: bool) -> dict:
    rng = np.random.default_rng(seed)
    obstacles = scenario_obstacles(scenario)
    noise, dropout, bias = scenario_sensor_params(scenario)
    start = np.array([1.0, 1.0]) + rng.normal(0, 0.06, size=2)
    goal = np.array([8.8, 8.7])
    pos = start.copy()
    heading = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
    controller.reset()

    positions = [pos.copy()]
    clearances = [obstacle_clearance(pos, obstacles)]
    override_frames = []
    status = "timeout"

    for step in range(120):
        true_ranges = ray_cast(pos, heading, obstacles)
        obs_ranges, uncertainty = noisy_sensor(true_ranges, rng, noise, dropout, bias)
        x = features(pos, heading, goal, obs_ranges, uncertainty)
        action, scores = controller.act(x, pos, heading, goal, obstacles)
        corrective = None
        if supervisor:
            safe_action, did_override = supervised_safe_action(pos, heading, goal, obstacles, scores)
            if did_override:
                corrective = safe_action
                action = safe_action
                override_frames.append(step)

        new_heading = wrap_angle(heading + float(ACTION_DELTAS[action]))
        new_pos = pos + 0.32 * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        progress = np.linalg.norm(goal - pos) - np.linalg.norm(goal - new_pos)
        reward = progress * 2.0 + min(clearance, 1.0) * 0.15
        if clearance <= 0:
            reward -= 4.0
        if progress < -0.02:
            reward -= 0.25

        if corrective is None and clearance < SAFE_MARGIN:
            # Give the online liquid policy a local expert signal after risky behavior.
            corrective = int(np.argmax(expert_scores(pos, heading, goal, obstacles)))
        controller.learn(action, reward, corrective)

        positions.append(new_pos.copy())
        clearances.append(clearance)
        pos = new_pos
        heading = new_heading

        if clearance <= 0:
            status = "collision"
            break
        if np.linalg.norm(goal - pos) < 0.45:
            status = "success"
            break

    return {
        "name": controller_name,
        "positions": np.vstack(positions),
        "clearances": np.array(clearances),
        "override_frames": override_frames,
        "status": status,
        "steps": len(positions) - 1,
    }


def draw_world(ax, obstacles, goal):
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for obs in obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.r + ROBOT_RADIUS, color="#c64b4b", alpha=0.25)
        core = plt.Circle((obs.x, obs.y), obs.r, color="#8f1d1d", alpha=0.65)
        ax.add_patch(circle)
        ax.add_patch(core)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=180, color="#198754", zorder=4)
    ax.scatter([1.0], [1.0], marker="o", s=45, color="#333333", zorder=4)


def make_animation(
    scenario: str,
    seed: int,
    out_path: Path,
    liquid_backend: str = "cfc",
    ncp_hidden: int = 32,
    ncp_sparsity: float = 0.50,
    ncp_baseline_scale: float = 1.0,
    ncp_residual_scale: float = 0.35,
    ncp_learning_rate: float = 0.006,
) -> None:
    rng = np.random.default_rng(42)
    base_weights = train_fixed_policy(rng)
    input_dim = base_weights.shape[0]
    traces = [
        simulate_trace("fixed policy", FixedPolicy(base_weights.copy()), scenario, seed, False),
        simulate_trace(
            "liquid online",
            make_liquid_policy(
                np.random.default_rng(seed + 1),
                input_dim,
                base_weights.copy(),
                adaptive=True,
                backend=liquid_backend,
                hidden_dim=ncp_hidden,
                sparsity=ncp_sparsity,
                baseline_scale=ncp_baseline_scale,
                residual_scale=ncp_residual_scale,
                learning_rate=ncp_learning_rate,
            ),
            scenario,
            seed,
            False,
        ),
        simulate_trace(
            "liquid + symbolic supervisor",
            make_liquid_policy(
                np.random.default_rng(seed + 2),
                input_dim,
                base_weights.copy(),
                adaptive=True,
                backend=liquid_backend,
                hidden_dim=ncp_hidden,
                sparsity=ncp_sparsity,
                baseline_scale=ncp_baseline_scale,
                residual_scale=ncp_residual_scale,
                learning_rate=ncp_learning_rate,
            ),
            scenario,
            seed,
            True,
        ),
    ]
    obstacles = scenario_obstacles(scenario)
    goal = np.array([8.8, 8.7])
    colors = ["#315f9f", "#b85c00", "#1b7f55"]
    max_frames = max(len(t["positions"]) for t in traces)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    artists = []
    for ax, trace, color in zip(axes, traces, colors):
        draw_world(ax, obstacles, goal)
        ax.set_title(f"{trace['name']}\nstatus: pending")
        (line,) = ax.plot([], [], color=color, lw=2.3)
        robot = plt.Circle((1, 1), ROBOT_RADIUS, color=color, alpha=0.95, zorder=5)
        ax.add_patch(robot)
        text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=8)
        artists.append((line, robot, text, ax))

    fig.suptitle(f"H6 2D navigation simulation - {scenario} / seed {seed}", fontsize=12)

    def update(frame: int):
        changed = []
        for trace, (line, robot, text, ax) in zip(traces, artists):
            positions = trace["positions"]
            idx = min(frame, len(positions) - 1)
            xy = positions[: idx + 1]
            line.set_data(xy[:, 0], xy[:, 1])
            robot.center = tuple(positions[idx])
            clearance = trace["clearances"][idx]
            status = trace["status"] if frame >= len(positions) - 1 else "running"
            text.set_text(
                f"step={idx}\nclearance={clearance:.2f}\noverrides={len(trace['override_frames'])}\n{status}"
            )
            ax.set_title(f"{trace['name']}\nstatus: {status}")
            changed.extend([line, robot, text, ax.title])
        return changed

    ani = animation.FuncAnimation(fig, update, frames=max_frames + 10, interval=130, blit=False)
    writer = animation.PillowWriter(fps=8)
    ani.save(out_path, writer=writer)
    png_path = out_path.with_suffix(".png")
    update(max_frames)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="shifted_clutter")
    parser.add_argument("--seed", type=int, default=420120)
    parser.add_argument("--out", default="h6_2d_lnn_neurosymbolic.gif")
    parser.add_argument("--liquid-backend", choices=["legacy", "cfc", "ltc"], default="cfc")
    parser.add_argument("--ncp-hidden", type=int, default=32)
    parser.add_argument("--ncp-sparsity", type=float, default=0.50)
    parser.add_argument("--ncp-baseline-scale", type=float, default=1.0)
    parser.add_argument("--ncp-residual-scale", type=float, default=0.35)
    parser.add_argument("--ncp-learning-rate", type=float, default=0.006)
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    make_animation(
        args.scenario,
        args.seed,
        FIGURES / args.out,
        liquid_backend=args.liquid_backend,
        ncp_hidden=args.ncp_hidden,
        ncp_sparsity=args.ncp_sparsity,
        ncp_baseline_scale=args.ncp_baseline_scale,
        ncp_residual_scale=args.ncp_residual_scale,
        ncp_learning_rate=args.ncp_learning_rate,
    )


if __name__ == "__main__":
    main()
