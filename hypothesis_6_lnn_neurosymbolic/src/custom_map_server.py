from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
import sys
import threading
import time
import unicodedata
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_lnn_experiment import (
    ACTION_DELTAS,
    FIGURES,
    MAX_RANGE,
    ROBOT_RADIUS,
    WORLD_SIZE,
    FixedPolicy,
    Obstacle,
    features,
    make_liquid_policy,
    noisy_sensor,
    norm,
    obstacle_clearance,
    plan_grid_route,
    ray_cast,
    train_fixed_policy,
    wrap_angle,
)


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
OUTPUT_DIR = FIGURES / "custom_maps"
MAP_DIR = ROOT / "saved_maps"
DEFAULT_START = np.array([1.0, 1.0], dtype=float)
DEFAULT_GOAL = np.array([8.8, 8.7], dtype=float)

BASE_WEIGHTS: np.ndarray | None = None
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()

DEFAULT_PARAMS = {
    "seed": 770005,
    "noise": 0.18,
    "dropout": 0.12,
    "sensorBias": 0.96,
    "speed": 0.32,
    "maxSteps": 220,
    "successRadius": 0.45,
    "safeMargin": 0.35,
    "plannerClearance": 0.04,
    "plannerLookahead": 0.85,
    "plannerWeight": 1.0,
    "policyWeight": 0.35,
    "liquidCell": "cfc",
    "ncpHidden": 32,
    "ncpSparsity": 0.50,
    "ncpBaselineScale": 1.0,
    "ncpResidualScale": 0.35,
    "ncpLearningRate": 0.006,
    "makeGif": True,
}


def get_base_weights() -> np.ndarray:
    global BASE_WEIGHTS
    if BASE_WEIGHTS is None:
        BASE_WEIGHTS = train_fixed_policy(np.random.default_rng(42))
    return BASE_WEIGHTS


def sanitize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name or "custom_map")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")
    return slug or "custom_map"


def unique_slug(slug: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 1000):
        candidate = slug if idx == 1 else f"{slug}_run{idx:02d}"
        if not (OUTPUT_DIR / f"{candidate}_config.json").exists():
            return candidate
    raise RuntimeError("Could not allocate output file name.")


def update_job(job_id: str, **fields) -> None:
    with JOB_LOCK:
        job = JOBS.setdefault(job_id, {"id": job_id})
        job.update(fields)
        job["updated"] = time.time()


def get_job(job_id: str) -> dict:
    with JOB_LOCK:
        return dict(JOBS.get(job_id, {"ok": False, "error": "Job not found."}))


def start_simulation_job(payload: dict) -> dict:
    job_id = uuid.uuid4().hex
    update_job(
        job_id,
        ok=True,
        status="queued",
        progress=1,
        message="Simülasyon kuyruğa alındı.",
        created=time.time(),
    )

    def worker() -> None:
        try:
            update_job(job_id, status="running", progress=2, message="Hazırlanıyor.")

            def progress(percent: int, message: str) -> None:
                update_job(job_id, progress=int(max(0, min(99, percent))), message=message)

            result = write_outputs(payload, progress=progress)
            update_job(job_id, status="done", progress=100, message="Tamamlandı.", result=result)
        except Exception as exc:  # noqa: BLE001
            update_job(job_id, status="error", progress=100, message=str(exc), error=str(exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return {"ok": True, "job_id": job_id}


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def parse_obstacles(raw: list[dict]) -> list[Obstacle]:
    obstacles = []
    for item in raw:
        x = clamp(float(item.get("x", 5.0)), 0.25, WORLD_SIZE - 0.25)
        y = clamp(float(item.get("y", 5.0)), 0.25, WORLD_SIZE - 0.25)
        r = clamp(float(item.get("r", 0.35)), 0.08, 1.10)
        obstacles.append(Obstacle(x, y, r))
    return obstacles


def parse_point(raw: dict | None, default: np.ndarray) -> np.ndarray:
    raw = raw or {}
    return np.array(
        [
            clamp(float(raw.get("x", default[0])), 0.25, WORLD_SIZE - 0.25),
            clamp(float(raw.get("y", default[1])), 0.25, WORLD_SIZE - 0.25),
        ],
        dtype=float,
    )


def parse_params(raw: dict) -> dict:
    params = dict(DEFAULT_PARAMS)
    params.update(raw or {})
    params["seed"] = int(params["seed"])
    params["noise"] = clamp(float(params["noise"]), 0.0, 0.45)
    params["dropout"] = clamp(float(params["dropout"]), 0.0, 0.45)
    params["sensorBias"] = clamp(float(params["sensorBias"]), 0.70, 1.30)
    params["speed"] = clamp(float(params["speed"]), 0.12, 0.55)
    params["maxSteps"] = int(clamp(float(params["maxSteps"]), 30, 1200))
    params["successRadius"] = clamp(float(params["successRadius"]), 0.20, 0.85)
    params["safeMargin"] = clamp(float(params["safeMargin"]), 0.02, 0.85)
    params["plannerClearance"] = clamp(float(params["plannerClearance"]), 0.0, 0.35)
    params["plannerLookahead"] = clamp(float(params["plannerLookahead"]), 0.25, 2.20)
    params["plannerWeight"] = clamp(float(params["plannerWeight"]), 0.0, 2.50)
    params["policyWeight"] = clamp(float(params["policyWeight"]), 0.0, 1.50)
    liquid_cell = str(params.get("liquidCell", "cfc")).lower().strip()
    params["liquidCell"] = liquid_cell if liquid_cell in {"legacy", "cfc", "ltc"} else "cfc"
    params["ncpHidden"] = int(clamp(float(params["ncpHidden"]), 12, 96))
    params["ncpSparsity"] = clamp(float(params["ncpSparsity"]), 0.10, 0.90)
    params["ncpBaselineScale"] = clamp(float(params["ncpBaselineScale"]), 0.0, 1.50)
    params["ncpResidualScale"] = clamp(float(params["ncpResidualScale"]), 0.0, 1.50)
    params["ncpLearningRate"] = clamp(float(params["ncpLearningRate"]), 0.0, 0.05)
    params["makeGif"] = bool(params.get("makeGif", True))
    return params


def local_expert_scores(
    pos: np.ndarray,
    heading: float,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    speed: float,
    safe_margin: float,
) -> np.ndarray:
    current_dist = norm(goal - pos)
    scores = []
    for delta in ACTION_DELTAS:
        new_heading = wrap_angle(heading + float(delta))
        new_pos = pos + speed * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        progress = current_dist - norm(goal - new_pos)
        turn_cost = abs(float(delta)) / math.pi
        penalty = 0.0
        if clearance <= 0:
            penalty += 8.0
        if clearance < safe_margin:
            penalty += (safe_margin - clearance) * 4.0
        scores.append(progress * 3.0 - turn_cost - penalty)
    return np.array(scores, dtype=float)


def route_guided_scores_custom(
    pos: np.ndarray,
    heading: float,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    route: np.ndarray | None,
    params: dict,
) -> np.ndarray:
    if route is None or len(route) < 2:
        return local_expert_scores(pos, heading, goal, obstacles, params["speed"], params["safeMargin"])

    nearest = int(np.argmin(np.linalg.norm(route - pos, axis=1)))
    lookahead_cells = max(2, int(round(params["plannerLookahead"] / 0.20)))
    waypoint = route[min(len(route) - 1, nearest + lookahead_cells)]
    waypoint_dist = norm(waypoint - pos)
    goal_dist = norm(goal - pos)
    desired_heading = math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0])

    scores = []
    for delta in ACTION_DELTAS:
        new_heading = wrap_angle(heading + float(delta))
        new_pos = pos + params["speed"] * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        waypoint_progress = waypoint_dist - norm(waypoint - new_pos)
        goal_progress = goal_dist - norm(goal - new_pos)
        heading_error = abs(wrap_angle(desired_heading - new_heading)) / math.pi
        penalty = 0.0
        if clearance <= 0:
            penalty += 10.0
        if clearance < params["plannerClearance"]:
            penalty += (params["plannerClearance"] - clearance) * 8.0
        scores.append(4.5 * waypoint_progress + goal_progress - 0.75 * heading_error - penalty)
    return np.array(scores, dtype=float)


def supervised_safe_action_custom(
    pos: np.ndarray,
    heading: float,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    scores: np.ndarray,
    route: np.ndarray | None,
    params: dict,
) -> tuple[int, bool]:
    guided_scores = route_guided_scores_custom(pos, heading, goal, obstacles, route, params)
    combined_scores = params["policyWeight"] * scores + params["plannerWeight"] * guided_scores
    safe = []
    clearances = []
    for delta in ACTION_DELTAS:
        new_heading = wrap_angle(heading + float(delta))
        new_pos = pos + params["speed"] * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        clearances.append(clearance)
        safe.append(clearance > params["safeMargin"])
    safe = np.array(safe, dtype=bool)
    if safe.any():
        masked = np.where(safe, combined_scores, -1e9)
        chosen = int(np.argmax(masked))
        raw = int(np.argmax(scores))
        return chosen, chosen != raw
    escape_scores = np.array(clearances, dtype=float) * 3.0 + guided_scores
    return int(np.argmax(escape_scores)), True


def simulate_trace(
    controller_key: str,
    controller,
    obstacles: list[Obstacle],
    params: dict,
    supervisor: bool,
    route: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    pos = start.copy()
    heading = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
    controller.reset()

    positions = [pos.copy()]
    clearances = [obstacle_clearance(pos, obstacles)]
    override_frames = []
    status = "timeout"
    min_clearance = clearances[0]

    for step in range(params["maxSteps"]):
        true_ranges = ray_cast(pos, heading, obstacles)
        obs_ranges, uncertainty = noisy_sensor(
            true_ranges,
            rng,
            params["noise"],
            params["dropout"],
            params["sensorBias"],
        )
        x = features(pos, heading, goal, obs_ranges, uncertainty)
        action, scores = controller.act(x, pos, heading, goal, obstacles)
        corrective = None
        if supervisor:
            safe_action, did_override = supervised_safe_action_custom(pos, heading, goal, obstacles, scores, route, params)
            if did_override:
                corrective = safe_action
                action = safe_action
                override_frames.append(step)

        new_heading = wrap_angle(heading + float(ACTION_DELTAS[action]))
        new_pos = pos + params["speed"] * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        min_clearance = min(min_clearance, clearance)
        progress = norm(goal - pos) - norm(goal - new_pos)
        reward = progress * 2.0 + min(clearance, 1.0) * 0.15
        if clearance <= 0:
            reward -= 4.0
        if progress < -0.02:
            reward -= 0.25
        if corrective is None and clearance < params["safeMargin"]:
            corrective = int(np.argmax(local_expert_scores(pos, heading, goal, obstacles, params["speed"], params["safeMargin"])))
        controller.learn(action, reward, corrective)

        positions.append(new_pos.copy())
        clearances.append(clearance)
        pos = new_pos
        heading = new_heading

        if clearance <= 0:
            status = "collision"
            break
        if norm(goal - pos) < params["successRadius"]:
            status = "success"
            break

    liquid_name = getattr(controller, "display_name", "liquid")
    controller_names = {
        "fixed": "fixed policy",
        "liquid": f"{liquid_name} online",
        "supervisor": f"{liquid_name} + symbolic supervisor",
    }
    return {
        "key": controller_key,
        "name": controller_names[controller_key],
        "positions": np.vstack(positions),
        "clearances": np.array(clearances),
        "override_frames": override_frames,
        "status": status,
        "steps": len(positions) - 1,
        "min_clearance": float(min_clearance),
        "dist_goal": float(norm(goal - positions[-1])),
    }


def build_traces(
    obstacles: list[Obstacle],
    params: dict,
    controllers: list[str],
    start: np.ndarray,
    goal: np.ndarray,
    progress=None,
) -> tuple[list[dict], np.ndarray | None]:
    if progress:
        progress(12, "Politika ağırlıkları ve NCP katmanı hazırlanıyor; ilk koşu biraz sürebilir.")
    base_weights = get_base_weights()
    input_dim = base_weights.shape[0]
    if progress:
        progress(18, "Global rota planlanıyor.")
    route = plan_grid_route(obstacles, start, goal, clearance=params["plannerClearance"])
    traces = []
    specs = []
    if "fixed" in controllers:
        specs.append(("fixed", FixedPolicy(base_weights.copy()), False, params["seed"]))
    if "liquid" in controllers:
        specs.append(
            (
                "liquid",
                make_liquid_policy(
                    np.random.default_rng(params["seed"] + 1),
                    input_dim,
                    base_weights.copy(),
                    adaptive=True,
                    backend=params["liquidCell"],
                    hidden_dim=params["ncpHidden"],
                    sparsity=params["ncpSparsity"],
                    baseline_scale=params["ncpBaselineScale"],
                    residual_scale=params["ncpResidualScale"],
                    learning_rate=params["ncpLearningRate"],
                ),
                False,
                params["seed"],
            )
        )
    if "supervisor" in controllers:
        specs.append(
            (
                "supervisor",
                make_liquid_policy(
                    np.random.default_rng(params["seed"] + 2),
                    input_dim,
                    base_weights.copy(),
                    adaptive=True,
                    backend=params["liquidCell"],
                    hidden_dim=params["ncpHidden"],
                    sparsity=params["ncpSparsity"],
                    baseline_scale=params["ncpBaselineScale"],
                    residual_scale=params["ncpResidualScale"],
                    learning_rate=params["ncpLearningRate"],
                ),
                True,
                params["seed"],
            )
        )
    for index, (key, controller, supervisor, seed) in enumerate(specs):
        if progress:
            progress(22 + int((index / max(1, len(specs))) * 38), f"{key} denetleyicisi simüle ediliyor.")
        traces.append(simulate_trace(key, controller, obstacles, params, supervisor, route, start, goal, seed))
    if progress:
        progress(62, "Denetleyici izleri tamamlandı.")
    return traces, route


def draw_world(
    ax,
    obstacles: list[Obstacle],
    route: np.ndarray | None = None,
    start: np.ndarray | None = None,
    goal: np.ndarray | None = None,
) -> None:
    start = DEFAULT_START if start is None else start
    goal = DEFAULT_GOAL if goal is None else goal
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if route is not None and len(route) > 1:
        ax.plot(route[:, 0], route[:, 1], color="#2b7a78", lw=1.5, ls="--", alpha=0.8)
    for obs in obstacles:
        ax.add_patch(plt.Circle((obs.x, obs.y), obs.r + ROBOT_RADIUS, color="#cc4b4b", alpha=0.20))
        ax.add_patch(plt.Circle((obs.x, obs.y), obs.r, color="#9f2f3f", alpha=0.72))
    ax.scatter([start[0]], [start[1]], marker="o", s=55, color="#222222", zorder=4)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=180, color="#16837a", zorder=4)


def render_map_png(
    path: Path,
    map_title: str,
    obstacles: list[Obstacle],
    route: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2), constrained_layout=True)
    draw_world(ax, obstacles, route, start, goal)
    ax.set_title(f"{map_title} - custom map")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_simulation(
    path_base: Path,
    map_title: str,
    obstacles: list[Obstacle],
    traces: list[dict],
    route: np.ndarray | None,
    make_gif: bool,
    start: np.ndarray,
    goal: np.ndarray,
    progress=None,
) -> dict:
    colors = {
        "fixed": "#555555",
        "liquid": "#b64252",
        "supervisor": "#16837a",
    }
    max_frames = max(len(t["positions"]) for t in traces)
    fig_width = max(4.2, 4.1 * len(traces))
    fig, axes = plt.subplots(1, len(traces), figsize=(fig_width, 4.1), constrained_layout=True)
    if len(traces) == 1:
        axes = [axes]

    artists = []
    for ax, trace in zip(axes, traces):
        draw_world(ax, obstacles, route, start, goal)
        color = colors[trace["key"]]
        ax.set_title(f"{trace['name']}\nstatus: pending")
        (line,) = ax.plot([], [], color=color, lw=2.3)
        robot = plt.Circle((start[0], start[1]), ROBOT_RADIUS, color=color, alpha=0.95, zorder=5)
        ax.add_patch(robot)
        text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=8)
        artists.append((line, robot, text, ax))

    fig.suptitle(f"H6 custom map simulation - {map_title}", fontsize=12)

    def update(frame: int):
        changed = []
        for trace, (line, robot, text, ax) in zip(traces, artists):
            positions = trace["positions"]
            idx = min(frame, len(positions) - 1)
            xy = positions[: idx + 1]
            line.set_data(xy[:, 0], xy[:, 1])
            robot.center = tuple(positions[idx])
            clearance = float(trace["clearances"][idx])
            status = trace["status"] if frame >= len(positions) - 1 else "running"
            text.set_text(
                f"step={idx}\nclearance={clearance:.2f}\noverrides={len(trace['override_frames'])}\n{status}"
            )
            ax.set_title(f"{trace['name']}\nstatus: {status}")
            changed.extend([line, robot, text, ax.title])
        return changed

    final_png = path_base.with_name(f"{path_base.name}_final.png")
    if progress:
        progress(72, "Final PNG çiziliyor.")
    update(max_frames)
    fig.savefig(final_png, dpi=180)

    gif_path = None
    if make_gif:
        if progress:
            progress(82, "GIF oluşturuluyor.")
        stride = max(1, max_frames // 90)
        frames = list(range(0, max_frames + 1, stride))
        if frames[-1] != max_frames:
            frames.append(max_frames)
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
        gif_path = path_base.with_name(f"{path_base.name}_simulation.gif")
        ani.save(gif_path, writer=animation.PillowWriter(fps=8))
        if progress:
            progress(95, "GIF kaydedildi.")
    plt.close(fig)
    return {"gif": gif_path, "final_png": final_png}


def public_output_url(path: Path | None) -> str | None:
    if path is None:
        return None
    return f"/outputs/{quote(path.name)}"


def write_outputs(payload: dict, progress=None) -> dict:
    if progress:
        progress(5, "Harita ve parametreler okunuyor.")
    map_name = str(payload.get("name") or "custom_map").strip() or "custom_map"
    slug = unique_slug(sanitize_name(map_name))
    obstacles = parse_obstacles(payload.get("obstacles") or [])
    start = parse_point(payload.get("start"), DEFAULT_START)
    goal = parse_point(payload.get("goal"), DEFAULT_GOAL)
    params = parse_params(payload.get("params") or {})
    controllers = [c for c in (payload.get("controllers") or ["fixed", "liquid", "supervisor"]) if c in {"fixed", "liquid", "supervisor"}]
    if not controllers:
        controllers = ["supervisor"]

    traces, route = build_traces(obstacles, params, controllers, start, goal, progress=progress)
    base_path = OUTPUT_DIR / slug
    map_png = OUTPUT_DIR / f"{slug}_map.png"
    if progress:
        progress(67, "Harita PNG çiziliyor.")
    render_map_png(map_png, map_name, obstacles, route, start, goal)
    rendered = render_simulation(base_path, map_name, obstacles, traces, route, params["makeGif"], start, goal, progress=progress)
    config_path = OUTPUT_DIR / f"{slug}_config.json"
    if progress:
        progress(97, "Config JSON yazılıyor.")
    config_path.write_text(
        json.dumps(
            {
                "name": map_name,
                "slug": slug,
                "start": {"x": float(start[0]), "y": float(start[1])},
                "goal": {"x": float(goal[0]), "y": float(goal[1])},
                "obstacles": [{"x": o.x, "y": o.y, "r": o.r} for o in obstacles],
                "params": params,
                "controllers": controllers,
                "route_found": route is not None,
                "traces": [
                    {
                        "controller": t["key"],
                        "status": t["status"],
                        "steps": t["steps"],
                        "min_clearance": t["min_clearance"],
                        "dist_goal": t["dist_goal"],
                        "overrides": len(t["override_frames"]),
                    }
                    for t in traces
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "name": map_name,
        "slug": slug,
        "route_found": route is not None,
        "files": {
            "map_png": public_output_url(map_png),
            "final_png": public_output_url(rendered["final_png"]),
            "gif": public_output_url(rendered["gif"]),
            "config_json": public_output_url(config_path),
        },
        "traces": [
            {
                "controller": t["name"],
                "status": t["status"],
                "steps": t["steps"],
                "min_clearance": round(t["min_clearance"], 3),
                "dist_goal": round(t["dist_goal"], 3),
                "overrides": len(t["override_frames"]),
            }
            for t in traces
        ],
    }


def map_document_from_payload(payload: dict) -> dict:
    map_name = str(payload.get("name") or "custom_map").strip() or "custom_map"
    slug = sanitize_name(map_name)
    obstacles = parse_obstacles(payload.get("obstacles") or [])
    start = parse_point(payload.get("start"), DEFAULT_START)
    goal = parse_point(payload.get("goal"), DEFAULT_GOAL)
    params = parse_params(payload.get("params") or {})
    controllers = [c for c in (payload.get("controllers") or ["fixed", "liquid", "supervisor"]) if c in {"fixed", "liquid", "supervisor"}]
    if not controllers:
        controllers = ["supervisor"]
    return {
        "name": map_name,
        "slug": slug,
        "start": {"x": float(start[0]), "y": float(start[1])},
        "goal": {"x": float(goal[0]), "y": float(goal[1])},
        "obstacles": [{"x": o.x, "y": o.y, "r": o.r} for o in obstacles],
        "params": params,
        "controllers": controllers,
    }


def saved_map_path(slug: str) -> Path:
    safe_slug = sanitize_name(slug)
    if not safe_slug:
        raise ValueError("Invalid map name.")
    return MAP_DIR / f"{safe_slug}.json"


def save_map(payload: dict) -> dict:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    document = map_document_from_payload(payload)
    path = saved_map_path(document["slug"])
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "name": document["name"], "slug": document["slug"], "path": str(path)}


def list_saved_maps() -> dict:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    maps = []
    for path in sorted(MAP_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        maps.append(
            {
                "slug": sanitize_name(data.get("slug") or path.stem),
                "name": data.get("name") or path.stem,
                "obstacles": len(data.get("obstacles") or []),
                "updated": path.stat().st_mtime,
            }
        )
    return {"ok": True, "maps": maps}


def load_saved_map(slug: str) -> dict:
    path = saved_map_path(slug)
    if not path.exists():
        return {"ok": False, "error": "Map not found."}
    return {"ok": True, "map": json.loads(path.read_text(encoding="utf-8"))}


def delete_saved_map(slug: str) -> dict:
    path = saved_map_path(slug)
    if not path.exists():
        return {"ok": False, "error": "Map not found."}
    path.unlink()
    return {"ok": True, "slug": sanitize_name(slug)}


class CustomMapHandler(BaseHTTPRequestHandler):
    server_version = "H6CustomMapServer/1.0"

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self.send_json({"ok": True})
            return
        if path == "/api/maps":
            self.send_json(list_saved_maps())
            return
        if path.startswith("/api/jobs/"):
            job_id = unquote(path.removeprefix("/api/jobs/"))
            self.send_json(get_job(job_id))
            return
        if path.startswith("/api/maps/"):
            slug = unquote(path.removeprefix("/api/maps/"))
            self.send_json(load_saved_map(slug))
            return
        if path.startswith("/outputs/"):
            name = Path(unquote(path.removeprefix("/outputs/"))).name
            self.serve_file(OUTPUT_DIR / name)
            return
        if path in {"/", "/index.html"}:
            self.serve_file(UI_DIR / "index.html")
            return
        static_path = UI_DIR / Path(path.lstrip("/")).name
        self.serve_file(static_path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/simulate", "/api/simulate/start", "/api/maps"}:
            self.send_json({"ok": False, "error": "Unknown endpoint."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if parsed.path == "/api/maps":
                result = save_map(payload)
            elif parsed.path == "/api/simulate/start":
                result = start_simulation_job(payload)
            else:
                result = write_outputs(payload)
            self.send_json(result)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"ok": False, "error": str(exc)}, 500)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/maps/"):
            self.send_json({"ok": False, "error": "Unknown endpoint."}, 404)
            return
        try:
            slug = unquote(parsed.path.removeprefix("/api/maps/"))
            self.send_json(delete_saved_map(slug))
        except Exception as exc:  # noqa: BLE001
            self.send_json({"ok": False, "error": str(exc)}, 500)

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[custom-map-ui] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), CustomMapHandler)
    print(f"Serving H6 custom map UI at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
