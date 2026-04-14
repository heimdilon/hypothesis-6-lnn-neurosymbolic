from __future__ import annotations

import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from ncps.torch import CfC, LTC
    from ncps.wirings import AutoNCP

    TORCH_NCPS_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    CfC = None
    LTC = None
    AutoNCP = None
    TORCH_NCPS_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


ACTION_DELTAS = np.deg2rad(np.array([-70, -40, -18, 0, 18, 40, 70], dtype=float))
N_RAYS = 12
MAX_RANGE = 4.0
WORLD_SIZE = 10.0
ROBOT_RADIUS = 0.18
SAFE_MARGIN = 0.35
MAX_STEPS = 120
DEFAULT_SCENARIOS = ["train_like", "shifted_clutter", "narrow_gate", "u_trap"]
HARD_SCENARIOS = ["zigzag_corridor", "dense_maze", "deceptive_u_trap", "sensor_shadow", "labyrinth_maze"]
PLANNER_RESOLUTION = 0.20
PLANNER_CLEARANCE = 0.04
PLANNER_LOOKAHEAD = 0.85
ROUTE_CACHE: dict[tuple, np.ndarray | None] = {}


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    r: float


@dataclass
class EpisodeResult:
    controller: str
    scenario: str
    seed: int
    success: int
    collision: int
    steps: int
    path_length: float
    min_clearance: float
    near_misses: int
    overrides: int
    adaptation_lag: int
    mean_progress: float


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def obstacle_clearance(pos: np.ndarray, obstacles: Iterable[Obstacle]) -> float:
    clearances = [norm(pos - np.array([o.x, o.y])) - o.r - ROBOT_RADIUS for o in obstacles]
    wall = min(pos[0], pos[1], WORLD_SIZE - pos[0], WORLD_SIZE - pos[1]) - ROBOT_RADIUS
    return float(min([wall, *clearances]))


def collides(pos: np.ndarray, obstacles: Iterable[Obstacle]) -> bool:
    return obstacle_clearance(pos, obstacles) <= 0.0


def obstacle_cache_key(obstacles: Iterable[Obstacle]) -> tuple[tuple[float, float, float], ...]:
    return tuple((round(o.x, 3), round(o.y, 3), round(o.r, 3)) for o in obstacles)


def nearest_free_cell(
    pos: np.ndarray,
    free: np.ndarray,
    resolution: float,
) -> tuple[int, int] | None:
    n = free.shape[0]
    ix = int(np.clip(round(float(pos[0]) / resolution), 0, n - 1))
    iy = int(np.clip(round(float(pos[1]) / resolution), 0, n - 1))
    best = None
    best_dist = float("inf")
    for radius in range(n):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                cx = ix + dx
                cy = iy + dy
                if 0 <= cx < n and 0 <= cy < n and free[cx, cy]:
                    candidate = np.array([cx * resolution, cy * resolution])
                    dist = norm(candidate - pos)
                    if dist < best_dist:
                        best = (cx, cy)
                        best_dist = dist
        if best is not None:
            return best
    return None


def plan_grid_route(
    obstacles: list[Obstacle],
    start: np.ndarray,
    goal: np.ndarray,
    resolution: float = PLANNER_RESOLUTION,
    clearance: float = PLANNER_CLEARANCE,
) -> np.ndarray | None:
    key = (
        obstacle_cache_key(obstacles),
        tuple(np.round(start, 3)),
        tuple(np.round(goal, 3)),
        round(resolution, 3),
        round(clearance, 3),
    )
    if key in ROUTE_CACHE:
        return ROUTE_CACHE[key]

    n = int(round(WORLD_SIZE / resolution)) + 1
    clearance_grid = np.empty((n, n), dtype=float)
    free = np.zeros((n, n), dtype=bool)
    for ix in range(n):
        for iy in range(n):
            p = np.array([ix * resolution, iy * resolution])
            c = obstacle_clearance(p, obstacles)
            clearance_grid[ix, iy] = c
            free[ix, iy] = c > clearance

    start_cell = nearest_free_cell(start, free, resolution)
    goal_cell = nearest_free_cell(goal, free, resolution)
    if start_cell is None or goal_cell is None:
        ROUTE_CACHE[key] = None
        return None

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start_cell: 0.0}

    def heuristic(cell: tuple[int, int]) -> float:
        return resolution * math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            route = np.array([[ix * resolution, iy * resolution] for ix, iy in cells], dtype=float)
            ROUTE_CACHE[key] = route
            return route

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or not free[nxt]:
                continue
            move_cost = resolution * math.hypot(dx, dy)
            clearance_cost = max(0.0, 0.35 - float(clearance_grid[nxt])) * 1.25
            tentative = g_score[current] + move_cost + clearance_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nxt), nxt))

    ROUTE_CACHE[key] = None
    return None


def ray_cast(pos: np.ndarray, heading: float, obstacles: list[Obstacle]) -> np.ndarray:
    readings = []
    for rel in np.linspace(-math.pi, math.pi, N_RAYS, endpoint=False):
        theta = heading + rel
        direction = np.array([math.cos(theta), math.sin(theta)])
        hit = MAX_RANGE
        for step in np.linspace(0.10, MAX_RANGE, 42):
            p = pos + direction * step
            if p[0] <= 0 or p[0] >= WORLD_SIZE or p[1] <= 0 or p[1] >= WORLD_SIZE:
                hit = min(hit, float(step))
                break
            if any(norm(p - np.array([o.x, o.y])) <= o.r + ROBOT_RADIUS for o in obstacles):
                hit = min(hit, float(step))
                break
        readings.append(hit)
    return np.array(readings, dtype=float)


def noisy_sensor(
    true_ranges: np.ndarray,
    rng: np.random.Generator,
    noise: float,
    dropout: float,
    scale_bias: float,
) -> tuple[np.ndarray, float]:
    observed = true_ranges * scale_bias + rng.normal(0.0, noise, size=true_ranges.shape)
    observed = np.clip(observed, 0.05, MAX_RANGE)
    mask = rng.random(size=true_ranges.shape) < dropout
    observed[mask] = MAX_RANGE
    uncertainty = float(np.clip(noise / 0.45 + dropout + abs(scale_bias - 1.0), 0.0, 2.0))
    return observed, uncertainty


def features(pos: np.ndarray, heading: float, goal: np.ndarray, ranges: np.ndarray, uncertainty: float) -> np.ndarray:
    delta = goal - pos
    dist = norm(delta)
    bearing = wrap_angle(math.atan2(delta[1], delta[0]) - heading)
    normalized_ranges = ranges / MAX_RANGE
    return np.concatenate(
        [
            normalized_ranges,
            np.array(
                [
                    math.sin(bearing),
                    math.cos(bearing),
                    min(dist / 12.0, 1.0),
                    min(ranges.min() / MAX_RANGE, 1.0),
                    uncertainty / 2.0,
                    1.0,
                ],
                dtype=float,
            ),
        ]
    )


def expert_scores(pos: np.ndarray, heading: float, goal: np.ndarray, obstacles: list[Obstacle]) -> np.ndarray:
    current_dist = norm(goal - pos)
    scores = []
    for delta in ACTION_DELTAS:
        new_heading = wrap_angle(heading + float(delta))
        step = 0.32
        new_pos = pos + step * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        progress = current_dist - norm(goal - new_pos)
        turn_cost = abs(float(delta)) / math.pi
        penalty = 0.0
        if clearance <= 0:
            penalty += 8.0
        if clearance < SAFE_MARGIN:
            penalty += (SAFE_MARGIN - clearance) * 4.0
        scores.append(progress * 3.0 - turn_cost - penalty)
    return np.array(scores, dtype=float)


def route_guided_scores(pos: np.ndarray, heading: float, goal: np.ndarray, obstacles: list[Obstacle]) -> np.ndarray:
    route = plan_grid_route(obstacles, np.array([1.0, 1.0]), goal)
    if route is None or len(route) < 2:
        return expert_scores(pos, heading, goal, obstacles)

    nearest = int(np.argmin(np.linalg.norm(route - pos, axis=1)))
    lookahead_cells = max(2, int(round(PLANNER_LOOKAHEAD / PLANNER_RESOLUTION)))
    waypoint = route[min(len(route) - 1, nearest + lookahead_cells)]
    waypoint_dist = norm(waypoint - pos)
    goal_dist = norm(goal - pos)
    desired_heading = math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0])

    scores = []
    for delta in ACTION_DELTAS:
        new_heading = wrap_angle(heading + float(delta))
        new_pos = pos + 0.32 * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        waypoint_progress = waypoint_dist - norm(waypoint - new_pos)
        goal_progress = goal_dist - norm(goal - new_pos)
        heading_error = abs(wrap_angle(desired_heading - new_heading)) / math.pi
        penalty = 0.0
        if clearance <= 0:
            penalty += 10.0
        if clearance < PLANNER_CLEARANCE:
            penalty += (PLANNER_CLEARANCE - clearance) * 8.0
        scores.append(4.5 * waypoint_progress + 1.0 * goal_progress - 0.75 * heading_error - penalty)
    return np.array(scores, dtype=float)


def train_fixed_policy(rng: np.random.Generator, n_samples: int = 900) -> np.ndarray:
    xs = []
    ys = []
    train_obstacles = scenario_obstacles("train_like")
    goal = np.array([8.8, 8.7])
    for _ in range(n_samples):
        pos = rng.uniform(1.0, 9.0, size=2)
        if collides(pos, train_obstacles):
            continue
        heading = rng.uniform(-math.pi, math.pi)
        true_ranges = ray_cast(pos, heading, train_obstacles)
        obs_ranges, uncertainty = noisy_sensor(true_ranges, rng, noise=0.06, dropout=0.02, scale_bias=1.0)
        x = features(pos, heading, goal, obs_ranges, uncertainty)
        y = expert_scores(pos, heading, goal, train_obstacles)
        xs.append(x)
        ys.append(y)
    xmat = np.vstack(xs)
    ymat = np.vstack(ys)
    reg = 1e-2 * np.eye(xmat.shape[1])
    return np.linalg.solve(xmat.T @ xmat + reg, xmat.T @ ymat)


class FixedPolicy:
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def reset(self) -> None:
        pass

    def act(self, x: np.ndarray, *_args) -> tuple[int, np.ndarray]:
        scores = x @ self.weights
        return int(np.argmax(scores)), scores

    def learn(self, *_args) -> None:
        pass


class LiquidPolicy:
    def __init__(self, rng: np.random.Generator, input_dim: int, base_weights: np.ndarray, adaptive: bool):
        self.input_dim = input_dim
        self.hidden_dim = 24
        self.adaptive = adaptive
        self.a = rng.normal(0.0, 0.45, size=(input_dim, self.hidden_dim))
        self.b = rng.normal(0.0, 0.18, size=(self.hidden_dim, self.hidden_dim))
        # Start from the same deployed policy as the fixed controller. The liquid
        # state only changes behavior after online feedback, otherwise the
        # comparison confounds adaptation with random initialization.
        self.w = np.vstack([base_weights, np.zeros((self.hidden_dim, len(ACTION_DELTAS)))])
        self.h = np.zeros(self.hidden_dim)
        self.trace = np.zeros(input_dim + self.hidden_dim)

    def reset(self) -> None:
        self.h = np.zeros(self.hidden_dim)
        self.trace = np.zeros_like(self.trace)

    def liquid_features(self, x: np.ndarray) -> np.ndarray:
        uncertainty = float(x[-2]) * 2.0
        tau = 1.0 + 2.8 * uncertainty
        alpha = 1.0 / tau
        drive = np.tanh(x @ self.a + self.h @ self.b)
        self.h = (1.0 - alpha) * self.h + alpha * drive
        z = np.concatenate([x, self.h])
        self.trace = 0.82 * self.trace + 0.18 * z
        return z

    def act(self, x: np.ndarray, *_args) -> tuple[int, np.ndarray]:
        z = self.liquid_features(x)
        scores = z @ self.w
        return int(np.argmax(scores)), scores

    def learn(self, action: int, reward: float, corrective_action: int | None = None) -> None:
        if not self.adaptive:
            return
        z = self.trace
        lr = 0.018
        decay = 0.001
        self.w *= 1.0 - decay
        if corrective_action is not None and corrective_action != action:
            self.w[:, action] -= lr * z
            self.w[:, corrective_action] += 1.6 * lr * z
            return
        # Positive reward is deliberately weak: the unsafe failure mode in this
        # hypothesis is over-adaptation from sparse rewards.
        scale = 0.30 if reward > 0 else 1.0
        self.w[:, action] += scale * lr * float(np.clip(reward, -1.0, 1.0)) * z


class NCPPolicy:
    """Official MIT ncps CfC/LTC residual policy with online updates."""

    def __init__(
        self,
        rng: np.random.Generator,
        input_dim: int,
        base_weights: np.ndarray,
        adaptive: bool,
        cell_type: str = "cfc",
        hidden_dim: int = 32,
        sparsity: float = 0.50,
        baseline_scale: float = 1.0,
        residual_scale: float = 0.35,
        learning_rate: float = 0.006,
    ):
        if not TORCH_NCPS_AVAILABLE:
            raise RuntimeError("ncps and torch are required for the real NCP policy. Install with: pip install ncps torch")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(max(12, hidden_dim))
        self.adaptive = adaptive
        self.cell_type = cell_type.lower().strip()
        self.baseline_scale = float(baseline_scale)
        self.residual_scale = float(residual_scale)
        self.learning_rate = float(learning_rate)
        self.device = torch.device("cpu")
        self.seed = int(rng.integers(1, 2_000_000_000))
        torch.manual_seed(self.seed)

        wiring = AutoNCP(
            self.hidden_dim,
            len(ACTION_DELTAS),
            sparsity_level=float(np.clip(sparsity, 0.10, 0.90)),
            seed=self.seed,
        )
        if self.cell_type == "cfc":
            self.net = CfC(self.input_dim, wiring, return_sequences=True, batch_first=True)
            self.display_name = "MIT ncps CfC"
        elif self.cell_type == "ltc":
            self.net = LTC(self.input_dim, wiring, return_sequences=True, batch_first=True)
            self.display_name = "MIT ncps LTC"
        else:
            raise ValueError(f"Unknown NCP cell type: {cell_type}")

        self.net.to(self.device)
        self.base_weights = torch.as_tensor(base_weights, dtype=torch.float32, device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate) if self.adaptive else None
        self.hx = None
        self.last_scores = None

    def reset(self) -> None:
        self.hx = None
        self.last_scores = None

    def _detach_state(self, state):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(s.detach() for s in state)
        return state.detach()

    def act(self, x: np.ndarray, *_args) -> tuple[int, np.ndarray]:
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, 1, self.input_dim)
        timespan = torch.tensor([[1.0 + 2.0 * float(x[-2])]], dtype=torch.float32, device=self.device)
        self.hx = self._detach_state(self.hx)
        with torch.set_grad_enabled(self.adaptive):
            output, next_hx = self.net(xt, self.hx, timespans=timespan)
            ncp_scores = output[:, -1, :]
            baseline_scores = xt[:, -1, :] @ self.base_weights
            scores = self.baseline_scale * baseline_scores + self.residual_scale * ncp_scores

        self.hx = self._detach_state(next_hx)
        self.last_scores = scores.squeeze(0) if self.adaptive else None
        scores_np = scores.detach().cpu().numpy().reshape(-1)
        return int(np.argmax(scores_np)), scores_np

    def learn(self, action: int, reward: float, corrective_action: int | None = None) -> None:
        if not self.adaptive or self.optimizer is None or self.last_scores is None:
            return

        scores = self.last_scores
        if corrective_action is not None and corrective_action != action:
            target = torch.tensor([int(corrective_action)], dtype=torch.long, device=self.device)
            loss = F.cross_entropy(scores.unsqueeze(0), target)
        else:
            bounded_reward = float(np.clip(reward, -1.0, 1.0))
            chosen = scores[int(action)]
            if bounded_reward >= 0:
                loss = -0.06 * bounded_reward * chosen
            else:
                loss = 0.18 * (-bounded_reward) * chosen

        if not torch.isfinite(loss):
            self.last_scores = None
            return
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.last_scores = None


def make_liquid_policy(
    rng: np.random.Generator,
    input_dim: int,
    base_weights: np.ndarray,
    adaptive: bool = True,
    backend: str = "cfc",
    hidden_dim: int = 32,
    sparsity: float = 0.50,
    baseline_scale: float = 1.0,
    residual_scale: float = 0.35,
    learning_rate: float = 0.006,
):
    backend = (backend or "cfc").lower().strip()
    if backend in {"legacy", "prototype", "manual"}:
        return LiquidPolicy(rng, input_dim, base_weights, adaptive=adaptive)
    if backend in {"cfc", "ltc"}:
        return NCPPolicy(
            rng,
            input_dim,
            base_weights,
            adaptive=adaptive,
            cell_type=backend,
            hidden_dim=hidden_dim,
            sparsity=sparsity,
            baseline_scale=baseline_scale,
            residual_scale=residual_scale,
            learning_rate=learning_rate,
        )
    raise ValueError(f"Unknown liquid backend: {backend}")


def vertical_wall(
    x: float,
    y_min: float,
    y_max: float,
    gap_center: float,
    gap_width: float,
    r: float = 0.30,
    spacing: float = 0.56,
) -> list[Obstacle]:
    obstacles = []
    for y in np.arange(y_min, y_max + 1e-6, spacing):
        if abs(float(y) - gap_center) > gap_width / 2.0:
            obstacles.append(Obstacle(x, float(y), r))
    return obstacles


def horizontal_wall(
    y: float,
    x_min: float,
    x_max: float,
    gap_center: float,
    gap_width: float,
    r: float = 0.30,
    spacing: float = 0.56,
) -> list[Obstacle]:
    obstacles = []
    for x in np.arange(x_min, x_max + 1e-6, spacing):
        if abs(float(x) - gap_center) > gap_width / 2.0:
            obstacles.append(Obstacle(float(x), y, r))
    return obstacles


def scenario_obstacles(name: str) -> list[Obstacle]:
    if name == "train_like":
        return [
            Obstacle(3.0, 3.2, 0.55),
            Obstacle(5.2, 4.6, 0.70),
            Obstacle(6.8, 6.3, 0.55),
            Obstacle(4.2, 7.1, 0.45),
        ]
    if name == "shifted_clutter":
        return [
            Obstacle(3.1, 2.5, 0.60),
            Obstacle(4.2, 4.2, 0.60),
            Obstacle(5.2, 5.4, 0.65),
            Obstacle(6.2, 6.5, 0.60),
            Obstacle(6.8, 3.8, 0.45),
            Obstacle(3.5, 6.4, 0.50),
        ]
    if name == "narrow_gate":
        return [
            Obstacle(4.6, 4.4, 0.72),
            Obstacle(5.4, 5.6, 0.72),
            Obstacle(4.1, 5.9, 0.55),
            Obstacle(6.0, 4.0, 0.55),
            Obstacle(6.8, 6.8, 0.50),
        ]
    if name == "u_trap":
        return [
            Obstacle(4.6, 4.0, 0.55),
            Obstacle(5.4, 4.0, 0.55),
            Obstacle(4.2, 4.8, 0.55),
            Obstacle(5.8, 4.8, 0.55),
            Obstacle(4.2, 5.6, 0.55),
            Obstacle(5.8, 5.6, 0.55),
            Obstacle(7.2, 7.0, 0.50),
        ]
    if name == "zigzag_corridor":
        return [
            *vertical_wall(2.8, 0.9, 8.8, gap_center=2.6, gap_width=1.25),
            *vertical_wall(5.0, 1.1, 9.1, gap_center=6.6, gap_width=1.20),
            *vertical_wall(7.1, 1.0, 8.6, gap_center=4.2, gap_width=1.15),
            Obstacle(4.0, 3.8, 0.38),
            Obstacle(6.0, 7.7, 0.35),
        ]
    if name == "dense_maze":
        return [
            *horizontal_wall(2.7, 0.9, 8.7, gap_center=7.4, gap_width=1.15),
            *horizontal_wall(4.8, 1.2, 9.0, gap_center=2.4, gap_width=1.10),
            *horizontal_wall(6.8, 0.9, 8.8, gap_center=7.5, gap_width=1.15),
            Obstacle(3.6, 3.8, 0.42),
            Obstacle(5.5, 5.8, 0.42),
            Obstacle(7.8, 4.1, 0.35),
        ]
    if name == "deceptive_u_trap":
        return [
            Obstacle(4.4, 4.0, 0.50),
            Obstacle(5.1, 4.0, 0.50),
            Obstacle(5.8, 4.0, 0.50),
            Obstacle(4.1, 4.7, 0.50),
            Obstacle(4.1, 5.4, 0.50),
            Obstacle(5.9, 4.7, 0.50),
            Obstacle(5.9, 5.4, 0.50),
            Obstacle(4.8, 6.0, 0.46),
            Obstacle(6.9, 5.9, 0.48),
            Obstacle(7.2, 7.1, 0.40),
            Obstacle(3.3, 6.7, 0.36),
        ]
    if name == "sensor_shadow":
        return [
            *vertical_wall(3.3, 0.9, 8.8, gap_center=3.4, gap_width=1.0, r=0.28),
            *vertical_wall(5.9, 1.0, 9.0, gap_center=6.0, gap_width=1.0, r=0.28),
            Obstacle(4.2, 4.5, 0.55),
            Obstacle(4.7, 5.3, 0.45),
            Obstacle(6.7, 4.5, 0.45),
            Obstacle(7.3, 6.3, 0.45),
            Obstacle(2.2, 6.8, 0.35),
        ]
    if name == "labyrinth_maze":
        return [
            *horizontal_wall(2.35, 0.9, 9.0, gap_center=8.0, gap_width=1.65, r=0.28, spacing=0.52),
            *horizontal_wall(4.15, 0.9, 8.8, gap_center=2.0, gap_width=1.65, r=0.28, spacing=0.52),
            *horizontal_wall(5.95, 1.1, 9.0, gap_center=8.0, gap_width=1.65, r=0.28, spacing=0.52),
            *horizontal_wall(7.70, 0.9, 8.5, gap_center=2.2, gap_width=1.65, r=0.28, spacing=0.52),
            *vertical_wall(3.55, 2.75, 3.75, gap_center=10.0, gap_width=0.0, r=0.24, spacing=0.48),
            *vertical_wall(6.35, 4.55, 5.55, gap_center=10.0, gap_width=0.0, r=0.24, spacing=0.48),
            *vertical_wall(4.85, 6.35, 7.30, gap_center=10.0, gap_width=0.0, r=0.24, spacing=0.48),
            Obstacle(5.1, 2.95, 0.24),
            Obstacle(3.2, 5.25, 0.24),
            Obstacle(6.6, 7.05, 0.24),
        ]
    raise ValueError(name)


def scenario_sensor_params(name: str) -> tuple[float, float, float]:
    if name == "train_like":
        return 0.06, 0.02, 1.0
    if name == "shifted_clutter":
        return 0.13, 0.08, 1.0
    if name == "narrow_gate":
        return 0.16, 0.12, 0.88
    if name == "u_trap":
        return 0.20, 0.16, 1.12
    if name == "zigzag_corridor":
        return 0.18, 0.12, 0.94
    if name == "dense_maze":
        return 0.20, 0.15, 1.05
    if name == "deceptive_u_trap":
        return 0.22, 0.18, 1.10
    if name == "sensor_shadow":
        return 0.25, 0.24, 0.86
    if name == "labyrinth_maze":
        return 0.18, 0.12, 0.96
    raise ValueError(name)


def supervised_safe_action(
    pos: np.ndarray,
    heading: float,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    scores: np.ndarray,
) -> tuple[int, bool]:
    guided_scores = route_guided_scores(pos, heading, goal, obstacles)
    combined_scores = 0.35 * scores + guided_scores
    safe = []
    clearances = []
    for i, delta in enumerate(ACTION_DELTAS):
        new_heading = wrap_angle(heading + float(delta))
        new_pos = pos + 0.32 * np.array([math.cos(new_heading), math.sin(new_heading)])
        clearance = obstacle_clearance(new_pos, obstacles)
        clearances.append(clearance)
        safe.append(clearance > SAFE_MARGIN)
    safe = np.array(safe, dtype=bool)
    if safe.any():
        masked = np.where(safe, combined_scores, -1e9)
        chosen = int(np.argmax(masked))
        raw = int(np.argmax(scores))
        return chosen, chosen != raw
    clearance_scores = np.array(clearances, dtype=float) * 3.0 + guided_scores
    return int(np.argmax(clearance_scores)), True


def run_episode(
    controller_name: str,
    controller,
    scenario: str,
    seed: int,
    supervisor: bool,
) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    obstacles = scenario_obstacles(scenario)
    noise, dropout, bias = scenario_sensor_params(scenario)
    start = np.array([1.0, 1.0]) + rng.normal(0, 0.06, size=2)
    goal = np.array([8.8, 8.7])
    pos = start.copy()
    heading = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
    controller.reset()

    path_length = 0.0
    near_misses = 0
    overrides = 0
    min_clearance = 99.0
    progress_history = []
    adaptation_lag = MAX_STEPS
    stable_count = 0
    prev_dist = norm(goal - pos)

    for step in range(MAX_STEPS):
        true_ranges = ray_cast(pos, heading, obstacles)
        obs_ranges, uncertainty = noisy_sensor(true_ranges, rng, noise, dropout, bias)
        x = features(pos, heading, goal, obs_ranges, uncertainty)
        action, scores = controller.act(x, pos, heading, goal, obstacles)
        corrective = None
        did_override = False
        if supervisor:
            safe_action, did_override = supervised_safe_action(pos, heading, goal, obstacles, scores)
            if did_override:
                corrective = safe_action
                action = safe_action
                overrides += 1

        new_heading = wrap_angle(heading + float(ACTION_DELTAS[action]))
        speed = 0.32
        new_pos = pos + speed * np.array([math.cos(new_heading), math.sin(new_heading)])
        path_length += norm(new_pos - pos)
        clearance = obstacle_clearance(new_pos, obstacles)
        min_clearance = min(min_clearance, clearance)
        new_dist = norm(goal - new_pos)
        progress = prev_dist - new_dist
        progress_history.append(progress)
        if clearance < SAFE_MARGIN:
            near_misses += 1

        reward = progress * 2.0 + min(clearance, 1.0) * 0.15
        if clearance <= 0:
            reward -= 4.0
        if progress < -0.02:
            reward -= 0.25
        controller.learn(action, reward, corrective)

        if clearance <= 0:
            return EpisodeResult(
                controller_name,
                scenario,
                seed,
                0,
                1,
                step + 1,
                path_length,
                min_clearance,
                near_misses,
                overrides,
                adaptation_lag,
                float(np.mean(progress_history)),
            )

        pos = new_pos
        heading = new_heading
        prev_dist = new_dist

        if progress > 0.015 and clearance > SAFE_MARGIN:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= 8 and adaptation_lag == MAX_STEPS:
            adaptation_lag = step + 1

        if new_dist < 0.45:
            return EpisodeResult(
                controller_name,
                scenario,
                seed,
                1,
                0,
                step + 1,
                path_length,
                min_clearance,
                near_misses,
                overrides,
                adaptation_lag,
                float(np.mean(progress_history)),
            )

    return EpisodeResult(
        controller_name,
        scenario,
        seed,
        0,
        0,
        MAX_STEPS,
        path_length,
        min_clearance,
        near_misses,
        overrides,
        adaptation_lag,
        float(np.mean(progress_history)),
    )


def aggregate(results: list[EpisodeResult]) -> list[dict[str, float | str]]:
    rows = []
    keys = sorted({(r.controller, r.scenario) for r in results})
    for controller, scenario in keys:
        group = [r for r in results if r.controller == controller and r.scenario == scenario]
        rows.append(
            {
                "controller": controller,
                "scenario": scenario,
                "n": len(group),
                "success_rate": np.mean([r.success for r in group]),
                "collision_rate": np.mean([r.collision for r in group]),
                "mean_steps": np.mean([r.steps for r in group]),
                "mean_path_length": np.mean([r.path_length for r in group]),
                "mean_min_clearance": np.mean([r.min_clearance for r in group]),
                "mean_near_misses": np.mean([r.near_misses for r in group]),
                "mean_overrides": np.mean([r.overrides for r in group]),
                "mean_adaptation_lag": np.mean([r.adaptation_lag for r in group]),
                "mean_progress": np.mean([r.mean_progress for r in group]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tagged_path(path: Path, tag: str) -> Path:
    if not tag:
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def plot_summary(rows: list[dict], scenarios: list[str], out_path: Path) -> None:
    controllers = ["fixed_policy", "liquid_online", "liquid_supervisor"]
    success = np.array(
        [[next(r for r in rows if r["controller"] == c and r["scenario"] == s)["success_rate"] for s in scenarios] for c in controllers]
    )
    collision = np.array(
        [[next(r for r in rows if r["controller"] == c and r["scenario"] == s)["collision_rate"] for s in scenarios] for c in controllers]
    )
    x = np.arange(len(scenarios))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for i, c in enumerate(controllers):
        axes[0].bar(x + (i - 1) * width, success[i], width, label=c)
        axes[1].bar(x + (i - 1) * width, collision[i], width, label=c)
    axes[0].set_title("Success rate")
    axes[1].set_title("Collision rate")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(rows: list[dict], out_path: Path, liquid_backend: str = "cfc") -> None:
    shifted = [r for r in rows if r["scenario"] != "train_like"]
    by_controller = {}
    for controller in ["fixed_policy", "liquid_online", "liquid_supervisor"]:
        group = [r for r in shifted if r["controller"] == controller]
        by_controller[controller] = {
            "success": float(np.mean([r["success_rate"] for r in group])),
            "collision": float(np.mean([r["collision_rate"] for r in group])),
            "lag": float(np.mean([r["mean_adaptation_lag"] for r in group])),
            "near": float(np.mean([r["mean_near_misses"] for r in group])),
        }
    lines = [
        "# H6 ilk deney sonuçları",
        "",
        f"Liquid politika backend'i: `{liquid_backend}`.",
        "Bu deney hâlâ küçük ölçekli bir navigasyon benzetimidir; liquid denetleyici artık resmi MIT `ncps.torch` CfC/LTC katmanlarıyla çalıştırılabilir.",
        "Sabit politika, resmi NCP tabanlı çevrimiçi sıvı politika ve liquid + sembolik güvenlik süpervizörü karşılaştırıldı.",
        "",
        "## Değişmiş dağılım senaryoları ortalaması",
        "",
        "| Denetleyici | Başarı | Çarpışma | Adaptasyon gecikmesi | Yakın geçiş |",
        "|---|---:|---:|---:|---:|",
    ]
    for controller, vals in by_controller.items():
        lines.append(
            f"| {controller} | {vals['success']:.3f} | {vals['collision']:.3f} | {vals['lag']:.1f} | {vals['near']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Yorum",
            "",
            "H6'nın beklediği ana desen şudur: sıvı durumlu çevrimiçi politika dağılım kaymasında sabit politikadan daha hızlı uyum sağlayabilir, ancak güvenlik kısıtı eklenmezse yakın geçiş ve çarpışma riski artabilir. Sembolik süpervizör, seçilen aksiyonu kısa ufuklu çarpışma/clearance kuralıyla maskeleyerek bu riski sınırlamayı hedefler.",
            "",
            "Bir sonraki adım, bu prototipi gerçek LNN/LTC hücresi eğitimi, daha zengin sensör modeli ve ablation çalışmalarıyla genişletmektir.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-set",
        choices=["default", "hard", "all"],
        default="default",
        help="Which scenario group to run.",
    )
    parser.add_argument("--tag", default="", help="Optional suffix for output files, e.g. hard_maps.")
    parser.add_argument(
        "--liquid-backend",
        choices=["legacy", "cfc", "ltc"],
        default="cfc",
        help="Liquid controller implementation. cfc/ltc use MIT ncps.torch; legacy uses the old handwritten prototype.",
    )
    parser.add_argument("--ncp-hidden", type=int, default=32, help="Number of neurons in the AutoNCP wiring.")
    parser.add_argument("--ncp-sparsity", type=float, default=0.50, help="AutoNCP sparsity level.")
    parser.add_argument("--ncp-baseline-scale", type=float, default=1.0, help="Scale of the fixed-policy baseline score. Use 0 for pure NCP scores.")
    parser.add_argument("--ncp-residual-scale", type=float, default=0.35, help="Scale of NCP residual scores over the fixed baseline.")
    parser.add_argument("--ncp-learning-rate", type=float, default=0.006, help="Online Adam learning rate for the NCP residual.")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base_weights = train_fixed_policy(rng)
    input_dim = base_weights.shape[0]

    if args.scenario_set == "default":
        scenarios = DEFAULT_SCENARIOS
    elif args.scenario_set == "hard":
        scenarios = HARD_SCENARIOS
    else:
        scenarios = [*DEFAULT_SCENARIOS, *HARD_SCENARIOS]
    results: list[EpisodeResult] = []
    for scenario in scenarios:
        for ep in range(args.episodes):
            seed = args.seed * 10_000 + ep * 17 + scenarios.index(scenario)
            controllers = [
                ("fixed_policy", FixedPolicy(base_weights.copy()), False),
                (
                    "liquid_online",
                    make_liquid_policy(
                        np.random.default_rng(seed + 1),
                        input_dim,
                        base_weights.copy(),
                        adaptive=True,
                        backend=args.liquid_backend,
                        hidden_dim=args.ncp_hidden,
                        sparsity=args.ncp_sparsity,
                        baseline_scale=args.ncp_baseline_scale,
                        residual_scale=args.ncp_residual_scale,
                        learning_rate=args.ncp_learning_rate,
                    ),
                    False,
                ),
                (
                    "liquid_supervisor",
                    make_liquid_policy(
                        np.random.default_rng(seed + 2),
                        input_dim,
                        base_weights.copy(),
                        adaptive=True,
                        backend=args.liquid_backend,
                        hidden_dim=args.ncp_hidden,
                        sparsity=args.ncp_sparsity,
                        baseline_scale=args.ncp_baseline_scale,
                        residual_scale=args.ncp_residual_scale,
                        learning_rate=args.ncp_learning_rate,
                    ),
                    True,
                ),
            ]
            for name, controller, supervisor in controllers:
                results.append(run_episode(name, controller, scenario, seed, supervisor))

    detail_rows = [r.__dict__ for r in results]
    summary_rows = aggregate(results)
    episode_path = tagged_path(RESULTS / "episode_results.csv", args.tag)
    summary_path = tagged_path(RESULTS / "summary_results.csv", args.tag)
    report_path = tagged_path(RESULTS / "summary.md", args.tag)
    figure_path = tagged_path(FIGURES / "h6_success_collision.png", args.tag)
    write_csv(episode_path, detail_rows)
    write_csv(summary_path, summary_rows)
    plot_summary(summary_rows, scenarios, figure_path)
    write_report(summary_rows, report_path, liquid_backend=args.liquid_backend)
    print(f"Wrote {episode_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
