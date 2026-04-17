from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import math
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

from run_lnn_experiment import (
    ACTION_DELTAS,
    DEFAULT_SCENARIOS,
    FIGURES,
    HARD_SCENARIOS,
    MAX_STEPS,
    RESULTS,
    EpisodeResult,
    FixedPolicy,
    wilson_ci,
    aggregate,
    collides,
    features,
    noisy_sensor,
    norm,
    obstacle_clearance,
    ray_cast,
    route_guided_scores,
    scenario_obstacles,
    scenario_sensor_params,
    train_fixed_policy,
    wrap_angle,
)


MODELS = RESULTS / "models"
OUTPUT_DIM = len(ACTION_DELTAS)


# ---------------------------------------------------------------------------
# Statistical utilities (no scipy dependency)
# ---------------------------------------------------------------------------

def mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test with normal (z-score) approximation.

    The normal approximation is most reliable when min(n1, n2) >= 8; for
    smaller samples the returned p-value is still defined and continuously
    scaled, but it should be interpreted cautiously. Callers in this module
    flag such comparisons with a ``small_n_warning`` column rather than
    rejecting them outright. For exact p-values use a permutation test or an
    implementation that tabulates the U distribution for small n.

    Returns (U_statistic, two_sided_p_value).
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])
    # Assign ranks with tie handling
    ranks: list[float] = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    # Tie correction
    tie_counts: list[int] = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        if j - i > 1:
            tie_counts.append(j - i)
        i = j
    n_total = n1 + n2
    tie_correction = sum(t ** 3 - t for t in tie_counts) / (12 * n_total * (n_total - 1)) if tie_counts else 0.0
    sigma = math.sqrt(n1 * n2 * ((n_total + 1) / 12 - tie_correction))
    if sigma < 1e-12:
        return u1, 1.0
    z = (u1 - mu) / sigma
    # Two-sided p-value via standard normal CDF approximation
    p = 2 * _normal_cdf(-abs(z))
    p = max(0.0, min(1.0, p))
    return u1, p


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via erf approximation (Abramowitz & Stegun 7.1.26).

    The A&S coefficients approximate erf(x), so we evaluate at x = |z|/sqrt(2)
    and convert: Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
    """
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p_coef = 0.3275911
    x = abs(z) / math.sqrt(2)
    sign = 1.0 if z >= 0 else -1.0
    t = 1.0 / (1.0 + p_coef * x)
    erf_approx = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * erf_approx)


def cohens_d(x: list[float], y: list[float]) -> float:
    """Cohen's d with pooled standard deviation.

    When the pooled SD is ~0 but means differ (e.g. binary data with perfect
    separation), returns ±inf to signal a maximal effect rather than a
    misleading 0.0.
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(x), np.mean(y)
    mean_diff = float(m1 - m2)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        if math.isclose(mean_diff, 0.0, abs_tol=1e-12):
            return 0.0
        return math.copysign(math.inf, mean_diff)
    return float(mean_diff / pooled)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    corrected = [0.0] * m
    prev = 1.0
    for rank_from_end, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = m - rank_from_end  # 1-based rank
        adjusted = min(prev, p * m / rank)
        corrected[orig_idx] = min(adjusted, 1.0)
        prev = adjusted
    return corrected


class NCPDiscreteModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cell_type: str,
        hidden_dim: int,
        sparsity: float,
        seed: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.cell_type = cell_type
        self.hidden_dim = int(hidden_dim)
        self.sparsity = float(sparsity)
        self.seed = int(seed)
        wiring = AutoNCP(
            self.hidden_dim,
            OUTPUT_DIM,
            sparsity_level=float(np.clip(self.sparsity, 0.10, 0.90)),
            seed=self.seed,
        )
        if cell_type == "cfc":
            self.rnn = CfC(self.input_dim, wiring, return_sequences=True, batch_first=True)
        elif cell_type == "ltc":
            self.rnn = LTC(self.input_dim, wiring, return_sequences=True, batch_first=True)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    def forward(self, x: torch.Tensor, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        # ncps 1.0.1 squeezes batched timespans too aggressively for wired
        # cells, so the offline batched trainer uses the library default dt=1.
        return self.rnn(x, hx)


class MLPDiscreteModel(nn.Module):
    """Feedforward MLP baseline -- no recurrence, same capacity budget."""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.cell_type = "mlp"
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)  # (batch, seq, OUTPUT_DIM)
        return logits, hx  # hx passthrough for API compatibility


class OfflineNCPPolicy:
    """Wraps a trained NCP or MLP model for offline evaluation."""

    def __init__(
        self,
        model: nn.Module,
        base_weights: np.ndarray,
        baseline_scale: float,
        ncp_scale: float,
    ):
        self.model = model.eval()
        self.base_weights = torch.as_tensor(base_weights, dtype=torch.float32)
        self.baseline_scale = float(baseline_scale)
        self.ncp_scale = float(ncp_scale)
        self.hx = None

    def reset(self) -> None:
        self.hx = None

    def act(self, x: np.ndarray, *_args) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32).reshape(1, 1, -1)
            ncp_logits, self.hx = self.model(xt, self.hx)
            baseline = xt[:, -1, :] @ self.base_weights
            scores = self.baseline_scale * baseline + self.ncp_scale * ncp_logits[:, -1, :]
            values = scores.cpu().numpy().reshape(-1)
        return int(np.argmax(values)), values

    def learn(self, *_args) -> None:
        pass


def reset_training_state(
    rng: np.random.Generator,
    scenario: str,
) -> tuple[list, np.ndarray, np.ndarray, float, float, float, float]:
    obstacles = scenario_obstacles(scenario)
    noise, dropout, bias = scenario_sensor_params(scenario)
    start = np.array([1.0, 1.0]) + rng.normal(0.0, 0.16, size=2)
    start = np.clip(start, 0.7, 1.4)
    goal = np.array([8.8, 8.7])
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0]) + rng.normal(0.0, 0.25)
    return obstacles, start, goal, heading, noise, dropout, bias


def expert_action(pos: np.ndarray, heading: float, goal: np.ndarray, obstacles: list) -> int:
    return int(np.argmax(route_guided_scores(pos, heading, goal, obstacles)))


def generate_imitation_dataset(
    rng: np.random.Generator,
    scenarios: list[str],
    n_sequences: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.zeros((n_sequences, seq_len, 18), dtype=np.float32)
    ys = np.zeros((n_sequences, seq_len), dtype=np.int64)
    for seq in range(n_sequences):
        scenario = scenarios[seq % len(scenarios)]
        obstacles, pos, goal, heading, noise, dropout, bias = reset_training_state(rng, scenario)
        for step in range(seq_len):
            true_ranges = ray_cast(pos, heading, obstacles)
            obs_ranges, uncertainty = noisy_sensor(true_ranges, rng, noise, dropout, bias)
            x = features(pos, heading, goal, obs_ranges, uncertainty)
            action = expert_action(pos, heading, goal, obstacles)
            xs[seq, step] = x.astype(np.float32)
            ys[seq, step] = action
            heading = wrap_angle(heading + float(ACTION_DELTAS[action]))
            pos = pos + 0.32 * np.array([math.cos(heading), math.sin(heading)])
            if collides(pos, obstacles) or norm(goal - pos) < 0.45:
                obstacles, pos, goal, heading, noise, dropout, bias = reset_training_state(rng, scenario)
    return xs, ys


def train_imitation(
    model: NCPDiscreteModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
    cell_type: str,
    log_rows: list[dict],
) -> NCPDiscreteModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)
    counts = np.bincount(y_train.reshape(-1), minlength=OUTPUT_DIM).astype(float)
    weights = counts.sum() / (OUTPUT_DIM * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.25, 6.0)
    weights = weights / weights.mean()
    class_weights = torch.as_tensor(weights, dtype=torch.float32)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    n = x_train.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        correct = 0
        total = 0
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            logits, _ = model(x_train_t[idx])
            loss = F.cross_entropy(logits.reshape(-1, OUTPUT_DIM), y_train_t[idx].reshape(-1), weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == y_train_t[idx]).sum())
            total += int(y_train_t[idx].numel())
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x_val_t)
            val_loss = F.cross_entropy(val_logits.reshape(-1, OUTPUT_DIM), y_val_t.reshape(-1), weight=class_weights)
            val_acc = float((val_logits.argmax(dim=-1) == y_val_t).float().mean())
        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
        log_rows.append(
            {
                "cell": cell_type,
                "phase": "imitation",
                "epoch_or_episode": epoch,
                "loss": float(np.mean(losses)),
                "train_accuracy": correct / max(1, total),
                "val_loss": float(val_loss),
                "val_accuracy": val_acc,
                "episode_return": "",
                "success": "",
                "collision": "",
            }
        )
    model.load_state_dict(best_state)
    return model


def rollout_policy_gradient_episode(
    model: NCPDiscreteModel,
    rng: np.random.Generator,
    scenario: str,
    max_steps: int,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict]:
    obstacles, pos, goal, heading, noise, dropout, bias = reset_training_state(rng, scenario)
    hx = None
    log_probs = []
    entropies = []
    rewards = []
    path_length = 0.0
    min_clearance = 99.0
    status = "timeout"
    prev_dist = norm(goal - pos)
    for step in range(max_steps):
        true_ranges = ray_cast(pos, heading, obstacles)
        obs_ranges, uncertainty = noisy_sensor(true_ranges, rng, noise, dropout, bias)
        x = features(pos, heading, goal, obs_ranges, uncertainty)
        xt = torch.as_tensor(x, dtype=torch.float32).reshape(1, 1, -1)
        logits, hx = model(xt, hx)
        dist = torch.distributions.Categorical(logits=logits[:, -1, :])
        action_t = dist.sample()
        action = int(action_t.item())
        log_probs.append(dist.log_prob(action_t).squeeze())
        entropies.append(dist.entropy().squeeze())

        new_heading = wrap_angle(heading + float(ACTION_DELTAS[action]))
        new_pos = pos + 0.32 * np.array([math.cos(new_heading), math.sin(new_heading)])
        path_length += norm(new_pos - pos)
        clearance = obstacle_clearance(new_pos, obstacles)
        min_clearance = min(min_clearance, clearance)
        new_dist = norm(goal - new_pos)
        progress = prev_dist - new_dist
        reward = 2.5 * progress + 0.08 * min(clearance, 1.0) - 0.015 * abs(float(ACTION_DELTAS[action]))
        if clearance < 0.35:
            reward -= 0.35 * (0.35 - clearance)
        if progress < -0.02:
            reward -= 0.08
        pos = new_pos
        heading = new_heading
        prev_dist = new_dist
        if clearance <= 0:
            reward -= 8.0
            rewards.append(reward)
            status = "collision"
            break
        if new_dist < 0.45:
            reward += 12.0
            rewards.append(reward)
            status = "success"
            break
        rewards.append(reward)

    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + 0.97 * running
        returns.append(running)
    returns.reverse()
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    if len(returns_t) > 1 and float(returns_t.std()) > 1e-6:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)
    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)
    loss = -(log_probs_t * returns_t).sum() - entropy_coef * entropies_t.sum()
    info = {
        "return": float(sum(rewards)),
        "success": int(status == "success"),
        "collision": int(status == "collision"),
        "steps": len(rewards),
        "path_length": path_length,
        "min_clearance": min_clearance,
    }
    return loss, info


def fine_tune_rl(
    model: NCPDiscreteModel,
    rng: np.random.Generator,
    scenarios: list[str],
    episodes: int,
    max_steps: int,
    lr: float,
    entropy_coef: float,
    cell_type: str,
    log_rows: list[dict],
) -> NCPDiscreteModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for episode in range(1, episodes + 1):
        scenario = scenarios[(episode - 1) % len(scenarios)]
        loss, info = rollout_policy_gradient_episode(model, rng, scenario, max_steps, entropy_coef)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log_rows.append(
            {
                "cell": cell_type,
                "phase": "rl_finetune",
                "epoch_or_episode": episode,
                "loss": float(loss.detach()),
                "train_accuracy": "",
                "val_loss": "",
                "val_accuracy": "",
                "episode_return": info["return"],
                "success": info["success"],
                "collision": info["collision"],
            }
        )
    return model


def save_checkpoint(path: Path, model: NCPDiscreteModel, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def run_eval_episode(
    controller_name: str,
    controller,
    scenario: str,
    seed: int,
    max_steps: int,
    baseline_weights: np.ndarray | None = None,
) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    obstacles = scenario_obstacles(scenario)
    noise, dropout, bias = scenario_sensor_params(scenario)
    start = np.array([1.0, 1.0]) + rng.normal(0.0, 0.06, size=2)
    goal = np.array([8.8, 8.7])
    pos = start.copy()
    heading = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
    controller.reset()

    path_length = 0.0
    near_misses = 0
    min_clearance = 99.0
    progress_history: list[float] = []
    action_disagreements = 0
    beneficial_disagreements = 0
    prev_dist = norm(goal - pos)
    for step in range(max_steps):
        true_ranges = ray_cast(pos, heading, obstacles)
        obs_ranges, uncertainty = noisy_sensor(true_ranges, rng, noise, dropout, bias)
        x = features(pos, heading, goal, obs_ranges, uncertainty)
        action, _scores = controller.act(x, pos, heading, goal, obstacles)
        new_heading = wrap_angle(heading + float(ACTION_DELTAS[action]))
        new_pos = pos + 0.32 * np.array([math.cos(new_heading), math.sin(new_heading)])

        # Action disagreement tracking
        if baseline_weights is not None:
            baseline_action = int(np.argmax(x @ baseline_weights))
            if action != baseline_action:
                action_disagreements += 1
                bl_heading = wrap_angle(heading + float(ACTION_DELTAS[baseline_action]))
                bl_pos = pos + 0.32 * np.array([math.cos(bl_heading), math.sin(bl_heading)])
                bl_clearance = obstacle_clearance(bl_pos, obstacles)
                ctrl_clearance = obstacle_clearance(new_pos, obstacles)
                bl_dist = norm(goal - bl_pos)
                if ctrl_clearance > bl_clearance or norm(goal - new_pos) < bl_dist:
                    beneficial_disagreements += 1

        path_length += norm(new_pos - pos)
        clearance = obstacle_clearance(new_pos, obstacles)
        min_clearance = min(min_clearance, clearance)
        new_dist = norm(goal - new_pos)
        progress = prev_dist - new_dist
        progress_history.append(progress)
        if clearance < 0.35:
            near_misses += 1
        if clearance <= 0:
            return EpisodeResult(
                controller_name, scenario, seed, 0, 1, step + 1,
                path_length, min_clearance, near_misses, 0, max_steps,
                float(np.mean(progress_history)),
                action_disagreements, beneficial_disagreements,
            )
        pos = new_pos
        heading = new_heading
        prev_dist = new_dist
        if new_dist < 0.45:
            return EpisodeResult(
                controller_name, scenario, seed, 1, 0, step + 1,
                path_length, min_clearance, near_misses, 0, step + 1,
                float(np.mean(progress_history)),
                action_disagreements, beneficial_disagreements,
            )
    return EpisodeResult(
        controller_name, scenario, seed, 0, 0, max_steps,
        path_length, min_clearance, near_misses, 0, max_steps,
        float(np.mean(progress_history)),
        action_disagreements, beneficial_disagreements,
    )


def evaluate_models(
    trained: dict[tuple[str, str], nn.Module],
    base_weights: np.ndarray,
    scenarios: list[str],
    episodes: int,
    max_steps: int,
    residual_scale: float,
    seed: int,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for scenario in scenarios:
        for ep in range(episodes):
            episode_seed = seed * 100_000 + ep * 113 + scenarios.index(scenario)
            results.append(
                run_eval_episode(
                    "fixed_policy",
                    FixedPolicy(base_weights.copy()),
                    scenario,
                    episode_seed,
                    max_steps,
                    baseline_weights=None,
                )
            )
            for (cell, stage), model in trained.items():
                if stage == "random":
                    # Random residual control -- untrained model
                    controller = OfflineNCPPolicy(model, base_weights, 1.0, residual_scale)
                    name = f"{cell}_random_residual"
                    results.append(run_eval_episode(
                        name, controller, scenario, episode_seed, max_steps,
                        baseline_weights=base_weights))
                    continue
                variants = [
                    ("pure", 0.0, 1.0),
                    ("residual", 1.0, residual_scale),
                ]
                for variant, baseline_scale, ncp_scale in variants:
                    controller = OfflineNCPPolicy(model, base_weights, baseline_scale, ncp_scale)
                    name = f"{cell}_{stage}_{variant}"
                    results.append(run_eval_episode(
                        name, controller, scenario, episode_seed, max_steps,
                        baseline_weights=base_weights))
    return results


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def controller_metadata(name: str) -> dict:
    if name == "fixed_policy":
        return {"cell": "fixed", "stage": "fixed", "variant": "baseline"}
    if name.endswith("_random_residual"):
        cell = name.replace("_random_residual", "")
        return {"cell": cell, "stage": "random", "variant": "residual"}
    parts = name.split("_")
    cell = parts[0]
    if cell == "mlp":
        stage = f"{parts[1]}_{parts[2]}" if len(parts) > 2 and parts[1] == "rl" else parts[1]
        return {"cell": "mlp", "stage": stage, "variant": parts[-1]}
    return {"cell": parts[0], "stage": f"{parts[1]}_{parts[2]}" if parts[1] == "rl" else parts[1], "variant": parts[-1]}


def scenario_group(scenario: str) -> str:
    if scenario in HARD_SCENARIOS:
        return "hard"
    if scenario in DEFAULT_SCENARIOS:
        return "default"
    return "other"


def build_group_tables(
    records: list[tuple[int, EpisodeResult]],
) -> tuple[list[dict], list[dict]]:
    """Build detail + group summary tables.

    ``records`` is a list of ``(seed_idx, EpisodeResult)`` pairs so the detail
    rows and group summaries can surface *both* pooled (episode-level) and
    seed-level uncertainty. Seed-level CIs are what the "multi-seed" claim
    actually supports: they estimate variability across independent training
    runs rather than treating every episode as independent.
    """
    detail = []
    for seed_idx, result in records:
        row = asdict(result)
        row.update(controller_metadata(result.controller))
        row["scenario_group"] = scenario_group(result.scenario)
        row["seed_idx"] = int(seed_idx)
        detail.append(row)

    def _stats(vals: list[float]) -> tuple[float, float, float, float, float]:
        n = len(vals)
        m = float(np.mean(vals))
        if n <= 1:
            return m, 0.0, 0.0, m, m
        s = float(np.std(vals, ddof=1))
        se = s / math.sqrt(n)
        return m, s, se, m - 1.96 * se, m + 1.96 * se

    def _seed_level_ci(
        rows: list[dict], key: str
    ) -> tuple[int, float | None, float | None, float | None, float | None]:
        """Aggregate by seed_idx first, then compute across-seed statistics.

        Returns (n_seeds, seed_mean, seed_std, seed_ci_lower, seed_ci_upper).
        With n_seeds < 2 the std/CI are undefined and returned as ``None``.
        """
        by_seed: dict[int, list[float]] = {}
        for r in rows:
            by_seed.setdefault(r["seed_idx"], []).append(float(r.get(key, 0.0)))
        per_seed_means = [float(np.mean(v)) for v in by_seed.values() if len(v) > 0]
        n_seeds = len(per_seed_means)
        if n_seeds == 0:
            return 0, None, None, None, None
        mean = float(np.mean(per_seed_means))
        if n_seeds < 2:
            return n_seeds, mean, None, None, None
        std = float(np.std(per_seed_means, ddof=1))
        se = std / math.sqrt(n_seeds)
        # Normal-approximation CI (n_seeds typically 3-10; a t-based CI would be
        # stricter but would add a scipy/t-distribution dependency).
        return n_seeds, mean, std, mean - 1.96 * se, mean + 1.96 * se

    grouped = []
    keys = sorted({(r["controller"], r["scenario_group"]) for r in detail})
    for controller, group in keys:
        rows = [r for r in detail if r["controller"] == controller and r["scenario_group"] == group]
        meta = controller_metadata(controller)
        n = len(rows)
        successes = sum(r["success"] for r in rows)
        sr = successes / max(n, 1)
        sr_lo, sr_hi = wilson_ci(successes, n)
        cr = float(np.mean([r["collision"] for r in rows]))
        cr_successes = sum(r["collision"] for r in rows)
        cr_lo, cr_hi = wilson_ci(cr_successes, n)
        steps_m, steps_s, steps_se, _, _ = _stats([r["steps"] for r in rows])
        mc_m, mc_s, mc_se, _, _ = _stats([r["min_clearance"] for r in rows])
        nm_m, _, _, _, _ = _stats([float(r["near_misses"]) for r in rows])
        pr_m, pr_s, pr_se, _, _ = _stats([r["mean_progress"] for r in rows])
        ad_m, _, _, _, _ = _stats([float(r.get("action_disagreements", 0)) for r in rows])
        bd_m, _, _, _, _ = _stats([float(r.get("beneficial_disagreements", 0)) for r in rows])

        n_seeds_used, sr_seed_m, sr_seed_s, sr_seed_lo, sr_seed_hi = _seed_level_ci(rows, "success")
        _, cr_seed_m, cr_seed_s, cr_seed_lo, cr_seed_hi = _seed_level_ci(rows, "collision")

        grouped.append(
            {
                "controller": controller,
                **meta,
                "scenario_group": group,
                "n": n,
                "n_seeds": n_seeds_used,
                "success_rate": sr,
                "success_ci_lower": sr_lo,
                "success_ci_upper": sr_hi,
                "success_seed_mean": sr_seed_m,
                "success_seed_std": sr_seed_s,
                "success_seed_ci_lower": sr_seed_lo,
                "success_seed_ci_upper": sr_seed_hi,
                "collision_rate": cr,
                "collision_ci_lower": cr_lo,
                "collision_ci_upper": cr_hi,
                "collision_seed_mean": cr_seed_m,
                "collision_seed_std": cr_seed_s,
                "collision_seed_ci_lower": cr_seed_lo,
                "collision_seed_ci_upper": cr_seed_hi,
                "mean_steps": steps_m,
                "std_steps": steps_s,
                "mean_min_clearance": mc_m,
                "std_min_clearance": mc_s,
                "mean_near_misses": nm_m,
                "mean_progress": pr_m,
                "std_progress": pr_s,
                "mean_action_disagreements": ad_m,
                "mean_beneficial_disagreements": bd_m,
            }
        )
    return detail, grouped


def build_pairwise_tables(grouped_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key = {(r["cell"], r["stage"], r["variant"], r["scenario_group"]): r for r in grouped_rows}
    residual_vs_pure = []
    for cell in ["cfc", "ltc", "mlp"]:
        for stage in ["imitation", "rl_finetune"]:
            for group in ["default", "hard"]:
                pure = by_key.get((cell, stage, "pure", group))
                residual = by_key.get((cell, stage, "residual", group))
                if pure is None or residual is None:
                    continue
                residual_vs_pure.append(
                    {
                        "cell": cell,
                        "stage": stage,
                        "scenario_group": group,
                        "pure_success": pure["success_rate"],
                        "residual_success": residual["success_rate"],
                        "delta_success": residual["success_rate"] - pure["success_rate"],
                        "pure_collision": pure["collision_rate"],
                        "residual_collision": residual["collision_rate"],
                        "delta_collision": residual["collision_rate"] - pure["collision_rate"],
                        "pure_min_clearance": pure["mean_min_clearance"],
                        "residual_min_clearance": residual["mean_min_clearance"],
                        "delta_min_clearance": residual["mean_min_clearance"] - pure["mean_min_clearance"],
                    }
                )

    cfc_vs_ltc = []
    for stage in ["imitation", "rl_finetune"]:
        for variant in ["pure", "residual"]:
            for group in ["default", "hard"]:
                cfc = by_key.get(("cfc", stage, variant, group))
                ltc = by_key.get(("ltc", stage, variant, group))
                if cfc is None or ltc is None:
                    continue
                cfc_vs_ltc.append(
                    {
                        "stage": stage,
                        "variant": variant,
                        "scenario_group": group,
                        "cfc_success": cfc["success_rate"],
                        "ltc_success": ltc["success_rate"],
                        "delta_success_cfc_minus_ltc": cfc["success_rate"] - ltc["success_rate"],
                        "cfc_collision": cfc["collision_rate"],
                        "ltc_collision": ltc["collision_rate"],
                        "delta_collision_cfc_minus_ltc": cfc["collision_rate"] - ltc["collision_rate"],
                        "cfc_min_clearance": cfc["mean_min_clearance"],
                        "ltc_min_clearance": ltc["mean_min_clearance"],
                        "delta_min_clearance_cfc_minus_ltc": cfc["mean_min_clearance"] - ltc["mean_min_clearance"],
                    }
                )
    return residual_vs_pure, cfc_vs_ltc


def markdown_table(rows: list[dict], columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if column in formats and isinstance(value, (float, int)):
                values.append(format(value, formats[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_summary_markdown(
    path: Path,
    grouped_rows: list[dict],
    residual_vs_pure: list[dict],
    cfc_vs_ltc: list[dict],
    statistical_comparisons: list[dict],
    training_rows: list[dict],
    scenarios: list[str],
    args: argparse.Namespace,
) -> None:
    hard_rows = [r for r in grouped_rows if r["scenario_group"] == "hard"]
    hard_rows = sorted(hard_rows, key=lambda r: (r["cell"], r["stage"], r["variant"]))
    default_rows = [r for r in grouped_rows if r["scenario_group"] == "default"]
    default_rows = sorted(default_rows, key=lambda r: (r["cell"], r["stage"], r["variant"]))
    train_tail = []
    for cell in ["cfc", "ltc", "mlp"]:
        for phase in ["imitation", "rl_finetune"]:
            candidates = [r for r in training_rows if r["cell"] == cell and r["phase"] == phase]
            if candidates:
                train_tail.append(candidates[-1])

    formats = {
        "success_rate": ".3f",
        "success_ci_lower": ".3f",
        "success_ci_upper": ".3f",
        "collision_rate": ".3f",
        "collision_ci_lower": ".3f",
        "collision_ci_upper": ".3f",
        "mean_steps": ".1f",
        "std_steps": ".1f",
        "mean_min_clearance": ".3f",
        "std_min_clearance": ".3f",
        "mean_near_misses": ".1f",
        "mean_progress": ".4f",
        "std_progress": ".4f",
        "mean_action_disagreements": ".1f",
        "mean_beneficial_disagreements": ".1f",
        "pure_success": ".3f",
        "residual_success": ".3f",
        "delta_success": ".3f",
        "pure_collision": ".3f",
        "residual_collision": ".3f",
        "delta_collision": ".3f",
        "pure_min_clearance": ".3f",
        "residual_min_clearance": ".3f",
        "delta_min_clearance": ".3f",
        "cfc_success": ".3f",
        "ltc_success": ".3f",
        "delta_success_cfc_minus_ltc": ".3f",
        "cfc_collision": ".3f",
        "ltc_collision": ".3f",
        "delta_collision_cfc_minus_ltc": ".3f",
        "cfc_min_clearance": ".3f",
        "ltc_min_clearance": ".3f",
        "delta_min_clearance_cfc_minus_ltc": ".3f",
        "loss": ".4f",
        "val_accuracy": ".3f",
        "episode_return": ".3f",
        "mean_a": ".3f",
        "mean_b": ".3f",
        "delta": ".3f",
        "mann_whitney_U": ".1f",
        "p_value_raw": ".4f",
        "p_value_bh_corrected": ".4f",
        "cohens_d": ".3f",
    }
    n_seeds = getattr(args, "n_seeds", 1)
    lines = [
        "# NCP / MLP ablation sonuçları (istatistiksel analiz dahil)",
        "",
        "Bu dosya resmi `ncps.torch` CfC/LTC katmanları ve MLP baseline ile üretilen offline imitation,",
        "kısa policy-gradient fine-tune, pure/residual ablation ve istatistiksel karşılaştırma sonuçlarını özetler.",
        "",
        "## Deney konfigürasyonu",
        "",
        f"- Bağımsız eğitim seed sayısı: {n_seeds}",
        f"- Eğitim senaryoları: {', '.join(DEFAULT_SCENARIOS)}",
        f"- Test senaryoları: {', '.join(scenarios)}",
        f"- Imitation sequence sayısı: {args.train_sequences}, doğrulama: {args.val_sequences}, sequence length: {args.seq_len}",
        f"- Imitation epoch: {args.imitation_epochs}, RL fine-tune episode: {args.rl_episodes}",
        f"- NCP hidden: {args.hidden_dim}, sparsity: {args.sparsity}, residual scale: {args.residual_scale}",
        f"- Değerlendirme episode: {args.eval_episodes} (senaryo başına)",
        "",
        "## Eğitim özeti (son seed)",
        "",
    ]
    lines.extend(markdown_table(train_tail, ["cell", "phase", "epoch_or_episode", "loss", "val_accuracy", "episode_return", "success", "collision"], formats))
    lines.extend(["", "## Residual vs pure fark tablosu", ""])
    lines.extend(
        markdown_table(
            residual_vs_pure,
            ["cell", "stage", "scenario_group", "pure_success", "residual_success",
             "delta_success", "pure_collision", "residual_collision", "delta_collision", "delta_min_clearance"],
            formats,
        )
    )
    lines.extend(["", "## CfC vs LTC fark tablosu", ""])
    lines.extend(
        markdown_table(
            cfc_vs_ltc,
            ["stage", "variant", "scenario_group", "cfc_success", "ltc_success",
             "delta_success_cfc_minus_ltc", "cfc_collision", "ltc_collision",
             "delta_collision_cfc_minus_ltc", "delta_min_clearance_cfc_minus_ltc"],
            formats,
        )
    )
    # Hard map table with CI
    lines.extend(["", "## Hard map ortalaması (95% CI dahil)", ""])
    lines.extend(
        markdown_table(
            hard_rows,
            ["controller", "cell", "stage", "variant", "n",
             "success_rate", "success_ci_lower", "success_ci_upper",
             "collision_rate", "mean_steps", "mean_min_clearance",
             "mean_near_misses", "mean_action_disagreements", "mean_beneficial_disagreements"],
            formats,
        )
    )
    # Default map table with CI
    lines.extend(["", "## Default map ortalaması (95% CI dahil)", ""])
    lines.extend(
        markdown_table(
            default_rows,
            ["controller", "cell", "stage", "variant", "n",
             "success_rate", "success_ci_lower", "success_ci_upper",
             "collision_rate", "mean_steps", "mean_min_clearance",
             "mean_near_misses", "mean_action_disagreements", "mean_beneficial_disagreements"],
            formats,
        )
    )
    # Statistical comparisons
    if statistical_comparisons:
        lines.extend(["", "## İstatistiksel karşılaştırmalar (Mann-Whitney U, BH düzeltmeli)", ""])
        lines.extend(
            markdown_table(
                statistical_comparisons,
                ["scenario_group", "comparison", "controller_a", "controller_b",
                 "n_a", "n_b", "small_n_warning", "mean_a", "mean_b", "delta",
                 "mann_whitney_U", "p_value_raw", "p_value_bh_corrected",
                 "cohens_d", "significant_005"],
                formats,
            )
        )
    lines.extend(
        [
            "",
            "## Bilimsel okuma",
            "",
            "- `pure` varyantında karar tamamen NCP/MLP logits ile verilir; bu saf kapasite ve kararlılık testidir.",
            "- `residual` varyantında sabit uzman politikanın üstüne NCP/MLP düzeltmesi eklenir; bu daha güvenli dağıtım senaryosunu temsil eder.",
            "- `random_residual` eğitilmemiş (rastgele ağırlıklı) NCP ile residual yapıyı test eder; öğrenilmiş bilginin etkisini izole eder.",
            "- `mlp` baseline, recurrent olmayan feedforward ağdır; NCP mimarisinin (sürekli zaman dinamikleri) etkisini ayırt etmeyi sağlar.",
            "- CfC/LTC kıyası aynı veri, seed ailesi ve harita setinde yapılır; farklar mimari ve kısa fine-tune dinamiğinden gelir.",
            f"- {n_seeds} bağımsız seed ile eğitim yapılmış, sonuçlar tüm seed'ler üzerinden toplanmıştır.",
            "- Wilson score interval (95% CI) küçük örneklem için uygun binomial güven aralığı sağlar.",
            "- Benjamini-Hochberg FDR düzeltmesi çoklu karşılaştırma hatasını kontrol eder.",
            "- `small_n_warning=yes`: Grup başına n<8 olduğunda Mann-Whitney U normal yaklaşımı güvenilirliğini yitirir; bu satırlardaki p-değerleri dikkatli yorumlanmalıdır.",
            "- Aksiyon uyuşmazlığı (action disagreement): NCP/MLP'nin sabit politikadan farklı karar verdiği adım sayısı.",
            "- Yararlı uyuşmazlık (beneficial disagreement): Farklı kararın daha iyi clearance veya ilerleme sağladığı adım sayısı.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_ablation(grouped_rows: list[dict], out_path: Path) -> None:
    hard = [r for r in grouped_rows if r["scenario_group"] == "hard"]
    order = [
        "fixed_policy",
        "mlp_imitation_pure",
        "mlp_imitation_residual",
        "mlp_rl_finetune_pure",
        "mlp_rl_finetune_residual",
        "cfc_imitation_pure",
        "cfc_imitation_residual",
        "cfc_rl_finetune_pure",
        "cfc_rl_finetune_residual",
        "cfc_random_residual",
        "ltc_imitation_pure",
        "ltc_imitation_residual",
        "ltc_rl_finetune_pure",
        "ltc_rl_finetune_residual",
        "ltc_random_residual",
    ]
    rows = [next((r for r in hard if r["controller"] == name), None) for name in order]
    rows = [r for r in rows if r is not None]
    labels = [r["controller"].replace("_", "\n") for r in rows]
    success = [r["success_rate"] for r in rows]
    collision = [r["collision_rate"] for r in rows]
    clearance = [r["mean_min_clearance"] for r in rows]

    # Error bars from Wilson CI
    s_err_lo = [r["success_rate"] - r.get("success_ci_lower", r["success_rate"]) for r in rows]
    s_err_hi = [r.get("success_ci_upper", r["success_rate"]) - r["success_rate"] for r in rows]
    c_err_lo = [r["collision_rate"] - r.get("collision_ci_lower", r["collision_rate"]) for r in rows]
    c_err_hi = [r.get("collision_ci_upper", r["collision_rate"]) - r["collision_rate"] for r in rows]
    cl_err = [r.get("std_min_clearance", 0.0) for r in rows]

    # Color coding
    colors = []
    for r in rows:
        cell = r.get("cell", "")
        variant = r.get("variant", "")
        if cell == "fixed":
            colors.append("#e8a838")
        elif cell == "mlp":
            colors.append("#4a90d9")
        elif variant == "residual" and r.get("stage") == "random":
            colors.append("#999999")
        elif cell == "cfc":
            colors.append("#16837a")
        elif cell == "ltc":
            colors.append("#2ca02c")
        else:
            colors.append("#555555")

    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].bar(x, success, color=colors, yerr=[s_err_lo, s_err_hi], capsize=3, ecolor="#333")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate (95% CI)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, collision, color=colors, yerr=[c_err_lo, c_err_hi], capsize=3, ecolor="#333")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("collision rate (95% CI)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, clearance, color=colors, yerr=cl_err, capsize=3, ecolor="#333")
    axes[2].set_ylabel("min clearance (±1 std)")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=0, fontsize=7)
    fig.suptitle("Hard-map ablation: NCP vs MLP vs random")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_statistical_comparisons(detail_rows: list[dict]) -> list[dict]:
    """Pairwise Mann-Whitney U tests with effect sizes and BH correction."""
    comparisons: list[dict] = []
    for group in ["default", "hard"]:
        group_rows = [r for r in detail_rows if r["scenario_group"] == group]
        controllers = sorted({r["controller"] for r in group_rows})
        pairs: list[tuple[str, str, str]] = []
        for cell in ["cfc", "ltc", "mlp"]:
            for stage in ["imitation", "rl_finetune"]:
                pure = f"{cell}_{stage}_pure"
                residual = f"{cell}_{stage}_residual"
                if pure in controllers and residual in controllers:
                    pairs.append((residual, pure, "residual_vs_pure"))
                if residual in controllers and "fixed_policy" in controllers:
                    label = "mlp_vs_fixed" if cell == "mlp" else "ncp_vs_fixed"
                    pairs.append((residual, "fixed_policy", label))
            random = f"{cell}_random_residual"
            best_res = f"{cell}_rl_finetune_residual"
            if random in controllers and best_res in controllers:
                pairs.append((best_res, random, "trained_vs_random"))
        for cell in ["cfc", "ltc"]:
            for stage in ["imitation", "rl_finetune"]:
                for variant in ["pure", "residual"]:
                    ncp_name = f"{cell}_{stage}_{variant}"
                    mlp_name = f"mlp_{stage}_{variant}"
                    if ncp_name in controllers and mlp_name in controllers:
                        pairs.append((ncp_name, mlp_name, "ncp_vs_mlp"))
        for stage in ["imitation", "rl_finetune"]:
            for variant in ["pure", "residual"]:
                cfc_name = f"cfc_{stage}_{variant}"
                ltc_name = f"ltc_{stage}_{variant}"
                if cfc_name in controllers and ltc_name in controllers:
                    pairs.append((cfc_name, ltc_name, "cfc_vs_ltc"))

        p_values: list[float] = []
        comparison_data: list[dict] = []
        for ctrl_a, ctrl_b, comparison_type in pairs:
            a_vals = [r["success"] for r in group_rows if r["controller"] == ctrl_a]
            b_vals = [r["success"] for r in group_rows if r["controller"] == ctrl_b]
            if len(a_vals) < 3 or len(b_vals) < 3:
                continue
            small_n = len(a_vals) < 8 or len(b_vals) < 8
            u, p = mann_whitney_u(a_vals, b_vals)
            d = cohens_d(a_vals, b_vals)
            p_values.append(p)
            comparison_data.append({
                "scenario_group": group,
                "comparison": comparison_type,
                "controller_a": ctrl_a,
                "controller_b": ctrl_b,
                "n_a": len(a_vals),
                "n_b": len(b_vals),
                "small_n_warning": "yes" if small_n else "no",
                "mean_a": float(np.mean(a_vals)),
                "mean_b": float(np.mean(b_vals)),
                "delta": float(np.mean(a_vals)) - float(np.mean(b_vals)),
                "mann_whitney_U": u,
                "p_value_raw": p,
                "cohens_d": d,
            })
        if p_values:
            corrected = benjamini_hochberg(p_values)
            for row, p_corr in zip(comparison_data, corrected):
                row["p_value_bh_corrected"] = p_corr
                row["significant_005"] = "yes" if p_corr < 0.05 else "no"
        comparisons.extend(comparison_data)
    return comparisons


def _train_and_evaluate_seed(
    seed_idx: int,
    args: argparse.Namespace,
    base_weights: np.ndarray,
    input_dim: int,
    train_scenarios: list,
    eval_scenarios: list,
) -> tuple[list[dict], list[tuple[int, EpisodeResult]]]:
    """Train + evaluate one seed. Safe for ProcessPoolExecutor workers.

    Returns (training_rows tagged with seed_idx, eval records as (seed_idx, EpisodeResult))
    so callers can merge independent workers' output without mixing seeds.
    """
    # Each worker process gets its own torch thread pool. With N workers on a
    # machine with M cores, the default M threads per worker would mean N*M
    # OS threads fighting for the same cores. Pin to 1 so the OS scheduler
    # distributes workers cleanly across cores.
    torch.set_num_threads(1)

    current_seed = args.seed + seed_idx * 1000
    print(f"\n=== Seed {seed_idx + 1}/{args.n_seeds} (seed={current_seed}) ===")
    rng = np.random.default_rng(current_seed)
    torch.manual_seed(current_seed)

    x_train, y_train = generate_imitation_dataset(rng, train_scenarios, args.train_sequences, args.seq_len)
    x_val, y_val = generate_imitation_dataset(rng, train_scenarios, args.val_sequences, args.seq_len)

    training_rows: list[dict] = []
    trained: dict[tuple[str, str], nn.Module] = {}

    for cell in ["cfc", "ltc", "mlp"]:
        cell_seed = current_seed + {"cfc": 1, "ltc": 2, "mlp": 3}[cell]
        torch.manual_seed(cell_seed)
        if cell == "mlp":
            model: nn.Module = MLPDiscreteModel(input_dim, args.hidden_dim, cell_seed)
        else:
            model = NCPDiscreteModel(input_dim, cell, args.hidden_dim, args.sparsity, cell_seed)
        model = train_imitation(
            model, x_train, y_train, x_val, y_val,
            args.imitation_epochs, args.batch_size, args.imitation_lr,
            rng, cell, training_rows,
        )
        trained[(cell, "imitation")] = copy.deepcopy(model)
        trained[(cell, "imitation")].train(False)
        save_checkpoint(
            MODELS / f"ncp_{cell}_imitation_seed{seed_idx}.pt",
            trained[(cell, "imitation")],
            {"cell": cell, "stage": "imitation", "seed_idx": seed_idx,
             "input_dim": input_dim, "hidden_dim": args.hidden_dim, "sparsity": args.sparsity},
        )

        rl_model = copy.deepcopy(model)
        rl_model = fine_tune_rl(
            rl_model, rng, train_scenarios,
            args.rl_episodes, args.rl_max_steps, args.rl_lr,
            args.entropy_coef, cell, training_rows,
        )
        trained[(cell, "rl_finetune")] = rl_model
        trained[(cell, "rl_finetune")].train(False)
        save_checkpoint(
            MODELS / f"ncp_{cell}_rl_finetune_seed{seed_idx}.pt",
            trained[(cell, "rl_finetune")],
            {"cell": cell, "stage": "rl_finetune", "seed_idx": seed_idx,
             "input_dim": input_dim, "hidden_dim": args.hidden_dim, "sparsity": args.sparsity},
        )

    for cell in ["cfc", "ltc"]:
        random_seed = current_seed + 999
        torch.manual_seed(random_seed)
        random_model = NCPDiscreteModel(input_dim, cell, args.hidden_dim, args.sparsity, random_seed)
        random_model.train(False)
        trained[(cell, "random")] = random_model

    for row in training_rows:
        row["seed_idx"] = seed_idx

    seed_results = evaluate_models(
        trained, base_weights, eval_scenarios,
        args.eval_episodes, args.eval_max_steps,
        args.residual_scale, current_seed,
    )
    eval_records = [(seed_idx, r) for r in seed_results]
    print(f"  Seed {seed_idx}: {len(seed_results)} episodes completed.")
    return training_rows, eval_records


def run_pipeline(args: argparse.Namespace) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train_scenarios = list(DEFAULT_SCENARIOS)
    eval_scenarios = list(DEFAULT_SCENARIOS) + list(HARD_SCENARIOS)

    all_training_rows: list[dict] = []
    all_eval_records: list[tuple[int, EpisodeResult]] = []

    # Train the fixed baseline once (seed-independent) so all NCP/MLP seeds
    # compare against the *same* controller and aggregated metrics do not mix
    # different baselines. Seed-level variability then reflects only NCP/MLP
    # training randomness, not baseline drift.
    base_weights = train_fixed_policy(np.random.default_rng(args.seed))
    input_dim = base_weights.shape[0]

    parallel_seeds = max(1, int(getattr(args, "parallel_seeds", 1)))
    effective_parallel = min(parallel_seeds, args.n_seeds)

    if effective_parallel <= 1:
        for seed_idx in range(args.n_seeds):
            training_rows, eval_records = _train_and_evaluate_seed(
                seed_idx, args, base_weights, input_dim, train_scenarios, eval_scenarios,
            )
            all_training_rows.extend(training_rows)
            all_eval_records.extend(eval_records)
    else:
        print(f"[parallel] Running {args.n_seeds} seeds across {effective_parallel} workers "
              "(each worker pinned to torch.set_num_threads(1)).")
        per_seed_rows: dict[int, list[dict]] = {}
        per_seed_records: dict[int, list[tuple[int, EpisodeResult]]] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_parallel) as pool:
            futures = {
                pool.submit(
                    _train_and_evaluate_seed,
                    seed_idx, args, base_weights, input_dim, train_scenarios, eval_scenarios,
                ): seed_idx
                for seed_idx in range(args.n_seeds)
            }
            for future in concurrent.futures.as_completed(futures):
                seed_idx = futures[future]
                training_rows, eval_records = future.result()
                per_seed_rows[seed_idx] = training_rows
                per_seed_records[seed_idx] = eval_records
        # Merge in seed_idx order so row ordering and downstream aggregation
        # stay deterministic regardless of worker completion order.
        for seed_idx in range(args.n_seeds):
            all_training_rows.extend(per_seed_rows[seed_idx])
            all_eval_records.extend(per_seed_records[seed_idx])

    # Aggregate all results across seeds
    all_eval_results = [r for _, r in all_eval_records]
    detail_rows, grouped_rows = build_group_tables(all_eval_records)
    residual_vs_pure, cfc_vs_ltc = build_pairwise_tables(grouped_rows)
    statistical_comparisons = build_statistical_comparisons(detail_rows)
    scenario_summary = aggregate(all_eval_results)
    scenario_summary = [{**row, **controller_metadata(row["controller"])} for row in scenario_summary]

    write_rows(RESULTS / "ncp_training_log.csv", all_training_rows)
    write_rows(RESULTS / "ncp_ablation_episode_results.csv", detail_rows)
    write_rows(RESULTS / "ncp_ablation_group_summary.csv", grouped_rows)
    write_rows(RESULTS / "ncp_residual_vs_pure_summary.csv", residual_vs_pure)
    write_rows(RESULTS / "ncp_cfc_vs_ltc_summary.csv", cfc_vs_ltc)
    write_rows(RESULTS / "ncp_ablation_scenario_summary.csv", scenario_summary)
    if statistical_comparisons:
        write_rows(RESULTS / "ncp_statistical_comparisons.csv", statistical_comparisons)
    write_summary_markdown(RESULTS / "ncp_ablation_summary.md", grouped_rows, residual_vs_pure,
                           cfc_vs_ltc, statistical_comparisons, all_training_rows, eval_scenarios, args)
    plot_ablation(grouped_rows, FIGURES / "ncp_ablation_success_collision.png")

    print(f"\nWrote {RESULTS / 'ncp_training_log.csv'}")
    print(f"Wrote {RESULTS / 'ncp_ablation_episode_results.csv'}")
    print(f"Wrote {RESULTS / 'ncp_ablation_group_summary.csv'}")
    print(f"Wrote {RESULTS / 'ncp_residual_vs_pure_summary.csv'}")
    print(f"Wrote {RESULTS / 'ncp_cfc_vs_ltc_summary.csv'}")
    print(f"Wrote {RESULTS / 'ncp_ablation_scenario_summary.csv'}")
    if statistical_comparisons:
        print(f"Wrote {RESULTS / 'ncp_statistical_comparisons.csv'}")
    print(f"Wrote {RESULTS / 'ncp_ablation_summary.md'}")
    print(f"Wrote {FIGURES / 'ncp_ablation_success_collision.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--train-sequences", type=int, default=96)
    parser.add_argument("--val-sequences", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--imitation-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imitation-lr", type=float, default=0.003)
    parser.add_argument("--rl-episodes", type=int, default=14)
    parser.add_argument("--rl-max-steps", type=int, default=80)
    parser.add_argument("--rl-lr", type=float, default=0.001)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sparsity", type=float, default=0.50)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of independent training seeds for variance estimation.")
    parser.add_argument("--parallel-seeds", type=int, default=1,
                        help="Number of seeds to train in parallel via ProcessPoolExecutor. "
                             "Default 1 (serial). Pass 0 or a negative number for 'all available' "
                             "(os.cpu_count()). Each worker is pinned to torch.set_num_threads(1) "
                             "to avoid CPU thread contention between workers.")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test preset: single seed, short training, 3 eval episodes. "
                             "Overrides --n-seeds, --eval-episodes, --imitation-epochs and "
                             "--rl-episodes. Useful for CI / local iteration.")
    args = parser.parse_args()
    if args.parallel_seeds <= 0:
        args.parallel_seeds = os.cpu_count() or 1
    if args.quick:
        # Explicit overrides keep the preset semantic (users see what changed).
        args.n_seeds = 1
        args.eval_episodes = 3
        args.imitation_epochs = 2
        args.rl_episodes = 3
        print("[--quick] Using smoke-test preset: n_seeds=1, eval_episodes=3, "
              "imitation_epochs=2, rl_episodes=3")
    run_pipeline(args)


if __name__ == "__main__":
    main()
