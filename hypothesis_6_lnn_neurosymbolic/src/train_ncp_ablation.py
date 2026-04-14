from __future__ import annotations

import argparse
import copy
import csv
import math
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


class OfflineNCPPolicy:
    def __init__(
        self,
        model: NCPDiscreteModel,
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
    progress_history = []
    prev_dist = norm(goal - pos)
    for step in range(max_steps):
        true_ranges = ray_cast(pos, heading, obstacles)
        obs_ranges, uncertainty = noisy_sensor(true_ranges, rng, noise, dropout, bias)
        x = features(pos, heading, goal, obs_ranges, uncertainty)
        action, _scores = controller.act(x, pos, heading, goal, obstacles)
        new_heading = wrap_angle(heading + float(ACTION_DELTAS[action]))
        new_pos = pos + 0.32 * np.array([math.cos(new_heading), math.sin(new_heading)])
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
                controller_name,
                scenario,
                seed,
                0,
                1,
                step + 1,
                path_length,
                min_clearance,
                near_misses,
                0,
                max_steps,
                float(np.mean(progress_history)),
            )
        pos = new_pos
        heading = new_heading
        prev_dist = new_dist
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
                0,
                step + 1,
                float(np.mean(progress_history)),
            )
    return EpisodeResult(
        controller_name,
        scenario,
        seed,
        0,
        0,
        max_steps,
        path_length,
        min_clearance,
        near_misses,
        0,
        max_steps,
        float(np.mean(progress_history)),
    )


def evaluate_models(
    trained: dict[tuple[str, str], NCPDiscreteModel],
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
                )
            )
            for (cell, stage), model in trained.items():
                variants = [
                    ("pure", 0.0, 1.0),
                    ("residual", 1.0, residual_scale),
                ]
                for variant, baseline_scale, ncp_scale in variants:
                    controller = OfflineNCPPolicy(model, base_weights, baseline_scale, ncp_scale)
                    name = f"{cell}_{stage}_{variant}"
                    results.append(run_eval_episode(name, controller, scenario, episode_seed, max_steps))
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
    parts = name.split("_")
    return {"cell": parts[0], "stage": f"{parts[1]}_{parts[2]}" if parts[1] == "rl" else parts[1], "variant": parts[-1]}


def scenario_group(scenario: str) -> str:
    if scenario in HARD_SCENARIOS:
        return "hard"
    if scenario in DEFAULT_SCENARIOS:
        return "default"
    return "other"


def build_group_tables(results: list[EpisodeResult]) -> tuple[list[dict], list[dict]]:
    detail = []
    for result in results:
        row = asdict(result)
        row.update(controller_metadata(result.controller))
        row["scenario_group"] = scenario_group(result.scenario)
        detail.append(row)

    grouped = []
    keys = sorted({(r["controller"], r["scenario_group"]) for r in detail})
    for controller, group in keys:
        rows = [r for r in detail if r["controller"] == controller and r["scenario_group"] == group]
        meta = controller_metadata(controller)
        grouped.append(
            {
                "controller": controller,
                **meta,
                "scenario_group": group,
                "n": len(rows),
                "success_rate": float(np.mean([r["success"] for r in rows])),
                "collision_rate": float(np.mean([r["collision"] for r in rows])),
                "mean_steps": float(np.mean([r["steps"] for r in rows])),
                "mean_min_clearance": float(np.mean([r["min_clearance"] for r in rows])),
                "mean_near_misses": float(np.mean([r["near_misses"] for r in rows])),
                "mean_progress": float(np.mean([r["mean_progress"] for r in rows])),
            }
        )
    return detail, grouped


def build_pairwise_tables(grouped_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key = {(r["cell"], r["stage"], r["variant"], r["scenario_group"]): r for r in grouped_rows}
    residual_vs_pure = []
    for cell in ["cfc", "ltc"]:
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
    training_rows: list[dict],
    scenarios: list[str],
    args: argparse.Namespace,
) -> None:
    hard_rows = [r for r in grouped_rows if r["scenario_group"] == "hard"]
    hard_rows = sorted(hard_rows, key=lambda r: (r["cell"], r["stage"], r["variant"]))
    default_rows = [r for r in grouped_rows if r["scenario_group"] == "default"]
    default_rows = sorted(default_rows, key=lambda r: (r["cell"], r["stage"], r["variant"]))
    train_tail = []
    for cell in ["cfc", "ltc"]:
        for phase in ["imitation", "rl_finetune"]:
            candidates = [r for r in training_rows if r["cell"] == cell and r["phase"] == phase]
            if candidates:
                train_tail.append(candidates[-1])

    formats = {
        "success_rate": ".3f",
        "collision_rate": ".3f",
        "mean_steps": ".1f",
        "mean_min_clearance": ".3f",
        "mean_near_misses": ".1f",
        "mean_progress": ".4f",
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
    }
    lines = [
        "# Saf NCP imitation/RL ve ablation sonuçları",
        "",
        "Bu dosya resmi `ncps.torch` CfC/LTC katmanları ile üretilen offline imitation, kısa policy-gradient fine-tune ve pure/residual ablation sonuçlarını özetler.",
        "",
        "## Deney konfigürasyonu",
        "",
        f"- Eğitim senaryoları: {', '.join(DEFAULT_SCENARIOS)}",
        f"- Test senaryoları: {', '.join(scenarios)}",
        f"- Imitation sequence sayısı: {args.train_sequences}, doğrulama: {args.val_sequences}, sequence length: {args.seq_len}",
        f"- Imitation epoch: {args.imitation_epochs}, RL fine-tune episode: {args.rl_episodes}",
        f"- NCP hidden: {args.hidden_dim}, sparsity: {args.sparsity}, residual scale: {args.residual_scale}",
        "",
        "## Eğitim özeti",
        "",
    ]
    lines.extend(markdown_table(train_tail, ["cell", "phase", "epoch_or_episode", "loss", "val_accuracy", "episode_return", "success", "collision"], formats))
    lines.extend(["", "## Residual vs pure fark tablosu", ""])
    lines.extend(
        markdown_table(
            residual_vs_pure,
            [
                "cell",
                "stage",
                "scenario_group",
                "pure_success",
                "residual_success",
                "delta_success",
                "pure_collision",
                "residual_collision",
                "delta_collision",
                "delta_min_clearance",
            ],
            formats,
        )
    )
    lines.extend(["", "## CfC vs LTC fark tablosu", ""])
    lines.extend(
        markdown_table(
            cfc_vs_ltc,
            [
                "stage",
                "variant",
                "scenario_group",
                "cfc_success",
                "ltc_success",
                "delta_success_cfc_minus_ltc",
                "cfc_collision",
                "ltc_collision",
                "delta_collision_cfc_minus_ltc",
                "delta_min_clearance_cfc_minus_ltc",
            ],
            formats,
        )
    )
    lines.extend(["", "## Hard map ortalaması", ""])
    lines.extend(
        markdown_table(
            hard_rows,
            ["controller", "cell", "stage", "variant", "n", "success_rate", "collision_rate", "mean_steps", "mean_min_clearance", "mean_near_misses"],
            formats,
        )
    )
    lines.extend(["", "## Default map ortalaması", ""])
    lines.extend(
        markdown_table(
            default_rows,
            ["controller", "cell", "stage", "variant", "n", "success_rate", "collision_rate", "mean_steps", "mean_min_clearance", "mean_near_misses"],
            formats,
        )
    )
    lines.extend(
        [
            "",
            "## Bilimsel okuma",
            "",
            "- `pure` varyantında karar tamamen NCP logits ile verilir; bu saf kapasite ve kararlılık testidir.",
            "- `residual` varyantında sabit uzman politikanın üstüne NCP düzeltmesi eklenir; bu daha güvenli dağıtım senaryosunu temsil eder.",
            "- CfC/LTC kıyası aynı veri, seed ailesi ve harita setinde yapılır; farklar mimari ve kısa fine-tune dinamiğinden gelir.",
            "- Bu çalışma hâlâ küçük ölçekli simülasyondur; sonuçlar hipotez taraması için kullanılır, robotik sistem iddiası için daha büyük seed sayısı ve fiziksel validasyon gerekir.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_ablation(grouped_rows: list[dict], out_path: Path) -> None:
    hard = [r for r in grouped_rows if r["scenario_group"] == "hard"]
    order = [
        "fixed_policy",
        "cfc_imitation_pure",
        "cfc_imitation_residual",
        "cfc_rl_finetune_pure",
        "cfc_rl_finetune_residual",
        "ltc_imitation_pure",
        "ltc_imitation_residual",
        "ltc_rl_finetune_pure",
        "ltc_rl_finetune_residual",
    ]
    rows = [next((r for r in hard if r["controller"] == name), None) for name in order]
    rows = [r for r in rows if r is not None]
    labels = [r["controller"].replace("_", "\n") for r in rows]
    success = [r["success_rate"] for r in rows]
    collision = [r["collision_rate"] for r in rows]
    clearance = [r["mean_min_clearance"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].bar(x, success, color="#16837a")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("success")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, collision, color="#b64252")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("collision")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, clearance, color="#555555")
    axes[2].set_ylabel("min clearance")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=0, fontsize=8)
    fig.suptitle("Hard-map ablation: pure vs residual and CfC vs LTC")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    base_weights = train_fixed_policy(np.random.default_rng(args.seed))
    input_dim = base_weights.shape[0]
    train_scenarios = list(DEFAULT_SCENARIOS)
    eval_scenarios = list(DEFAULT_SCENARIOS) + list(HARD_SCENARIOS)

    x_train, y_train = generate_imitation_dataset(rng, train_scenarios, args.train_sequences, args.seq_len)
    x_val, y_val = generate_imitation_dataset(rng, train_scenarios, args.val_sequences, args.seq_len)

    training_rows: list[dict] = []
    trained: dict[tuple[str, str], NCPDiscreteModel] = {}
    for cell in ["cfc", "ltc"]:
        model = NCPDiscreteModel(input_dim, cell, args.hidden_dim, args.sparsity, args.seed + (1 if cell == "cfc" else 2))
        model = train_imitation(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            args.imitation_epochs,
            args.batch_size,
            args.imitation_lr,
            rng,
            cell,
            training_rows,
        )
        trained[(cell, "imitation")] = copy.deepcopy(model).eval()
        save_checkpoint(
            MODELS / f"ncp_{cell}_imitation.pt",
            trained[(cell, "imitation")],
            {"cell": cell, "stage": "imitation", "input_dim": input_dim, "hidden_dim": args.hidden_dim, "sparsity": args.sparsity},
        )

        rl_model = copy.deepcopy(model)
        rl_model = fine_tune_rl(
            rl_model,
            rng,
            train_scenarios,
            args.rl_episodes,
            args.rl_max_steps,
            args.rl_lr,
            args.entropy_coef,
            cell,
            training_rows,
        )
        trained[(cell, "rl_finetune")] = rl_model.eval()
        save_checkpoint(
            MODELS / f"ncp_{cell}_rl_finetune.pt",
            trained[(cell, "rl_finetune")],
            {"cell": cell, "stage": "rl_finetune", "input_dim": input_dim, "hidden_dim": args.hidden_dim, "sparsity": args.sparsity},
        )

    eval_results = evaluate_models(
        trained,
        base_weights,
        eval_scenarios,
        args.eval_episodes,
        args.eval_max_steps,
        args.residual_scale,
        args.seed,
    )
    detail_rows, grouped_rows = build_group_tables(eval_results)
    residual_vs_pure, cfc_vs_ltc = build_pairwise_tables(grouped_rows)
    scenario_summary = aggregate(eval_results)
    scenario_summary = [{**row, **controller_metadata(row["controller"])} for row in scenario_summary]

    write_rows(RESULTS / "ncp_training_log.csv", training_rows)
    write_rows(RESULTS / "ncp_ablation_episode_results.csv", detail_rows)
    write_rows(RESULTS / "ncp_ablation_group_summary.csv", grouped_rows)
    write_rows(RESULTS / "ncp_residual_vs_pure_summary.csv", residual_vs_pure)
    write_rows(RESULTS / "ncp_cfc_vs_ltc_summary.csv", cfc_vs_ltc)
    write_rows(RESULTS / "ncp_ablation_scenario_summary.csv", scenario_summary)
    write_summary_markdown(RESULTS / "ncp_ablation_summary.md", grouped_rows, residual_vs_pure, cfc_vs_ltc, training_rows, eval_scenarios, args)
    plot_ablation(grouped_rows, FIGURES / "ncp_ablation_success_collision.png")

    print(f"Wrote {RESULTS / 'ncp_training_log.csv'}")
    print(f"Wrote {RESULTS / 'ncp_ablation_episode_results.csv'}")
    print(f"Wrote {RESULTS / 'ncp_ablation_group_summary.csv'}")
    print(f"Wrote {RESULTS / 'ncp_residual_vs_pure_summary.csv'}")
    print(f"Wrote {RESULTS / 'ncp_cfc_vs_ltc_summary.csv'}")
    print(f"Wrote {RESULTS / 'ncp_ablation_scenario_summary.csv'}")
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
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
