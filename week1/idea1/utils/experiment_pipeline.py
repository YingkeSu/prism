"""
Unified training/evaluation helpers for reproducible experiments.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def create_sac_model(
    env: Any,
    extractor_class: Type[BaseFeaturesExtractor],
    *,
    features_dim: int = 256,
    use_reliability: bool = True,
    use_reliability_modulation: bool = True,
    num_heads: int = 8,
    fixed_weights: Optional[Sequence[float]] = None,
    debug_mode: bool = False,
    imu_history_len: int = 16,
    share_features_extractor: bool = True,
    learning_rate: float = 3e-4,
    buffer_size: int = 50_000,
    batch_size: int = 8,
    learning_starts: int = 100,
    train_freq: int = 4,
    gradient_steps: int = 1,
    tensorboard_log: Optional[str] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
) -> SAC:
    """Create a SAC model with project-friendly defaults."""
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "use_reliability": use_reliability,
            "use_reliability_modulation": use_reliability_modulation,
            "num_heads": num_heads,
            "fixed_weights": fixed_weights,
            "imu_history_len": max(1, int(imu_history_len)),
            "debug_mode": debug_mode,
        },
        "share_features_extractor": share_features_extractor,
    }

    return SAC(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=(train_freq, "step"),
        gradient_steps=gradient_steps,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )


def train_model(
    model: SAC,
    *,
    total_timesteps: int,
    log_interval: Optional[int] = None,
    tb_log_name: Optional[str] = None,
    callback: Any = None,
) -> float:
    """Train and return elapsed time in seconds."""
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        tb_log_name=tb_log_name,
        callback=callback,
    )
    return time.time() - start


def evaluate_model(
    model: SAC,
    env: Any,
    *,
    num_episodes: int = 50,
    deterministic: bool = True,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Evaluate SAC model and return aggregate metrics.

    Expected info keys from env.step:
      - termination_reason: success/collision/out_of_bounds/time_limit/unknown
      - distance_to_goal
    """
    rewards = []
    steps = []
    final_distances = []
    inference_ms = []
    lidar_dropout_values = []
    rgb_occlusion_values = []
    imu_dropout_values = []

    reason_counts: Dict[str, int] = {
        "success": 0,
        "collision": 0,
        "out_of_bounds": 0,
        "time_limit": 0,
        "unknown": 0,
    }

    for episode in range(num_episodes):
        episode_seed = (seed + episode) if seed is not None else None
        obs, _ = env.reset(seed=episode_seed)
        done = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        last_info: Dict[str, Any] = {}

        while not (done or truncated):
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=deterministic)
            inference_ms.append((time.perf_counter() - t0) * 1000.0)

            obs, reward, done, truncated, info = env.step(action)
            last_info = info
            episode_reward += float(reward)
            episode_steps += 1
            degradation = info.get("degradation")
            if isinstance(degradation, dict):
                lidar_dropout_values.append(float(degradation.get("lidar_dropout_ratio", 0.0)))
                rgb_occlusion_values.append(float(degradation.get("rgb_occlusion_ratio", 0.0)))
                imu_dropout_values.append(float(degradation.get("imu_dropout_dims", 0.0)))

        reason = last_info.get("termination_reason", "unknown")
        if reason not in reason_counts:
            reason = "unknown"
        reason_counts[reason] += 1

        rewards.append(episode_reward)
        steps.append(episode_steps)
        final_distances.append(float(last_info.get("distance_to_goal", np.nan)))

    num_episodes_f = float(max(num_episodes, 1))
    return {
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "avg_final_distance": float(np.nanmean(final_distances)),
        "inference_ms_mean": float(np.mean(inference_ms)),
        "inference_ms_p95": float(np.percentile(inference_ms, 95)),
        "success_rate": reason_counts["success"] / num_episodes_f,
        "collision_rate": reason_counts["collision"] / num_episodes_f,
        "out_of_bounds_rate": reason_counts["out_of_bounds"] / num_episodes_f,
        "time_limit_rate": reason_counts["time_limit"] / num_episodes_f,
        "avg_lidar_dropout_ratio": float(np.mean(lidar_dropout_values)) if lidar_dropout_values else 0.0,
        "avg_rgb_occlusion_ratio": float(np.mean(rgb_occlusion_values)) if rgb_occlusion_values else 0.0,
        "avg_imu_dropout_dims": float(np.mean(imu_dropout_values)) if imu_dropout_values else 0.0,
        "termination_counts": reason_counts,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_results_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Save flat summary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
