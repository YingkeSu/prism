"""
Integration tests for sensor degradation injection in UAVMultimodalEnv.
"""

from __future__ import annotations

import numpy as np

from envs.uav_multimodal_env import UAVMultimodalEnv


def test_degradation_info_keys_present():
    env = UAVMultimodalEnv(max_steps=20, degradation_level=0.5)
    obs, info = env.reset(seed=123)
    assert "degradation_level" in info
    assert "degradation" in info

    action = np.zeros(4, dtype=np.float32)
    _, _, done, truncated, step_info = env.step(action)
    assert "degradation_level" in step_info
    assert "degradation" in step_info
    assert "lidar_dropout_ratio" in step_info["degradation"]
    assert "rgb_occlusion_ratio" in step_info["degradation"]
    assert "imu_dropout_dims" in step_info["degradation"]
    assert done in (True, False)
    assert truncated in (True, False)


def test_degradation_changes_observation_distribution():
    seed = 42
    action = np.array([0.2, -0.1, 0.1, 0.0], dtype=np.float32)

    clean_env = UAVMultimodalEnv(max_steps=30, degradation_level=0.0)
    degraded_env = UAVMultimodalEnv(max_steps=30, degradation_level=0.8)

    clean_obs, _ = clean_env.reset(seed=seed)
    degraded_obs, _ = degraded_env.reset(seed=seed)
    for _ in range(5):
        clean_obs, _, d1, t1, _ = clean_env.step(action)
        degraded_obs, _, d2, t2, _ = degraded_env.step(action)
        if d1 or t1:
            clean_obs, _ = clean_env.reset(seed=seed)
        if d2 or t2:
            degraded_obs, _ = degraded_env.reset(seed=seed)

    rgb_diff = float(np.mean(np.abs(clean_obs["rgb"].astype(np.float32) - degraded_obs["rgb"].astype(np.float32))))
    lidar_diff = float(np.mean(np.abs(clean_obs["lidar"] - degraded_obs["lidar"])))
    imu_diff = float(np.mean(np.abs(clean_obs["imu"] - degraded_obs["imu"])))

    assert rgb_diff > 1.0
    assert lidar_diff > 0.01
    assert imu_diff > 0.001


def test_difficulty_presets_change_environment_configuration():
    easy_env = UAVMultimodalEnv(max_steps=20, difficulty="easy")
    hard_env = UAVMultimodalEnv(max_steps=20, difficulty="hard")

    assert hard_env.num_obstacles > easy_env.num_obstacles
    assert hard_env.goal_radius < easy_env.goal_radius
    assert hard_env.world_bound < easy_env.world_bound


def test_invalid_difficulty_raises_value_error():
    try:
        UAVMultimodalEnv(max_steps=10, difficulty="unknown")
    except ValueError as exc:
        assert "Unknown difficulty" in str(exc)
        return
    raise AssertionError("Expected ValueError for unknown difficulty")
