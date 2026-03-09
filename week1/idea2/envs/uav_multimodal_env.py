"""uav_multimodal_env.py

Minimal multi-modal UAV environment for SB3-style experiments.

Observation is a Dict with three modalities:
- lidar: (N, 3) point cloud
- rgb: (H, W, 3) RGB image (channel-last)
- imu: (6,) IMU vector

This environment is intentionally lightweight and synthetic; it exists to
exercise the multi-modal feature extractor and SB3 integration scripts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UAVMultimodalEnv(gym.Env):
    """
    Synthetic UAV environment with multi-modal observations.

    Observation: Dict[str, np.ndarray]
        - 'lidar': (N, 3) float32
        - 'rgb': (H, W, 3) float32 in [0, 255]
        - 'imu': (6,) float32

    Action: Box(-1, 1, (4,))
        - (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)

    Args:
        max_steps: Episode length cap.
        render_mode: None | 'human' | 'rgb_array'.
        lidar_num_points: Number of LiDAR points N.
        rgb_size: (H, W) image size.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video_frames": 0}

    def __init__(
        self,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        lidar_num_points: int = 1000,
        rgb_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode={render_mode!r}. "
                f"Expected one of {self.metadata['render_modes']}."
            )

        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.lidar_num_points = int(lidar_num_points)
        self.rgb_size = (int(rgb_size[0]), int(rgb_size[1]))

        # Pre-compute RGB gradients for performance
        h, w = self.rgb_size
        grad_x = np.linspace(0.0, 30.0, w, dtype=np.float32)[None, :]
        grad_y = np.linspace(0.0, 30.0, h, dtype=np.float32)[:, None]
        self._rgb_grad = grad_x + grad_y  # (h, w)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=-50.0,
                    high=50.0,
                    shape=(self.lidar_num_points, 3),
                    dtype=np.float32,
                ),
                "rgb": spaces.Box(
                    low=0.0,
                    high=255.0,
                    shape=(self.rgb_size[0], self.rgb_size[1], 3),
                    dtype=np.float32,
                ),
                "imu": spaces.Box(
                    low=-20.0,
                    high=20.0,
                    shape=(6,),
                    dtype=np.float32,
                ),
            }
        )

        # Action space: (vx, vy, vz, yaw_rate)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Simple internal state
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._yaw = np.float32(0.0)
        self._step_count = 0

        self._goal = np.array([5.0, 5.0, 2.0], dtype=np.float32)
        self._goal_radius = np.float32(0.5)
        self._last_dist: Optional[float] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._yaw = np.float32(0.0)
        self._step_count = 0
        self._last_dist = float(np.linalg.norm(self._pos - self._goal))

        obs = self._get_obs()
        info: Dict[str, Any] = {"distance_to_goal": self._last_dist}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)
        action = action.astype(np.float32, copy=False)
        if action.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action.shape}.")

        self._step_count += 1

        dt = np.float32(0.1)
        vel_scale = np.float32(1.0)
        yaw_rate_scale = np.float32(1.0)

        vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd = action
        self._vel = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=np.float32) * vel_scale
        self._yaw = np.float32(self._yaw + yaw_rate_cmd * yaw_rate_scale * dt)
        self._pos = (self._pos + self._vel * dt).astype(np.float32, copy=False)

        dist = float(np.linalg.norm(self._pos - self._goal))
        progress = 0.0 if self._last_dist is None else (self._last_dist - dist)
        self._last_dist = dist

        reward = float(progress * 10.0 - 0.01 * dist)

        terminated = dist <= float(self._goal_radius)
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "distance_to_goal": dist,
            "is_success": bool(terminated),
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            print(f"step={self._step_count} pos={self._pos.tolist()} goal={self._goal.tolist()}")
            return None

        if self.render_mode == "rgb_array":
            h, w = self.rgb_size
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Draw a simple marker based on normalized position.
            x = int(np.clip((self._pos[0] + 10.0) / 20.0 * (w - 1), 0, w - 1))
            y = int(np.clip((self._pos[1] + 10.0) / 20.0 * (h - 1), 0, h - 1))
            frame[y, x, :] = np.array([255, 0, 0], dtype=np.uint8)
            return frame

        raise RuntimeError(f"Unhandled render_mode={self.render_mode!r}.")

    def _get_obs(self) -> Dict[str, np.ndarray]:
        rng = self.np_random

        # LiDAR: smaller point cloud for speed
        lidar = rng.standard_normal(size=(self.lidar_num_points, 3)).astype(np.float32)
        lidar = lidar * 5.0 + self._pos

        # RGB: use pre-computed gradients
        rgb = np.clip(self._rgb_grad[..., None] + rng.uniform(0, 5, size=(*self.rgb_size, 1)), 0, 255).astype(np.float32)
        rgb = np.broadcast_to(rgb, (*self.rgb_size, 3))

        # IMU
        imu = np.zeros(6, dtype=np.float32)
        imu[0:3] = self._vel
        imu[3:6] = np.array([0.0, 0.0, float(self._yaw)], dtype=np.float32)

        return {"lidar": lidar, "rgb": rgb, "imu": imu}


def test_uav_multimodal_env() -> None:
    """Minimal smoke test."""
    env = UAVMultimodalEnv(max_steps=5)
    obs, _ = env.reset()
    assert set(obs.keys()) == {"lidar", "rgb", "imu"}
    assert obs["lidar"].shape == (1000, 3)
    assert obs["rgb"].shape == (128, 128, 3)
    assert obs["imu"].shape == (6,)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    print("✅ UAVMultimodalEnv test passed")


if __name__ == "__main__":
    test_uav_multimodal_env()
