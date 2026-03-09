"""
Simple 2D Obstacle Environment

Simplified 2D UAV navigation environment for concept verification.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any


class Simple2DObstacleEnv(gym.Env):
    """
    2D Plane UAV Obstacle Avoidance Environment

    State: (x, y, vx, vy) - Position and velocity
    Action: (ax, ay) - Acceleration control
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'video_frames': 0}

    def __init__(self, num_obstacles: int = 5, max_steps: int = 1000):
        super().__init__()
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.dt = 0.1

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(4,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self.goal = np.array([8.0, 8.0])
        self.goal_radius = 1.0

        self.state = None
        self.step_count = 0
        self.obstacles = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0

        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = np.random.uniform(2, 7, size=2)
            radius = np.random.uniform(0.3, 0.6)
            self.obstacles.append({'pos': pos, 'radius': radius})

        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        x, y, vx, vy = self.state

        ax, ay = action

        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt

        self.state = np.array([x_new, y_new, vx_new, vy_new], dtype=np.float32)
        self.step_count += 1

        reward = self._compute_reward()
        done, truncated = self._is_done()

        info = {'distance_to_goal': np.linalg.norm([x_new, y_new] - self.goal)}

        return self.state.copy(), reward, done, truncated, info

    def _compute_reward(self) -> float:
        x, y, _, _ = self.state

        dist_to_goal = np.linalg.norm([x, y] - self.goal)
        distance_reward = -dist_to_goal

        collision_penalty = 0
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm([x, y] - obs['pos'])
            if dist_to_obs < obs['radius']:
                collision_penalty = -100

        _, _, vx, vy = self.state
        speed = np.sqrt(vx**2 + vy**2)
        speed_penalty = -0.1 * speed

        goal_reward = 0
        if dist_to_goal < self.goal_radius:
            goal_reward = 100

        return distance_reward + collision_penalty + speed_penalty + goal_reward

    def _is_done(self) -> Tuple[bool, bool]:
        x, y, _, _ = self.state

        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm([x, y] - obs['pos'])
            if dist_to_obs < obs['radius']:
                return True, False

        dist_to_goal = np.linalg.norm([x, y] - self.goal)
        if dist_to_goal < self.goal_radius:
            return True, False

        if x < -10 or x > 10 or y < -10 or y > 10:
            return True, False

        if self.step_count >= self.max_steps:
            return False, True

        return False, False


def test_simple_2d_env():
    """Test 2D Environment"""
    env = Simple2DObstacleEnv(num_obstacles=3)
    obs, info = env.reset()

    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.2f}, distance={info['distance_to_goal']:.2f}")

        if done or truncated:
            print(f"Episode finished at step {step}")
            break

    print("✅ Simple 2D Environment test passed")


if __name__ == "__main__":
    test_simple_2d_env()
