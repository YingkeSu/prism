"""
UAV Multimodal Environment

Multi-modal UAV simulation environment with LiDAR, RGB, and IMU sensors.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any


class UAVMultimodalEnv(gym.Env):
    """
    Multi-modal UAV Simulation Environment

    Observation Space:
    - lidar: (1000, 3) Point cloud
    - rgb: (128, 128, 3) RGB image
    - imu: (6,) IMU data

    Action Space:
    - velocity: (4,) Linear and angular velocity
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'video_frames': 0}

    def __init__(self, max_steps: int = 1000):
        super().__init__()

        self.max_steps = max_steps
        self.dt = 0.1

        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0, 100, (1000, 3), dtype=np.float32),
            "rgb": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            "imu": spaces.Box(-10, 10, (6,), dtype=np.float32)
        })

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        self.goal_position = np.array([8.0, 8.0, 5.0])
        self.goal_radius = 1.0

        self.obstacles = []
        self._generate_obstacles()

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().__init__(seed=seed)

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        self.goal_position = np.array([8.0, 8.0, 5.0])
        self.goal_radius = 1.0

        self.obstacles = []
        self._generate_obstacles()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        vx, vy, vz, omega = action

        self.velocity = self.velocity * 0.9 + np.array([vx, vy, vz]) * 0.1
        self.position = self.position + self.velocity * self.dt
        self.orientation[2] = self.orientation[2] + omega * self.dt

        reward = self._compute_reward()
        done, truncated = self._is_done()

        info = {
            'distance_to_goal': np.linalg.norm(self.position - self.goal_position),
            'position': self.position.copy()
        }

        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        lidar = self._generate_lidar()
        rgb = self._generate_rgb()
        imu = self._generate_imu()

        return {
            'lidar': lidar.astype(np.float32),
            'rgb': rgb.astype(np.uint8),
            'imu': imu.astype(np.float32)
        }

    def _generate_lidar(self) -> np.ndarray:
        num_points = 1000
        points = np.random.uniform(-10, 10, size=(num_points, 3))

        for obs in self.obstacles:
            for i in range(num_points):
                dist = np.linalg.norm(points[i] - obs['pos'])
                if dist < obs['radius']:
                    points[i] = obs['pos'] + np.random.normal(0, 0.1, 3)

        return points

    def _generate_rgb(self) -> np.ndarray:
        height, width = 128, 128
        rgb = np.zeros((height, width, 3), dtype=np.float32)

        for obs in self.obstacles:
            center = (int(obs['pos'][0] * 6.4) + 64, int(obs['pos'][1] * 6.4) + 64)
            center = np.clip(center, 0, width-64, 0, height-64)

            goal_center = (int(self.goal_position[0] * 6.4) + 64, int(self.goal_position[1] * 6.4) + 64)
            goal_radius = int(self.goal_radius * 6.4)

            # Draw red circle for goal
            y, x = np.ogrid(height, width)
            mask = (x - goal_center[0])**2 + (y - goal_center[1])**2 <= goal_radius**2

            rgb[mask] = [255, 0, 0]

        return rgb

    def _generate_imu(self) -> np.ndarray:
        acc = self.velocity + np.random.normal(0, 0.01, size=6)
        gyro = self.angular_velocity + np.random.normal(0, 0.01, size=3)

        return np.concatenate([acc, gyro])

    def _compute_reward(self) -> float:
        dist_to_goal = np.linalg.norm(self.position - self.goal_position)
        distance_reward = -dist_to_goal

        collision_penalty = 0
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.position - obs['pos'])
            if dist_to_obs < obs['radius']:
                collision_penalty = -100

        speed_penalty = -0.01 * np.linalg.norm(self.velocity)

        goal_reward = 0
        if dist_to_goal < self.goal_radius:
            goal_reward = 100

        return distance_reward + collision_penalty + speed_penalty + goal_reward

    def _is_done(self) -> Tuple[bool, bool]:
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.position - obs['pos'])
            if dist_to_obs < obs['radius']:
                return True, False

        dist_to_goal = np.linalg.norm(self.position - self.goal_position)
        if dist_to_goal < self.goal_radius:
            return True, False

        if np.any(np.abs(self.position) > 10):
            return True, False

        if self.step_count >= self.max_steps:
            return False, True

        return False, False


def test_uav_multimodal_env():
    """Test UAV Multimodal Environment"""
    env = UAVMultimodalEnv(max_steps=50)
    obs, info = env.reset()

    print(f"Observation keys: {obs.keys()}")
    print(f"LiDAR shape: {obs['lidar'].shape}")
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"IMU shape: {obs['imu'].shape}")

    for step in range(30):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.2f}, distance={info['distance_to_goal']:.2f}")

        if done or truncated:
            print(f"Episode finished after {step}")
            break

    print("✅ UAV Multimodal Environment test passed")


if __name__ == "__main__":
    test_uav_multimodal_env()
