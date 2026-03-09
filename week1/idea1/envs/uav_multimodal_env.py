"""
UAV Multimodal Environment

Multi-modal UAV simulation environment with LiDAR, RGB, and IMU sensors.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


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

    DEGRADATION_PRESETS = {
        "none": 0.0,
        "mild": 0.2,
        "moderate": 0.5,
        "severe": 0.8,
    }

    DIFFICULTY_PRESETS = {
        "easy": {
            "goal_position": (3.0, 3.0, 2.0),
            "goal_radius": 1.5,
            "num_obstacles": 5,
            "world_bound": 10.0,
            "obstacle_pos_range": (2.0, 7.0),
            "obstacle_radius_range": (0.3, 0.6),
        },
        "medium": {
            "goal_position": (4.0, 4.0, 2.5),
            "goal_radius": 1.0,
            "num_obstacles": 8,
            "world_bound": 10.0,
            "obstacle_pos_range": (1.5, 8.0),
            "obstacle_radius_range": (0.35, 0.75),
        },
        "hard": {
            "goal_position": (5.5, 5.5, 3.0),
            "goal_radius": 0.7,
            "num_obstacles": 12,
            "world_bound": 8.0,
            "obstacle_pos_range": (1.0, 8.5),
            "obstacle_radius_range": (0.4, 0.9),
        },
    }

    def __init__(
        self,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        degradation_level: float = 0.0,
        degradation_profile: Optional[Dict[str, float]] = None,
        difficulty: str = "easy",
    ):
        super().__init__()

        self.max_steps = max_steps
        self.dt = 0.1
        self.render_mode = render_mode

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
        self.step_count = 0

        # Cached RGB rendering geometry to avoid rebuilding per step.
        self._rgb_height = 128
        self._rgb_width = 128
        self._rgb_scale = 6.4
        self._rgb_origin = (64, 64)
        self._rgb_y, self._rgb_x = np.ogrid[:self._rgb_height, :self._rgb_width]
        
        self.last_distance = None

        if difficulty not in self.DIFFICULTY_PRESETS:
            raise ValueError(
                f"Unknown difficulty: {difficulty}. "
                f"Available: {', '.join(sorted(self.DIFFICULTY_PRESETS.keys()))}"
            )
        self.difficulty = difficulty
        preset = self.DIFFICULTY_PRESETS[difficulty]
        self.initial_goal_position = np.array(preset["goal_position"], dtype=np.float32)
        self.initial_goal_radius = float(preset["goal_radius"])
        self.num_obstacles = int(preset["num_obstacles"])
        self.world_bound = float(preset["world_bound"])
        self.obstacle_pos_range = (
            float(preset["obstacle_pos_range"][0]),
            float(preset["obstacle_pos_range"][1]),
        )
        self.obstacle_radius_range = (
            float(preset["obstacle_radius_range"][0]),
            float(preset["obstacle_radius_range"][1]),
        )

        self.goal_position = self.initial_goal_position.copy()
        self.goal_radius = self.initial_goal_radius

        self.degradation_level = float(np.clip(degradation_level, 0.0, 1.0))
        self.degradation_profile = self._default_degradation_profile()
        if degradation_profile is not None:
            self.degradation_profile.update(degradation_profile)

        self.obstacles = []
        self._generate_obstacles()

    def _generate_obstacles(self):
        """Generate random obstacles in the environment"""
        low, high = self.obstacle_pos_range
        r_low, r_high = self.obstacle_radius_range
        for _ in range(self.num_obstacles):
            pos = self.np_random.uniform(low, high, size=3)
            radius = self.np_random.uniform(r_low, r_high)
            self.obstacles.append({'pos': pos, 'radius': radius})

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.step_count = 0
        
        self.last_distance = np.linalg.norm(self.position - self.goal_position)

        self.goal_position = self.initial_goal_position.copy()
        self.goal_radius = self.initial_goal_radius

        self.obstacles = []
        self._generate_obstacles()

        observation, degradation_info = self._get_observation_with_info()
        return observation, {
            "degradation_level": self.degradation_level,
            "degradation": degradation_info,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        vx, vy, vz, omega = action

        self.velocity = self.velocity * 0.9 + np.array([vx, vy, vz]) * 0.1
        self.position = self.position + self.velocity * self.dt
        self.orientation[2] = self.orientation[2] + omega * self.dt
        self.step_count += 1

        reward = self._compute_reward()
        done, truncated, termination_reason = self._termination_status()

        observation, degradation_info = self._get_observation_with_info()

        info = {
            'distance_to_goal': np.linalg.norm(self.position - self.goal_position),
            'position': self.position.copy(),
            'termination_reason': termination_reason,
            'is_success': termination_reason == 'success',
            'is_collision': termination_reason == 'collision',
            'is_out_of_bounds': termination_reason == 'out_of_bounds',
            'is_time_limit': termination_reason == 'time_limit',
            'degradation_level': self.degradation_level,
            'degradation': degradation_info,
        }

        return observation, reward, done, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        observation, _ = self._get_observation_with_info()
        return observation

    def _get_observation_with_info(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        lidar = self._generate_lidar()
        rgb = self._generate_rgb()
        imu = self._generate_imu()
        lidar, rgb, imu, degradation_info = self._apply_degradation(lidar, rgb, imu)

        return {
            'lidar': lidar.astype(np.float32),
            'rgb': rgb.astype(np.uint8),
            'imu': imu.astype(np.float32)
        }, degradation_info

    def _generate_lidar(self) -> np.ndarray:
        num_points = 1000
        points = self.np_random.uniform(-10, 10, size=(num_points, 3))

        for obs in self.obstacles:
            dist = np.linalg.norm(points - obs['pos'], axis=1)
            mask = dist < obs['radius']
            if np.any(mask):
                points[mask] = obs['pos'] + self.np_random.normal(0, 0.1, size=(int(mask.sum()), 3))

        return points

    def _generate_rgb(self) -> np.ndarray:
        """
        Generate RGB observation showing agent, goal, obstacles, and trajectory.

        Visual elements:
        - Red circle: Goal position
        - Blue circle: Agent current position
        - Gray circles: Obstacles
        - Green dots: Recent trajectory (last 50 steps)
        """
        rgb = np.zeros((self._rgb_height, self._rgb_width, 3), dtype=np.float32)

        # Draw goal (red circle)
        goal_center = (
            int(self.goal_position[0] * self._rgb_scale) + self._rgb_origin[0],
            int(self.goal_position[1] * self._rgb_scale) + self._rgb_origin[1],
        )
        goal_radius = int(self.goal_radius * self._rgb_scale)
        goal_mask = (self._rgb_x - goal_center[0])**2 + (self._rgb_y - goal_center[1])**2 <= goal_radius**2
        rgb[goal_mask] = [255, 0, 0]

        # Draw agent (blue circle)
        agent_center = (
            int(self.position[0] * self._rgb_scale) + self._rgb_origin[0],
            int(self.position[1] * self._rgb_scale) + self._rgb_origin[1],
        )
        agent_radius = 4
        agent_mask = (self._rgb_x - agent_center[0])**2 + (self._rgb_y - agent_center[1])**2 <= agent_radius**2
        rgb[agent_mask] = [0, 0, 255]

        # Draw obstacles (gray circles)
        for obs in self.obstacles:
            obs_center = (
                int(obs['pos'][0] * self._rgb_scale) + self._rgb_origin[0],
                int(obs['pos'][1] * self._rgb_scale) + self._rgb_origin[1],
            )
            obs_radius = int(obs['radius'] * self._rgb_scale)
            obs_mask = (self._rgb_x - obs_center[0])**2 + (self._rgb_y - obs_center[1])**2 <= obs_radius**2
            rgb[obs_mask] = [128, 128, 128]

        # Draw origin marker (yellow cross)
        origin_center = (64, 64)
        rgb[62:66, 63:65] = [255, 255, 0]  # Horizontal line
        rgb[63:65, 62:66] = [255, 255, 0]  # Vertical line

        return rgb

    def _generate_imu(self) -> np.ndarray:
        acc = self.velocity + self.np_random.normal(0, 0.01, size=3)
        gyro = self.angular_velocity + self.np_random.normal(0, 0.01, size=3)

        return np.concatenate([acc, gyro])

    def _default_degradation_profile(self) -> Dict[str, float]:
        return {
            # RGB degradation
            "rgb_noise_std": 30.0,
            "rgb_occlusion_prob": 0.8,
            "rgb_occlusion_size_ratio": 0.45,
            "rgb_blur_passes": 3.0,
            # LiDAR degradation
            "lidar_dropout_ratio": 0.7,
            "lidar_noise_std": 0.8,
            # IMU degradation
            "imu_noise_std": 0.2,
            "imu_drift_per_step": 0.01,
            "imu_dropout_prob": 0.4,
        }

    def set_degradation(self, level: float) -> None:
        """Dynamically update degradation level in [0, 1]."""
        self.degradation_level = float(np.clip(level, 0.0, 1.0))

    def _box_blur_rgb(self, rgb: np.ndarray, passes: int) -> np.ndarray:
        if passes <= 0:
            return rgb
        x = rgb.astype(np.float32)
        for _ in range(passes):
            padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
            x = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        return x

    def _apply_degradation(
        self,
        lidar: np.ndarray,
        rgb: np.ndarray,
        imu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        level = self.degradation_level
        if level <= 0.0:
            return lidar, rgb, imu, {
                "lidar_dropout_ratio": 0.0,
                "rgb_occlusion_ratio": 0.0,
                "imu_dropout_dims": 0.0,
            }

        profile = self.degradation_profile

        # LiDAR: random point dropout + Gaussian jitter
        lidar_out = lidar.copy()
        dropout_ratio = float(np.clip(level * profile["lidar_dropout_ratio"], 0.0, 0.95))
        keep_mask = self.np_random.random(lidar_out.shape[0]) > dropout_ratio
        if not np.any(keep_mask):
            keep_mask[self.np_random.integers(0, lidar_out.shape[0])] = True
        lidar_out[~keep_mask] = 0.0
        lidar_noise_std = level * profile["lidar_noise_std"]
        if lidar_noise_std > 0:
            lidar_out = lidar_out + self.np_random.normal(0.0, lidar_noise_std, size=lidar_out.shape)

        # RGB: blur + additive noise + block occlusion
        rgb_out = rgb.astype(np.float32)
        blur_passes = int(round(level * profile["rgb_blur_passes"]))
        rgb_out = self._box_blur_rgb(rgb_out, blur_passes)

        rgb_noise_std = level * profile["rgb_noise_std"]
        if rgb_noise_std > 0:
            rgb_out = rgb_out + self.np_random.normal(0.0, rgb_noise_std, size=rgb_out.shape)

        occlusion_ratio = 0.0
        if self.np_random.random() < level * profile["rgb_occlusion_prob"]:
            h, w, _ = rgb_out.shape
            occ_h = max(4, int(h * (0.1 + level * profile["rgb_occlusion_size_ratio"])))
            occ_w = max(4, int(w * (0.1 + level * profile["rgb_occlusion_size_ratio"])))
            y0 = int(self.np_random.integers(0, max(1, h - occ_h + 1)))
            x0 = int(self.np_random.integers(0, max(1, w - occ_w + 1)))
            rgb_out[y0:y0 + occ_h, x0:x0 + occ_w, :] = 0.0
            occlusion_ratio = float((occ_h * occ_w) / float(h * w))
        rgb_out = np.clip(rgb_out, 0.0, 255.0)

        # IMU: Gaussian noise + drift + random channel dropout
        imu_out = imu.astype(np.float32).copy()
        imu_noise_std = level * profile["imu_noise_std"]
        if imu_noise_std > 0:
            imu_out += self.np_random.normal(0.0, imu_noise_std, size=imu_out.shape)

        drift_per_step = level * profile["imu_drift_per_step"]
        if drift_per_step > 0:
            drift = self.step_count * drift_per_step
            imu_out[:3] += drift
            imu_out[3:] += drift * 0.5

        imu_dropout_dims = 0
        if self.np_random.random() < level * profile["imu_dropout_prob"]:
            num_dims = int(self.np_random.integers(1, 4))
            drop_idx = self.np_random.choice(imu_out.shape[0], size=num_dims, replace=False)
            imu_out[drop_idx] = 0.0
            imu_dropout_dims = int(num_dims)

        degradation_info = {
            "lidar_dropout_ratio": 1.0 - float(np.mean(keep_mask)),
            "rgb_occlusion_ratio": occlusion_ratio,
            "imu_dropout_dims": float(imu_dropout_dims),
        }
        return lidar_out, rgb_out, imu_out, degradation_info

    def _compute_reward(self) -> float:
        dist_to_goal = np.linalg.norm(self.position - self.goal_position)
        
        # Progress reward: strongly reward getting closer to goal
        if self.last_distance is not None:
            distance_improved = self.last_distance - dist_to_goal
            progress_reward = distance_improved * 50.0  # Increased from 10.0
        else:
            progress_reward = 0.0
        
        self.last_distance = dist_to_goal

        # Directional reward: reward velocity aligned with goal direction
        if np.linalg.norm(self.velocity) > 0.001:
            to_goal = self.goal_position - self.position
            to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-8)
            velocity_norm = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
            alignment = np.dot(velocity_norm, to_goal_norm)
            directional_reward = alignment * 2.0  # Reward moving toward goal
        else:
            directional_reward = 0.0

        # Very small distance penalty
        distance_penalty = -0.01 * dist_to_goal

        collision_penalty = 0
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.position - obs['pos'])
            if dist_to_obs < obs['radius']:
                collision_penalty = -50  # Reduced from -100

        speed_penalty = -0.001 * np.linalg.norm(self.velocity)  # Reduced penalty

        goal_reward = 0
        if dist_to_goal < self.goal_radius:
            goal_reward = 500  # Increased from 200

        # Small bonus for just being alive
        alive_bonus = 0.1

        return progress_reward + directional_reward + distance_penalty + collision_penalty + speed_penalty + goal_reward + alive_bonus

    def _termination_status(self) -> Tuple[bool, bool, Optional[str]]:
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.position - obs['pos'])
            if dist_to_obs < obs['radius']:
                return True, False, 'collision'

        dist_to_goal = np.linalg.norm(self.position - self.goal_position)
        if dist_to_goal < self.goal_radius:
            return True, False, 'success'

        if np.any(np.abs(self.position) > self.world_bound):
            return True, False, 'out_of_bounds'

        if self.step_count >= self.max_steps:
            return False, True, 'time_limit'

        return False, False, None

    def _is_done(self) -> Tuple[bool, bool]:
        """Backward-compatible done API used by older scripts/tests."""
        done, truncated, _ = self._termination_status()
        return done, truncated

    def render(self):
        """Render the environment.

        Returns:
            - rgb_array mode: Returns numpy array of the RGB image
            - human mode: Displays the environment (not implemented)
        """
        if self.render_mode == 'rgb_array':
            # Generate and return the RGB observation
            rgb_obs = self._generate_rgb()
            return rgb_obs
        elif self.render_mode == 'human':
            # For human mode, we would display the environment
            # For now, just pass
            pass
        else:
            # Default behavior - return None
            return None


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
