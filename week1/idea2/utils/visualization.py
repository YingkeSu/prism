"""
Visualization utilities for training monitoring.

This module provides callbacks and utilities for comprehensive training visualization
with TensorBoard support, including agent-environment interaction visualization.
"""

from typing import Optional, List, Dict
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import json


class VisualizationCallback(BaseCallback):
    """
    Enhanced callback for visualization with TensorBoard support and agent-environment interaction tracking.

    Features:
    - Episode tracking (rewards, lengths)
    - 3D trajectory recording and visualization
    - Action logging (velocity commands, fusion weights)
    - TensorBoard logging integration
    - Configurable logging frequencies
    - Debug output with progress tracking
    - Support for custom metrics
    - Trajectory export to JSON for post-processing

    Args:
        debug: If True, print detailed episode information (default: True)
        log_every_steps: Print progress every N steps during episodes (default: 100)
        log_dir: Directory to save TensorBoard logs and trajectory data (default: None, uses SB3's logger)
        save_trajectories: If True, save trajectory data to JSON files (default: True)
        trajectory_log_interval: Save trajectory every N episodes (default: 10)
        verbose: Verbosity level for BaseCallback (default: 0)
    """

    def __init__(self, debug: bool = True, log_every_steps: int = 100, log_dir: Optional[str] = None,
                 save_trajectories: bool = True, trajectory_log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)

        self.debug = debug
        self.log_every_steps = log_every_steps
        self.log_dir = log_dir
        self.save_trajectories = save_trajectories
        self.trajectory_log_interval = trajectory_log_interval

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.last_print_timestep = 0

        # Agent-environment interaction tracking
        self.current_trajectory = []  # List of (position, action, reward, step)
        self.all_trajectories = []  # Store all complete trajectories
        self.actions_history = []  # Store action sequences
        self.rewards_history = []  # Store reward sequences

        # TensorBoard writer (initialized in _on_training_start)
        self.tb_writer = None

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.log_dir:
            self.tb_writer = SummaryWriter(self.log_dir)
            if self.debug:
                print(f"  [TensorBoard] Initialized writer with log_dir: {self.log_dir}")

            # Set TensorBoard writer to reliability fusion module
            self._setup_tensorboard_writer_to_fusion_module()
        elif self.debug:
            print("  [TensorBoard] No log_dir provided, skipping TensorBoard logging")

    def _setup_tensorboard_writer_to_fusion_module(self) -> None:
        """Set TensorBoard writer to reliability fusion module for internal logging."""
        try:
            policy = self.model.policy
            # if self.debug:
            #     print(f"  [TensorBoard] Policy attributes: {dir(policy)}")

            if hasattr(policy, 'extractors_dict'):
                extractors_dict = policy.extractors_dict
                if self.debug:
                    print(f"  [TensorBoard] Found extractors_dict with keys: {list(extractors_dict.keys()) if hasattr(extractors_dict, 'keys') else 'N/A'}")

                for key, extractor in extractors_dict.items():
                    if self.debug:
                        print(f"  [TensorBoard] Extractor '{key}' type: {type(extractor)}")

                    if hasattr(extractor, 'reliability_fusion'):
                        extractor.reliability_fusion.tensorboard_writer = self.tb_writer
                        if self.debug:
                            print(f"  [TensorBoard] Set writer to extractor '{key}'.reliability_fusion")
                    elif hasattr(extractor, 'last_fusion_output'):
                        # This might be a different structure
                        if self.debug:
                            print(f"  [TensorBoard] Extractor '{key}' has last_fusion_output but no reliability_fusion")
            elif hasattr(policy, 'features_extractor') and policy.features_extractor is not None:
                extractor = policy.features_extractor
                if self.debug:
                    print(f"  [TensorBoard] features_extractor type: {type(extractor)}")
                    print(f"  [TensorBoard] features_extractor attributes: {[a for a in dir(extractor) if not a.startswith('_')][:20]}")

                if hasattr(extractor, 'reliability_fusion'):
                    extractor.reliability_fusion.tensorboard_writer = self.tb_writer
                    if self.debug:
                        print(f"  [TensorBoard] Set writer to features_extractor.reliability_fusion")
        except Exception as e:
            if self.debug:
                print(f"  [TensorBoard] Failed to set writer to fusion module: {e}")
                import traceback
                traceback.print_exc()

    def _log_reliability_metrics(self) -> None:
        """Log reliability scores and fusion weights from feature extractor."""
        try:
            if self.tb_writer is None:
                return

            if not hasattr(self, 'model') or self.model is None:
                return

            policy = self.model.policy

            # MultiInputPolicy stores extractors differently
            # Check for extractors_dict attribute
            if hasattr(policy, 'extractors_dict'):
                extract_dict = policy.extractors_dict
                for key, extractor in extract_dict.items():
                    if hasattr(extractor, 'last_fusion_output') and extractor.last_fusion_output is not None:
                        self._log_fusion_output(extractor.last_fusion_output)
                        return
            elif hasattr(policy, 'features_extractor'):
                extractor = policy.features_extractor
                if extractor is not None and hasattr(extractor, 'last_fusion_output') and extractor.last_fusion_output is not None:
                    self._log_fusion_output(extractor.last_fusion_output)
                    return

        except Exception as e:
            if self.debug and self.num_timesteps <= 100:
                print(f"  [TensorBoard] Failed to log reliability metrics: {e}")

    def _log_fusion_output(self, fusion_output) -> None:
        """Helper method to log fusion output to TensorBoard."""
        # Log reliability scores
        reliability = fusion_output['reliability']
        self.tb_writer.add_scalar("reliability/lidar", reliability['lidar'].mean().item(), self.num_timesteps)
        self.tb_writer.add_scalar("reliability/rgb", reliability['rgb'].mean().item(), self.num_timesteps)
        self.tb_writer.add_scalar("reliability/imu", reliability['imu'].mean().item(), self.num_timesteps)

        # Log fusion weights
        weights = fusion_output['weights']
        self.tb_writer.add_scalar("fusion_weights/lidar", weights['w_lidar'].mean().item(), self.num_timesteps)
        self.tb_writer.add_scalar("fusion_weights/rgb", weights['w_rgb'].mean().item(), self.num_timesteps)
        self.tb_writer.add_scalar("fusion_weights/imu", weights['w_imu'].mean().item(), self.num_timesteps)

        # Log attention histograms
        if 'attention_scores' in weights:
            self.tb_writer.add_histogram("attention/scores", weights['attention_scores'].flatten(), self.num_timesteps)
        if 'attention_weights' in weights:
            self.tb_writer.add_histogram("attention/weights", weights['attention_weights'].flatten(), self.num_timesteps)

    def _log_sensor_data(self) -> None:
        """Log sensor data samples to TensorBoard."""
        try:
            if self.tb_writer is None:
                return

            # Get current observations from locals
            obs = self.locals.get("obs")
            if obs is None:
                return

            # Log RGB image (first in batch)
            if 'rgb' in obs:
                rgb = obs['rgb'][0]  # (H, W, 3)
                if isinstance(rgb, np.ndarray):
                    rgb_tensor = torch.from_numpy(rgb)
                else:
                    rgb_tensor = rgb
                # Convert to (C, H, W) format and normalize to [0, 1]
                rgb_tensor = rgb_tensor.permute(2, 0, 1).float() / 255.0
                self.tb_writer.add_image("sensors/rgb", rgb_tensor, self.num_timesteps)

            # Log LiDAR point cloud statistics
            if 'lidar' in obs:
                lidar = obs['lidar'][0]  # (N, 3)
                self.tb_writer.add_histogram("sensors/lidar_x", lidar[:, 0], self.num_timesteps)
                self.tb_writer.add_histogram("sensors/lidar_y", lidar[:, 1], self.num_timesteps)
                self.tb_writer.add_histogram("sensors/lidar_z", lidar[:, 2], self.num_timesteps)

            # Log IMU data
            if 'imu' in obs:
                imu = obs['imu'][0]  # (6,)
                for i, val in enumerate(imu):
                    self.tb_writer.add_scalar(f"sensors/imu_{i}", val.item(), self.num_timesteps)

        except Exception as e:
            if self.debug and self.num_timesteps <= 100:
                print(f"  [TensorBoard] Failed to log sensor data: {e}")

    def _on_step(self) -> bool:
        """Called at every step during training."""
        rewards = self.locals.get("rewards")
        if rewards is not None and len(rewards) > 0:
            self.current_episode_reward += rewards[0]

        self.current_episode_steps += 1

        # Track agent-environment interaction
        self._track_interaction()

        # Check if episode ended
        done = self.locals.get("dones")
        truncated = self.locals.get("truncated")
        episode_ended = (done and done[0]) or (truncated and truncated[0])

        if episode_ended:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_steps)

            # Save trajectory data
            if self.save_trajectories and len(self.current_trajectory) > 0:
                self.all_trajectories.append({
                    'episode': self.episode_count,
                    'trajectory': self.current_trajectory.copy(),
                    'total_reward': self.current_episode_reward,
                    'length': self.current_episode_steps
                })

                # Save to file periodically
                if self.episode_count % self.trajectory_log_interval == 0:
                    self._save_trajectories()

            if self.tb_writer and self.debug:
                self.tb_writer.add_scalar("episode/reward", self.current_episode_reward, self.num_timesteps)
                self.tb_writer.add_scalar("episode/length", self.current_episode_steps, self.num_timesteps)
                self._log_reliability_metrics()
                self._log_trajectory_metrics()
                if self.episode_count % 10 == 0:
                    self.tb_writer.flush()

            # Log to SB3 logger
            self.logger.record("episode/reward", self.current_episode_reward)
            self.logger.record("episode/length", self.current_episode_steps)
            self.logger.dump(self.num_timesteps)

            # Print debug info
            if self.debug:
                print(f"  Episode {self.episode_count:3d}: "
                      f"steps={self.current_episode_steps:4d}, "
                      f"reward={self.current_episode_reward:8.2f}, "
                      f"total_timesteps={self.num_timesteps:5d}")

            # Reset episode tracking
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0
            self.current_trajectory = []

        elif self.debug and self.num_timesteps - self.last_print_timestep >= self.log_every_steps:
            print(f"  Progress: timestep={self.num_timesteps:5d}, "
                  f"ep_reward={self.current_episode_reward:8.2f}, "
                  f"ep_steps={self.current_episode_steps:4d}")
            self.last_print_timestep = self.num_timesteps

            if self.tb_writer and self.debug:
                self.tb_writer.add_scalar("progress/episode_reward", self.current_episode_reward, self.num_timesteps)
                self.tb_writer.add_scalar("progress/episode_steps", self.current_episode_steps, self.num_timesteps)
                self._log_sensor_data()
                self._log_reliability_metrics()
                self._log_trajectory_metrics()
                if self.num_timesteps % 500 == 0:
                    self.tb_writer.flush()

        return True

    def _track_interaction(self) -> None:
        """Record agent action and environment state for trajectory visualization."""
        try:
            # Get position from environment info
            info = self.locals.get("infos")
            if info and len(info) > 0 and 'position' in info[0]:
                position = info[0]['position'].tolist()
            else:
                position = [0.0, 0.0, 0.0]

            # Get action
            actions = self.locals.get("actions")
            if actions is not None and len(actions) > 0:
                action = actions[0].cpu().numpy().tolist() if hasattr(actions[0], 'cpu') else actions[0].tolist()
            else:
                action = [0.0, 0.0, 0.0, 0.0]

            # Get reward
            rewards = self.locals.get("rewards")
            reward = float(rewards[0]) if rewards is not None and len(rewards) > 0 else 0.0

            # Store interaction
            interaction_step = {
                'timestep': self.num_timesteps,
                'position': position,
                'action': action,
                'reward': reward
            }
            self.current_trajectory.append(interaction_step)
        except Exception as e:
            if self.debug and self.num_timesteps <= 100:
                print(f"  [Trajectory] Failed to track interaction: {e}")

    def _log_trajectory_metrics(self) -> None:
        """Log trajectory-related metrics to TensorBoard."""
        try:
            if self.tb_writer is None or len(self.current_trajectory) == 0:
                return

            # Get recent positions
            recent_steps = self.current_trajectory[-10:] if len(self.current_trajectory) >= 10 else self.current_trajectory

            for i, step in enumerate(recent_steps):
                pos = step['position']
                step_num = self.num_timesteps - len(recent_steps) + i
                self.tb_writer.add_scalar("trajectory/x", pos[0], step_num)
                self.tb_writer.add_scalar("trajectory/y", pos[1], step_num)
                self.tb_writer.add_scalar("trajectory/z", pos[2], step_num)

                # Log action components
                action = step['action']
                if len(action) >= 4:
                    self.tb_writer.add_scalar("action/vx", action[0], step_num)
                    self.tb_writer.add_scalar("action/vy", action[1], step_num)
                    self.tb_writer.add_scalar("action/vz", action[2], step_num)
                    self.tb_writer.add_scalar("action/omega", action[3], step_num)

        except Exception as e:
            if self.debug and self.num_timesteps <= 100:
                print(f"  [Trajectory] Failed to log metrics: {e}")

    def _save_trajectories(self) -> None:
        """Save collected trajectories to JSON file for post-processing."""
        try:
            if self.log_dir is None or len(self.all_trajectories) == 0:
                return

            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            trajectory_file = log_path / "trajectories.json"

            # Convert to serializable format
            serializable_trajectories = []
            for traj in self.all_trajectories:
                serializable_trajectories.append({
                    'episode': traj['episode'],
                    'total_reward': float(traj['total_reward']),
                    'length': int(traj['length']),
                    'trajectory': traj['trajectory']
                })

            with open(trajectory_file, 'w') as f:
                json.dump(serializable_trajectories, f, indent=2)

            if self.debug:
                print(f"  [Trajectory] Saved {len(self.all_trajectories)} trajectories to {trajectory_file}")
        except Exception as e:
            if self.debug:
                print(f"  [Trajectory] Failed to save trajectories: {e}")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Save final trajectories
        if self.save_trajectories:
            self._save_trajectories()

        if self.tb_writer:
            self.tb_writer.close()
            if self.debug:
                print(f"  [TensorBoard] Closed writer")
        elif self.debug:
            print("  [TensorBoard] No writer to close")

        if self.debug and self.episode_count > 0:
            print(f"\n  Training Summary:")
            print(f"    Total episodes: {self.episode_count}")
            print(f"    Average reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
            print(f"    Average length: {np.mean(self.episode_lengths):.1f} ± {np.std(self.episode_lengths):.1f} steps")
            print(f"    Best reward: {max(self.episode_rewards):.2f}")
            print(f"    Worst reward: {min(self.episode_rewards):.2f}")

            if len(self.all_trajectories) > 0:
                print(f"    Total trajectories saved: {len(self.all_trajectories)}")
