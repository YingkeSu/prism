"""
Video recording utilities for agent behavior visualization.

This module provides wrappers and utilities for recording agent episodes
as videos for visualization and analysis.
"""

from pathlib import Path
from typing import Optional, Callable
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import BaseCallback


class VideoRecordCallback(BaseCallback):
    """
    Callback for recording agent episodes as videos during evaluation.

    This callback creates video recordings of agent behavior at specified intervals,
    useful for visualizing policy improvements during training.

    Args:
        eval_env: Environment to use for video recording (should have render_mode='rgb_array')
        record_freq: Create a video every N episodes (default: 10)
        n_episodes: Number of episodes to record per video (default: 1)
        video_folder: Directory to save videos (default: "videos")
        name_prefix: Prefix for video filenames (default: "agent")
        verbose: Verbosity level (default: 0)
    """

    def __init__(
        self,
        eval_env: gym.Env,
        record_freq: int = 10,
        n_episodes: int = 1,
        video_folder: str = "videos",
        name_prefix: str = "agent",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.n_episodes = n_episodes
        self.video_folder = video_folder
        self.name_prefix = name_prefix
        self.recording_env = None
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """Set up video recording environment."""
        # Wrap environment with RecordVideo
        self.recording_env = RecordVideo(
            self.eval_env,
            video_folder=self.video_folder,
            name_prefix=self.name_prefix,
            episode_trigger=lambda x: x % self.record_freq == 0,
            disable_logger=True
        )

    def _on_step(self) -> bool:
        """Record episodes periodically."""
        # Check if we should record this episode
        dones = self.locals.get("dones")
        truncated = self.locals.get("truncated")

        if (dones and dones[0]) or (truncated and truncated[0]):
            self.episode_count += 1

            # Record video every record_freq episodes
            if self.episode_count % self.record_freq == 0:
                if self.verbose > 0:
                    print(f"  [Video] Recording episode {self.episode_count}")

                self._record_episode()

        return True

    def _record_episode(self) -> None:
        """Record one complete episode."""
        obs, _ = self.recording_env.reset()
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.recording_env.step(action)
            step_count += 1

        if self.verbose > 0:
            print(f"  [Video] Episode {self.episode_count} recorded ({step_count} steps)")


def create_recording_env(
    base_env: gym.Env,
    video_folder: str = "videos",
    episode_trigger: Optional[Callable[[int], bool]] = None
) -> gym.Env:
    """
    Create an environment wrapper for video recording.

    Args:
        base_env: Base gym environment (must support rgb_array rendering)
        video_folder: Directory to save videos
        episode_trigger: Function that determines which episodes to record.
                        If None, records every episode.

    Returns:
        Wrapped environment that records videos

    Example:
        >>> env = UAVMultimodalEnv(max_steps=200)
        >>> env = create_recording_env(env, video_folder="eval_videos")
        >>> # Now run episodes - videos will be saved automatically
    """
    if episode_trigger is None:
        episode_trigger = lambda x: True  # Record all episodes

    return RecordVideo(
        base_env,
        video_folder=video_folder,
        name_prefix="evaluation",
        episode_trigger=episode_trigger,
        disable_logger=True
    )


def record_evaluation_video(
    model,
    env: gym.Env,
    video_path: str,
    n_episodes: int = 1,
    deterministic: bool = True
) -> None:
    """
    Record a video of model performance for evaluation purposes.

    This function creates a video of the agent's behavior, useful for
    qualitative assessment of policy performance.

    Args:
        model: Trained RL model (e.g., SAC, PPO from Stable-Baselines3)
        env: Gym environment (must support rgb_array rendering)
        video_path: Path where video will be saved
        n_episodes: Number of episodes to record (default: 1)
        deterministic: Whether to use deterministic actions (default: True)

    Example:
        >>> model = SAC.load("models/trained_model.zip")
        >>> env = UAVMultimodalEnv(max_steps=200)
        >>> record_evaluation_video(model, env, "evaluation.mp4", n_episodes=3)
    """
    # Create temporary recording wrapper
    video_folder = str(Path(video_path).parent)
    video_name = Path(video_path).stem

    # If env doesn't have rgb_array render mode, create new one
    if not hasattr(env, 'render_mode') or env.render_mode != 'rgb_array':
        # Try to get env class and parameters
        if hasattr(env, 'unwrapped'):
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'max_steps'):
                # Create new environment with rgb_array mode
                from envs.uav_multimodal_env import UAVMultimodalEnv
                env = UAVMultimodalEnv(max_steps=unwrapped.max_steps)

    recording_env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=video_name,
        episode_trigger=lambda x: x < n_episodes,  # Record first n_episodes
        disable_logger=True
    )

    # Run episodes
    for episode in range(n_episodes):
        obs, _ = recording_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = recording_env.step(action)
            episode_reward += reward
            step_count += 1

        print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, steps={step_count}")

    recording_env.close()
    print(f"  Video saved to: {video_path}")
