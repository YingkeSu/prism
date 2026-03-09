"""
Main Training Entry Point for Idea1 Project

Usage:
    python train.py [--timesteps N] [--no-reliability] [--save]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from stable_baselines3 import SAC

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import BaselineExtractor, GAPEncoder, MobileNetV3Encoder
from utils.visualization import VisualizationCallback


def warmup_model(model, env, num_steps: int = 10) -> None:
    """
    Warm up GPU cache allocator by running actual model forward passes.

    This preallocates GPU memory and avoids performance degradation from
    gradual memory allocation during training.

    Args:
        model: The RL model to warmup
        env: The gym environment
        num_steps: Number of warmup steps (default 10)
    """
    print(f"Warming up model with {num_steps} steps...")
    obs, _ = env.reset()

    for i in range(num_steps):
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        if (i + 1) % 5 == 0:
            print(f"  Warmup step {i+1}/{num_steps}")

        if terminated or truncated:
            obs, _ = env.reset()

    print("Model warmup complete.")


def train_model(
    total_timesteps: int = 100,
    use_reliability: bool = True,
    use_reliability_modulation: bool = True,
    save_model: bool = False,
    encoder_type: str = 'gap',
    debug: bool = False,
    batch_size: int = 8,
    train_freq: int = 4,
    gradient_steps: int = 1,
    learning_starts: int = 100,
    buffer_size: int = 50_000,
    degradation_level: float = 0.0,
    difficulty: str = "easy",
    max_steps: int = 200,
    share_features_extractor: bool = True,
    imu_history_len: int = 16,
    chunk_size: int = 0,
) -> bool:
    """
    Train model for specified number of timesteps

    Args:
        total_timesteps: Number of timesteps to train
        use_reliability: Whether to use reliability-aware fusion
        save_model: Whether to save the trained model

    Returns:
        True if training succeeded, False otherwise
    """

    print("="*60)
    print("TRAINING IDEA1 MODEL")
    print("="*60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Use reliability: {use_reliability}")
    print(f"Save model: {save_model}")
    print(f"Encoder type: {encoder_type}")
    print(f"Debug mode: {debug}")
    print(f"Use reliability modulation: {use_reliability_modulation}")
    print(f"Degradation level: {degradation_level}")
    print(f"Difficulty: {difficulty}")
    print(f"Max episode steps: {max_steps}")
    print(f"SAC batch_size: {batch_size}")
    print(f"SAC train_freq: {train_freq}")
    print(f"SAC gradient_steps: {gradient_steps}")
    print(f"SAC learning_starts: {learning_starts}")
    print(f"SAC buffer_size: {buffer_size}")
    print(f"Share features extractor: {share_features_extractor}")
    print(f"IMU history length: {imu_history_len}")

    env = UAVMultimodalEnv(
        max_steps=max(1, int(max_steps)),
        degradation_level=degradation_level,
        difficulty=difficulty,
    )

    # Create model
    # Select encoder class based on encoder_type
    encoder_class_map = {
        'baseline': BaselineExtractor,
        'gap': GAPEncoder,
        'mobilenet': MobileNetV3Encoder
    }
    encoder_class = encoder_class_map[encoder_type]

    policy_kwargs = {
        'features_extractor_class': encoder_class,
        'features_extractor_kwargs': {
            'features_dim': 256,
            'use_reliability': use_reliability,
            'use_reliability_modulation': use_reliability_modulation,
            'imu_history_len': max(1, int(imu_history_len)),
            'debug_mode': debug
        },
        'share_features_extractor': share_features_extractor,
    }

    model = SAC(
        'MultiInputPolicy',
        env,
        learning_rate=3e-4,
        verbose=0,
        tensorboard_log='logs/sb3' if debug else None,
        policy_kwargs=policy_kwargs,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=(train_freq, "step"),
        gradient_steps=gradient_steps
    )

    # NOTE: torch.compile() DISABLED - benchmarking showed it slows down training by >100x
    # The .expand() optimization and other improvements provide sufficient speedup

    # RGB encoder parameter mapping (actual parameter counts)
    rgb_encoder_param_map = {
        'baseline': 16.9,  # Original UAVMultimodalExtractor
        'gap': 0.102,  # ~102K
        'mobilenet': 1.075  # ~1.075M (mobilenet.features ~0.927M + projection ~0.148M)
    }

    # Count RGB encoder parameters specifically
    # Note: SB3 creates features_extractor lazily, so we use pre-calculated values
    rgb_encoder_params = rgb_encoder_param_map[encoder_type]
    total_params = sum(p.numel() for p in model.policy.parameters()) / 1e6

    # Display statistics
    print(f"RGB Encoder Type: {encoder_type}")
    print(f"RGB Encoder Parameters: {rgb_encoder_params:.3f}M")
    print(f"Total Model Parameters: {total_params:.1f}M")

    # Enable cuDNN benchmark mode for optimal convolution performance
    if torch.cuda.is_available():
        cudnn.benchmark = True
        print("cuDNN benchmark mode enabled")

    # Speed estimate
    speed_estimates = {
        'baseline': '~10 minutes per 1000 steps',
        'gap': '~20-40 seconds per 1000 rollout steps (without heavy updates)',
        'mobilenet': '~15-25 seconds per 1000 rollout steps (without heavy updates)'
    }
    print(f"Training Speed Estimate: {speed_estimates[encoder_type]}")
    print(f"Starting training...")

    # Warm up model to preallocate GPU memory
    warmup_model(model, env, num_steps=10)

    # Use single-call learn() by default in non-debug mode for lower overhead.
    if chunk_size > 0:
        effective_chunk_size = chunk_size
    else:
        effective_chunk_size = 100 if debug else total_timesteps
    total_chunks = (total_timesteps + effective_chunk_size - 1) // effective_chunk_size
    
    start_time = time.time()

    try:
        viz_callback = (
            VisualizationCallback(
                debug=debug,
                log_every_steps=100,
                log_dir=f"logs/sb3/train_{total_timesteps}ts",
                save_trajectories=debug
            )
            if debug else None
        )

        for chunk_idx in range(total_chunks):
            steps_in_chunk = min(
                effective_chunk_size,
                total_timesteps - chunk_idx * effective_chunk_size
            )

            print(f"\n[{chunk_idx+1}/{total_chunks}] Training {steps_in_chunk} steps...")

            model.learn(
                total_timesteps=steps_in_chunk,
                log_interval=1 if debug else None,
                tb_log_name=f"train_{total_timesteps}ts",
                callback=viz_callback,
                reset_num_timesteps=(chunk_idx == 0)  # Only reset timesteps for first chunk
            )

        elapsed_time = time.time() - start_time

        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        print(f"Time per timestep: {elapsed_time / total_timesteps * 1000:.2f} ms")

        # Save model if requested
        if save_model:
            save_dir = Path("models")
            save_dir.mkdir(exist_ok=True)

            model_name = f"idea1_model_{total_timesteps}ts.zip"
            model_path = save_dir / model_name

            model.save(str(model_path))
            print(f"Model saved to: {model_path}")

        return True

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model(
    model_path: str,
    num_episodes: int = 5,
    *,
    max_steps: int = 200,
    degradation_level: float = 0.0,
    difficulty: str = "easy",
):
    """
    Test trained model

    Args:
        model_path: Path to saved model
        num_episodes: Number of test episodes
    """

    print("\n" + "="*60)
    print("TESTING TRAINED MODEL")
    print("="*60)

    env = UAVMultimodalEnv(
        max_steps=max(1, int(max_steps)),
        degradation_level=float(max(0.0, min(1.0, degradation_level))),
        difficulty=difficulty,
    )

    model = SAC.load(model_path, env=env)

    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            if truncated:
                break

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)

        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={episode_steps}")

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print(f"\nTest Results ({num_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Steps: {avg_steps:.2f}")
    print(f"  Total Reward: {sum(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train Idea1 Model")

    parser.add_argument(
        '--timesteps',
        type=int,
        default=100,
        help='Number of timesteps to train (default: 100)'
    )

    parser.add_argument(
        '--no-reliability',
        action='store_true',
        help='Disable reliability-aware fusion (use baseline)'
    )

    parser.add_argument(
        '--no-reliability-modulation',
        action='store_true',
        help='Disable reliability-conditioned adaptive normalization modulation'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save trained model'
    )

    parser.add_argument(
        '--test',
        type=str,
        help='Test a saved model at given path'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of test episodes (default: 5)'
    )

    parser.add_argument(
        '--encoder-type',
        type=str,
        default='gap',
        choices=['baseline', 'gap', 'mobilenet'],
        help="RGB encoder type: baseline (16.9M), gap (~102K), mobilenet (~1.08M) (default: gap)"
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output and visualization callback (default: disabled)'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Deprecated flag. Debug mode is already disabled by default.'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='SAC batch_size (default: 8, faster on CPU/MPS for multimodal inputs)'
    )

    parser.add_argument(
        '--train-freq',
        type=int,
        default=4,
        help='SAC train_freq in steps (default: 4, reduces update overhead)'
    )

    parser.add_argument(
        '--gradient-steps',
        type=int,
        default=1,
        help='SAC gradient_steps per update (default: 1)'
    )

    parser.add_argument(
        '--learning-starts',
        type=int,
        default=100,
        help='SAC learning_starts (default: 100)'
    )

    parser.add_argument(
        '--buffer-size',
        type=int,
        default=50000,
        help='SAC replay buffer size (default: 50000)'
    )

    parser.add_argument(
        '--degradation-level',
        type=float,
        default=0.0,
        help='Sensor degradation level in [0, 1] for robustness experiments'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='Environment max steps per episode (default: 200)'
    )

    parser.add_argument(
        '--difficulty',
        type=str,
        default='easy',
        choices=['easy', 'medium', 'hard'],
        help='Environment difficulty preset (default: easy)'
    )

    parser.add_argument(
        '--imu-history-len',
        type=int,
        default=16,
        help='Pseudo IMU sequence length used by reliability branch (default: 16)'
    )

    parser.add_argument(
        '--no-share-features-extractor',
        action='store_true',
        help='Use separate actor/critic feature extractors (slower, but matches strict SAC default)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        help='Training chunk size. 0 means auto (single-call for non-debug, 100 for debug).'
    )

    args = parser.parse_args()

    if args.test:
        test_model(
            args.test,
            args.episodes,
            max_steps=max(1, int(args.max_steps)),
            degradation_level=float(max(0.0, min(1.0, args.degradation_level))),
            difficulty=args.difficulty,
        )
    else:
        success = train_model(
            total_timesteps=args.timesteps,
            use_reliability=not args.no_reliability,
            use_reliability_modulation=not args.no_reliability_modulation,
            save_model=args.save,
            encoder_type=args.encoder_type,
            debug=bool(args.debug and not args.no_debug),
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts,
            buffer_size=args.buffer_size,
            degradation_level=float(max(0.0, min(1.0, args.degradation_level))),
            difficulty=args.difficulty,
            max_steps=max(1, int(args.max_steps)),
            share_features_extractor=not args.no_share_features_extractor,
            imu_history_len=max(1, int(args.imu_history_len)),
            chunk_size=max(0, int(args.chunk_size)),
        )

        if success:
            print("\n" + "="*60)
            print("SUCCESS!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("FAILED!")
            print("="*60)
            exit(1)


if __name__ == "__main__":
    main()
