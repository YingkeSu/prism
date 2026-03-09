"""
Training Script for Idea1 Project with 10 Episodes

This script runs training for 10 episodes with multi-modal inputs.
"""

import os
import time

import numpy as np
from stable_baselines3 import SAC

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.uav_multimodal_extractor import UAVMultimodalExtractor

def train_with_epoch_10():
    """Train model for 10 episodes"""

    print("="*60)
    print("Training Idea1 Model with 10 Episodes")
    print("="*60)

    # Create single environment
    env = UAVMultimodalEnv(max_steps=100)

    # Device selection
    device = "cuda" if os.getenv("USE_CUDA") == "1" else "auto"

    # Create model with MultiInputPolicy (required for dict observations) and TensorBoard
    model = SAC(
        "MultiInputPolicy",
        env,
        device=device,
        tensorboard_log="logs/sb3",
        verbose=2,
        policy_kwargs={
            'features_extractor_class': UAVMultimodalExtractor,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': True
            }
        }
    )

    print(f"Model created with {sum(p.numel() for p in model.policy.parameters()) / 1e6:.1f}M parameters")
    print(f"Training for 10 episodes...")
    print(f"Approximate timesteps: {10 * 100} (10 episodes x 100 steps)")
    print(f"Requested device: {device}")
    print(f"Resolved device: {model.policy.device}")

    # Start training
    start_time = time.time()

    try:
        # Train for 10 episodes, each with 100 steps = 1000 total timesteps
        model.learn(total_timesteps=1000, log_interval=100, tb_log_name="train_epoch_10")

        elapsed_time = time.time() - start_time

        print(f"\n✅ Training completed successfully!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per episode: {elapsed_time / 10:.2f} seconds")

        # Save model
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "idea1_model.zip")
        model.save(model_path)

        print(f"Model saved to: {model_path}")

        # Test trained model
        print("\n" + "="*60)
        print("Testing Trained Model")
        print("="*60)

        test_env = UAVMultimodalEnv(max_steps=50)

        # Load trained model for testing
        test_model = SAC.load(model_path, env=test_env)

        num_episodes = 10
        total_rewards = []
        total_steps = []

        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                action, _states = test_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                episode_reward += reward
                episode_steps += 1

            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)

            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={episode_steps}")

        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)

        print(f"\n📊 Test Results (over {num_episodes} episodes):")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.2f}")
        print(f"  Total Reward: {sum(total_rewards):.2f}")

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = train_with_epoch_10()

    if success:
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Training failed!")
        print("="*60)
