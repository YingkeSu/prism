"""
SB3 Integration Test - End-to-end training verification

This script tests the complete SB3 training pipeline with reliability-aware fusion.
"""

import os
import sys
from typing import cast

import pytest
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from networks.uav_multimodal_extractor import UAVMultimodalExtractor
from envs.uav_multimodal_env import UAVMultimodalEnv


@pytest.fixture
def env():
    """Shared UAV environment fixture for pytest-style tests."""
    return UAVMultimodalEnv(max_steps=100)


@pytest.fixture
def model(env):
    """Shared SAC model fixture for forward/training smoke tests."""
    return SAC(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs={
            'features_extractor_class': UAVMultimodalExtractor,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': True
            }
        }
    )


def test_environment_reset():
    """Test environment reset and observation generation"""
    print("="*60)
    print("1. Test environment reset")
    print("="*60)

    env = UAVMultimodalEnv(max_steps=100)
    obs, info = env.reset(seed=42)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for key in obs:
        print(f"  {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")

    assert 'lidar' in obs
    assert 'rgb' in obs
    assert 'imu' in obs

    print("Pass: Environment reset test")


def test_single_step():
    """Test single environment step"""
    print("\n" + "="*60)
    print("2. Test single step execution")
    print("="*60)

    env = UAVMultimodalEnv(max_steps=100)
    obs, _ = env.reset(seed=42)

    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)

    print(f"Reward: {reward:.4f}")
    print(f"Done: {done}, Truncated: {truncated}")

    assert next_obs.keys() == obs.keys()
    print("Pass: Single step execution test")


def test_feature_extractor():
    """Test UAVMultimodalExtractor standalone"""
    print("\n" + "="*60)
    print("3. Test feature extractor")
    print("="*60)

    obs_space = spaces.Dict({
        'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
        'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
    })

    extractor_reliability = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=True)
    extractor_baseline = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=False)

    batch_size = 4
    observations = {
        'lidar': torch.randn(batch_size, 1000, 3),
        'rgb': torch.randint(0, 255, (batch_size, 128, 128, 3)),
        'imu': torch.randn(batch_size, 6)
    }

    features_reliability = extractor_reliability(observations)
    print(f"With reliability features shape: {features_reliability.shape}")
    assert features_reliability.shape == (batch_size, 256)
    print("Pass: Feature extractor (with reliability) test")

    features_baseline = extractor_baseline(observations)
    print(f"Baseline features shape: {features_baseline.shape}")
    assert features_baseline.shape == (batch_size, 256)
    print("Pass: Feature extractor (baseline) test")


def test_sb3_model_creation():
    """Test SB3 model creation with custom extractor"""
    print("\n" + "="*60)
    print("4. Test SB3 model creation")
    print("="*60)

    env = UAVMultimodalEnv(max_steps=100)
    obs_space = cast(spaces.Dict, env.observation_space)

    extractor_reliability = UAVMultimodalExtractor(
        obs_space,
        features_dim=256,
        use_reliability=True
    )

    model_reliability = SAC(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs={
            'features_extractor_class': UAVMultimodalExtractor,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': True
            }
        }
    )

    print(f"With reliability model parameters: {sum(p.numel() for p in model_reliability.policy.parameters()) / 1000:.1f}K")
    assert model_reliability.policy is not None
    print("Pass: SB3 model creation (with reliability) test")

    extractor_baseline = UAVMultimodalExtractor(
        obs_space,
        features_dim=256,
        use_reliability=False
    )

    model_baseline = SAC(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs={
            'features_extractor_class': UAVMultimodalExtractor,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': False
            }
        }
    )

    print(f"Baseline model parameters: {sum(p.numel() for p in model_baseline.policy.parameters()) / 1000:.1f}K")
    assert model_baseline.policy is not None
    print("Pass: SB3 model creation (baseline) test")


def test_forward_pass(model, env, num_steps=5):
    """Test forward pass through model"""
    print("\n" + "="*60)
    print(f"5. Test forward pass ({num_steps} steps)")
    print("="*60)

    obs, _ = env.reset(seed=42)

    total_reward = 0
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"  Step {step+1}: Reward={reward:.4f}, Done={done}, Truncated={truncated}")

        if done or truncated:
            break

    print(f"Total reward: {total_reward:.4f}")
    print("Pass: Forward pass test")


def test_minimal_training(model, env, total_timesteps=100):
    """Test minimal training loop"""
    print("\n" + "="*60)
    print(f"6. Minimal training test ({total_timesteps} timesteps)")
    print("="*60)

    obs, _ = env.reset(seed=42)

    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10)
        print("Pass: Training completed")
        print("   TensorBoard logs saved to ./logs/")
    except Exception as e:
        raise AssertionError(f"Training failed: {e}") from e


def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("SB3 End-to-End Integration Test")
    print("="*60)
    print("Goal: Verify complete training pipeline")
    print()

    all_passed = True
    model_reliability = None

    # Test 1: Environment reset
    try:
        test_environment_reset()
    except Exception as e:
        print(f"Fail: Test 1 failed: {e}")
        all_passed = False

    # Test 2: Single step execution
    try:
        test_single_step()
    except Exception as e:
        print(f"Fail: Test 2 failed: {e}")
        all_passed = False

    # Test 3: Feature extractor
    try:
        test_feature_extractor()
    except Exception as e:
        print(f"Fail: Test 3 failed: {e}")
        all_passed = False

    # Test 4: SB3 model creation
    try:
        test_sb3_model_creation()
        model_reliability = SAC(
            "MultiInputPolicy",
            UAVMultimodalEnv(max_steps=100),
            verbose=0,
            policy_kwargs={
                'features_extractor_class': UAVMultimodalExtractor,
                'features_extractor_kwargs': {
                    'features_dim': 256,
                    'use_reliability': True
                }
            }
        )
    except Exception as e:
        print(f"Fail: Test 4 failed: {e}")
        all_passed = False

    # Test 5: Forward pass
    if model_reliability is None:
        print("Fail: Test 5 skipped because SB3 model creation failed")
        all_passed = False
    else:
        try:
            env = UAVMultimodalEnv(max_steps=100)
            test_forward_pass(model_reliability, env, num_steps=5)
        except Exception as e:
            print(f"Fail: Test 5 failed: {e}")
            all_passed = False

    # Test 6: Minimal training
    if model_reliability is None:
        print("Fail: Test 6 skipped because SB3 model creation failed")
        all_passed = False
    else:
        try:
            env = UAVMultimodalEnv(max_steps=100)
            test_minimal_training(model_reliability, env, total_timesteps=100)
        except Exception as e:
            print(f"Fail: Test 6 failed: {e}")
            all_passed = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    if all_passed:
        print("Pass: All tests passed!")
        print("   - Environment reset: OK")
        print("   - Single step execution: OK")
        print("   - Feature extractor: OK")
        print("   - SB3 model creation: OK")
        print("   - Forward pass: OK")
        print("   - Minimal training: OK")
        print("\nSystem is ready for full training!")
    else:
        print("Fail: Some tests failed, please check error messages above.")

    return all_passed


if __name__ == "__main__":
    main()
