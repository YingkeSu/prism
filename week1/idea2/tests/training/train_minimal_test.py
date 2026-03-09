"""
Minimal Training Test - Just to verify system works
"""

import os
import sys
import time
from typing import Dict, cast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.uav_multimodal_extractor import UAVMultimodalExtractor

print("="*60)
print("MINIMAL TRAINING TEST")
print("="*60)

try:
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Simple single environment
    base_env = UAVMultimodalEnv(max_steps=10)

    obs_space = base_env.observation_space
    extractor = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=False)

    # Wrap with DummyVecEnv for SB3 compatibility (batched observations)
    env = DummyVecEnv([lambda: base_env])

    # Create batched observation (SB3 requires batch dimension)
    observation = cast(Dict[str, np.ndarray], env.reset())
    print(f"LiDAR shape: {observation['lidar'].shape}")
    print(f"RGB shape: {observation['rgb'].shape}")
    print(f"IMU shape: {observation['imu'].shape}")

    print("Testing feature extraction...")
    # Convert to tensors
    observation_tensors = {
        'lidar': torch.from_numpy(observation['lidar']),
        'rgb': torch.from_numpy(observation['rgb']),
        'imu': torch.from_numpy(observation['imu'])
    }
    features = extractor(observation_tensors)
    print(f"Features shape: {features.shape}")

    print("\n✅ MINIMAL TEST PASSED!")
    print(f"System is working correctly")

    # Quick training test (just 10 steps)
    print("\n" + "="*60)
    print("QUICK TRAINING TEST")
    print("="*60)

    model = SAC(
        'MultiInputPolicy',
        env,
        learning_rate=3e-4,
        verbose=2,
        policy_kwargs={
            'features_extractor_class': UAVMultimodalExtractor,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': False
            }
        }
    )

    print(f"Model created: {sum(p.numel() for p in model.policy.parameters()) / 1e6:.1f}M parameters")
    print("Training for 10 steps...")

    start_time = time.time()

    model.learn(total_timesteps=10, log_interval=5)

    elapsed_time = time.time() - start_time

    print(f"\n✅ Quick training completed!")
    print(f"Total time: {elapsed_time:.2f} seconds")

    print("\n" + "="*60)
    print("SUCCESS! System is working!")
    print("="*60)

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
