import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.uav_multimodal_extractor import UAVMultimodalExtractor

print("="*60)
print("Testing Basic Training Setup")
print("="*60)

try:
    env = UAVMultimodalEnv(max_steps=10)
    print(f"Environment created: {type(env)}")

    obs, info = env.reset()
    print(f"Reset successful")
    print(f"Observation keys: {obs.keys()}")
    print(f"  LiDAR shape: {obs['lidar'].shape}")
    print(f"  RGB shape: {obs['rgb'].shape}")
    print(f"  IMU shape: {obs['imu'].shape}")

    # Test one step
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step 1 successful")
    print(f"  Reward: {reward:.4f}")
    print(f"  Done: {done}")

    print("\n✅ Basic environment test passed!")

    # Test feature extractor
    print("\n" + "="*60)
    print("Testing Feature Extractor")
    print("="*60)

    obs_space = env.observation_space
    extractor = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=True)

    batch_obs = {
        'lidar': torch.randn(4, 1000, 3),
        'rgb': torch.rand(4, 128, 128, 3) * 255,
        'imu': torch.randn(4, 6)
    }

    features = extractor(batch_obs)
    print(f"Features shape: {features.shape}")

    print("\n✅ Feature extractor test passed!")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
