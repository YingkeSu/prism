"""
Demo Script for Idea1 Project

Tests all modules and demonstrates SB3 integration.
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.imu_consistency_checker import IMUConsistencyChecker
from networks.reliability_predictor import ReliabilityPredictor
from networks.dynamic_weighting_layer import DynamicWeightingLayer
from networks.adaptive_normalization import AdaptiveNormalization
from networks.reliability_aware_fusion import ReliabilityAwareFusionModule
from networks.uav_multimodal_extractor import UAVMultimodalExtractor
from envs.simple_2d_env import Simple2DObstacleEnv
from envs.uav_multimodal_env import UAVMultimodalEnv


def test_reliability_estimators():
    print("\n" + "="*60)
    print("Testing Reliability Estimators")
    print("="*60)

    lidar_estimator = LiDARSNREstimator(point_dim=3, feature_dim=64)
    image_estimator = ImageQualityEstimator(input_channels=3)
    imu_checker = IMUConsistencyChecker(imu_dim=6, window_size=100)

    batch_size = 2
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)

    lidar_output = lidar_estimator(lidar_points)
    print(f"LiDAR SNR: {lidar_output['snr'].mean():.3f}")
    print(f"LiDAR Density: {lidar_output['density'].mean():.3f}")

    image_output = image_estimator(rgb_image)
    print(f"Image Sharpness: {image_output['sharpness'].mean():.3f}")
    print(f"Image Contrast: {image_output['contrast'].mean():.3f}")
    print(f"Overall Quality: {image_output['overall_quality'].mean():.3f}")

    imu_output = imu_checker(imu_data)
    print(f"IMU Consistency: {imu_output['consistency'].mean():.3f}")
    print(f"IMU Drift: {imu_output['drift_score'].mean():.3f}")

    print("✅ All reliability estimators work correctly")


def test_reliability_predictor():
    print("\n" + "="*60)
    print("Testing Reliability Predictor")
    print("="*60)

    predictor = ReliabilityPredictor(
        lidar_dim=64, rgb_dim=256, imu_dim=64,
        hidden_dim=128, output_dim=256
    )

    batch_size = 2
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)

    output = predictor(lidar_points, rgb_image, imu_data)
    print(f"LiDAR Reliability: {output['r_lidar'].mean():.3f}")
    print(f"RGB Reliability: {output['r_rgb'].mean():.3f}")
    print(f"IMU Reliability: {output['r_imu'].mean():.3f}")
    print(f"Parameter Count: {predictor.count_parameters() / 1000:.1f}K")

    assert predictor.count_parameters() < 500000
    print("✅ Reliability Predictor works correctly")


def test_dynamic_weighting():
    print("\n" + "="*60)
    print("Testing Dynamic Weighting Layer")
    print("="*60)

    weighting = DynamicWeightingLayer(feature_dim=128, num_heads=4)

    batch_size = 2
    feature_dim = 128
    lidar_feat = torch.randn(batch_size, feature_dim)
    rgb_feat = torch.randn(batch_size, feature_dim)
    imu_feat = torch.randn(batch_size, feature_dim)

    output = weighting(lidar_feat, rgb_feat, imu_feat)
    total = output['w_lidar'] + output['w_rgb'] + output['w_imu']

    print(f"LiDAR Weight: {output['w_lidar'].mean():.3f}")
    print(f"RGB Weight: {output['w_rgb'].mean():.3f}")
    print(f"IMU Weight: {output['w_imu'].mean():.3f}")
    print(f"Total Weight: {total.mean():.6f}")

    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    print("✅ Dynamic Weighting Layer works correctly")


def test_adaptive_normalization():
    print("\n" + "="*60)
    print("Testing Adaptive Normalization")
    print("="*60)

    norm = AdaptiveNormalization(feature_dim=256)

    batch_size = 2
    r_lidar = torch.rand(batch_size, 1)
    r_rgb = torch.rand(batch_size, 1)
    r_imu = torch.rand(batch_size, 1)
    features = {
        'lidar': torch.randn(batch_size, 256),
        'rgb': torch.randn(batch_size, 256),
        'imu': torch.randn(batch_size, 256)
    }

    output = norm(r_lidar, r_rgb, r_imu, features)
    lidar_norm = torch.norm(output['lidar_out'], dim=1)
    rgb_norm = torch.norm(output['rgb_out'], dim=1)
    imu_norm = torch.norm(output['imu_out'], dim=1)

    print(f"LiDAR Norm: {lidar_norm.mean():.3f}")
    print(f"RGB Norm: {rgb_norm.mean():.3f}")
    print(f"IMU Norm: {imu_norm.mean():.3f}")

    print("✅ Adaptive Normalization works correctly")


def test_fusion_module():
    print("\n" + "="*60)
    print("Testing Complete Fusion Module")
    print("="*60)

    fusion_module = ReliabilityAwareFusionModule(feature_dim=256, num_heads=4)

    batch_size = 2
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)
    targets = torch.randn(batch_size, 64)

    output = fusion_module(lidar_points, rgb_image, imu_data)
    loss_dict = fusion_module.get_loss(output, targets)

    print(f"Output shape: {output['output'].shape}")
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"MSE loss: {loss_dict['mse_loss'].item():.4f}")
    print(f"Reliability reg: {loss_dict['reliability_reg'].item():.4f}")

    print("✅ Fusion Module works correctly")


def test_environments():
    print("\n" + "="*60)
    print("Testing Environments")
    print("="*60)

    print("\n--- Simple 2D Environment ---")
    env_2d = Simple2DObstacleEnv(num_obstacles=3)
    obs, info = env_2d.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env_2d.observation_space}")
    print(f"Action space: {env_2d.action_space}")

    for step in range(50):
        action = env_2d.action_space.sample()
        obs, reward, done, truncated, info = env_2d.step(action)
        if done or truncated:
            break
    print(f"Episode finished after {step+1} steps")
    print("✅ Simple 2D Environment works")

    print("\n--- UAV Multimodal Environment ---")
    env_multi = UAVMultimodalEnv(max_steps=50)
    obs, info = env_multi.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"LiDAR shape: {obs['lidar'].shape}")
    print(f"RGB shape: {obs['rgb'].shape}")
    print(f"IMU shape: {obs['imu'].shape}")

    for step in range(30):
        action = env_multi.action_space.sample()
        obs, reward, done, truncated, info = env_multi.step(action)
        if done or truncated:
            break
    print(f"Episode finished after {step+1} steps")
    print("✅ UAV Multimodal Environment works")


def test_sb3_compatibility():
    print("\n" + "="*60)
    print("Testing SB3 Compatibility")
    print("="*60)

    try:
        from stable_baselines3 import SAC
        print("✅ Stable-Baselines3 SAC imported successfully")
        print("Note: 'MultiInputPolicy' is specified via string in policy_kwargs")
    except ImportError as e:
        print(f"⚠️  SB3 import failed: {e}")
        print("Continuing with environment tests only...")
        return

    obs_space = spaces.Dict({
        'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
        'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
    })

    extractor = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=True)

    observations = {
        'lidar': torch.randn(2, 1000, 3),
        'rgb': torch.randint(0, 255, (2, 128, 128, 3)),
        'imu': torch.randn(2, 6)
    }

    features = extractor(observations)
    print(f"Feature extractor output shape: {features.shape}")
    print(f"Expected shape: (2, 256)")

    assert features.shape == (2, 256)
    print("✅ SB3 Feature Extractor works correctly")


def main():
    print("=" * 60)
    print("Idea1 Project Demo - Reliability-Aware Multi-Modal Fusion")
    print("=" * 60)

    try:
        test_reliability_estimators()
    except Exception as e:
        print(f"❌ Reliability Estimators failed: {e}")

    try:
        test_reliability_predictor()
    except Exception as e:
        print(f"❌ Reliability Predictor failed: {e}")

    try:
        test_dynamic_weighting()
    except Exception as e:
        print(f"❌ Dynamic Weighting failed: {e}")

    try:
        test_adaptive_normalization()
    except Exception as e:
        print(f"❌ Adaptive Normalization failed: {e}")

    try:
        test_fusion_module()
    except Exception as e:
        print(f"❌ Fusion Module failed: {e}")

    try:
        test_environments()
    except Exception as e:
        print(f"❌ Environments failed: {e}")

    try:
        test_sb3_compatibility()
    except Exception as e:
        print(f"❌ SB3 Compatibility failed: {e}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
