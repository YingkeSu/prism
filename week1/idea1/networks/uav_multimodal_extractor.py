"""
UAV Multimodal Feature Extractor

SB3-compatible custom feature extractor that integrates reliability-aware fusion.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from networks.reliability_aware_fusion import ReliabilityAwareFusionModule


class UAVMultimodalExtractor(BaseFeaturesExtractor):
    """
    Custom Multi-Modal Feature Extractor for UAV Navigation

    Compatible with Stable-Baselines3 BaseFeaturesExtractor.
    Integrates reliability-aware fusion for multi-modal inputs.

    Args:
        observation_space: Dict observation space from gym.Env
        features_dim: Output feature dimension (default 256)
        use_reliability: Whether to use reliability-aware fusion (default True)
        num_heads: Number of attention heads (default 8)

    Input:
        observations: Dict with keys 'lidar', 'rgb', 'imu'

    Output:
        features: (B, features_dim) Fused feature vector
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        use_reliability: bool = True,
        num_heads: int = 8
    ):
        super().__init__(observation_space, features_dim)

        self.use_reliability = use_reliability
        self.num_heads = num_heads

        # Extract modality dimensions from observation space
        lidar_shape = observation_space['lidar'].shape  # (N, 3)
        rgb_shape = observation_space['rgb'].shape  # (H, W, 3)
        imu_shape = observation_space['imu'].shape  # (6,)

        # Base encoders (encode raw data to features)
        self.lidar_base_encoder = nn.Sequential(
            nn.Linear(lidar_shape[1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        self.rgb_base_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * (rgb_shape[0]//4) * (rgb_shape[1]//4), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

        self.imu_base_encoder = nn.Sequential(
            nn.Linear(imu_shape[0], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        # Reliability-aware fusion module
        if use_reliability:
            self.reliability_fusion = ReliabilityAwareFusionModule(
                feature_dim=features_dim,
                num_heads=num_heads
            )

        # Output projection
        if not use_reliability:
            # Simple concatenation when not using reliability
            self.output_projection = nn.Sequential(
                nn.Linear(64 + 256 + 6, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, features_dim)
            )
        else:
            self.output_projection = nn.Linear(64, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            observations: Dict with keys 'lidar', 'rgb', 'imu'

        Returns:
            features: (B, features_dim) Fused feature vector
        """
        # Get modality data
        lidar_points = observations['lidar']  # (B, N, 3)
        rgb_image = observations['rgb']  # (B, H, W, 3)
        imu_data = observations['imu']  # (B, 6)

        # Convert RGB format: (B, H, W, 3) -> (B, 3, H, W)
        rgb_image = rgb_image.permute(0, 3, 1, 2)

        # Expand IMU to sequence (simulate history)
        imu_sequence = imu_data.unsqueeze(1).expand(-1, 100, -1)

        # Base encoding
        lidar_feat = self.lidar_base_encoder(lidar_points.mean(dim=1))  # (B, 64)
        rgb_feat = self.rgb_base_encoder(rgb_image)  # (B, 256)
        imu_feat = self.imu_base_encoder(imu_data)  # (B, 6)

        # Use reliability-aware fusion or simple concatenation
        if self.use_reliability:
            # Use reliability-aware fusion module
            fusion_output = self.reliability_fusion(
                lidar_points,
                rgb_image,
                imu_sequence
            )

            # Project to output dimension
            features = self.output_projection(fusion_output['output'])
        else:
            # Simple concatenation (baseline)
            combined = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
            features = self.output_projection(combined)

        return features


def test_uav_multimodal_extractor():
    """Test UAV Multimodal Extractor"""
    # Create test observation space
    obs_space = spaces.Dict({
        'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
        'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
    })

    # Test with reliability-aware fusion
    extractor = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=True)

    # Create test observations
    batch_size = 4
    observations = {
        'lidar': torch.randn(batch_size, 1000, 3),
        'rgb': torch.randint(0, 255, (batch_size, 128, 128, 3)),
        'imu': torch.randn(batch_size, 6)
    }

    # Forward pass
    features = extractor(observations)

    print(f"Output features shape: {features.shape}")
    print(f"Expected shape: ({batch_size}, 256)")

    assert features.shape == (batch_size, 256), "Output dimension incorrect"
    print("✅ UAV Multimodal Extractor (with reliability) test passed")

    # Test without reliability-aware fusion
    extractor_baseline = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=False)
    features_baseline = extractor_baseline(observations)

    print(f"Baseline output shape: {features_baseline.shape}")
    assert features_baseline.shape == (batch_size, 256), "Baseline output dimension incorrect"
    print("✅ UAV Multimodal Extractor (baseline) test passed")


if __name__ == "__main__":
    import numpy as np
    test_uav_multimodal_extractor()
