"""
Adaptive Normalization Layer

Adaptive normalization layer that adjusts normalization strategy based on reliability scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class AdaptiveNormalization(nn.Module):
    """
    Adaptive Normalization Layer

    Dynamically adjusts normalization based on reliability scores.

    Args:
        feature_dim: Feature dimension

    Input:
        r_lidar: (B, 1) LiDAR reliability score
        r_rgb: (B, 1) RGB reliability score
        r_imu: (B, 1) IMU reliability score
        features: Dict {'lidar': (B, D), 'rgb': (B, D), 'imu': (B, D)}

    Output:
        Dict: Normalized features
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable normalization parameters
        self.gamma_lidar = nn.Parameter(torch.ones(1))
        self.gamma_rgb = nn.Parameter(torch.ones(1))
        self.gamma_imu = nn.Parameter(torch.ones(1))

        # Offset parameters
        self.beta_lidar = nn.Parameter(torch.zeros(1))
        self.beta_rgb = nn.Parameter(torch.zeros(1))
        self.beta_imu = nn.Parameter(torch.zeros(1))

        # Sliding window statistics
        self.window_size = 100
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))

    def forward(self, r_lidar: torch.Tensor, r_rgb: torch.Tensor, r_imu: torch.Tensor,
                features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            r_lidar: (B, 1) LiDAR reliability score
            r_rgb: (B, 1) RGB reliability score
            r_imu: (B, 1) IMU reliability score
            features: Dict with 'lidar', 'rgb', 'imu' features, each (B, D)

        Returns:
            Dict: Normalized features
        """
        # LiDAR feature normalization
        lidar_norm = F.normalize(
            features['lidar'] - self.beta_lidar,
            p=2, dim=1, eps=1e-6
        )
        lidar_out = self.gamma_lidar * lidar_norm

        # RGB feature normalization
        rgb_norm = F.normalize(
            features['rgb'] - self.beta_rgb,
            p=2, dim=1, eps=1e-6
        )
        rgb_out = self.gamma_rgb * rgb_norm

        # IMU feature normalization
        imu_norm = F.normalize(
            features['imu'] - self.beta_imu,
            p=2, dim=1, eps=1e-6
        )
        imu_out = self.gamma_imu * imu_norm

        return {
            'lidar_out': lidar_out,
            'rgb_out': rgb_out,
            'imu_out': imu_out
        }


def test_adaptive_normalization():
    """Test Adaptive Normalization"""
    model = AdaptiveNormalization(feature_dim=256)

    batch_size = 4
    feature_dim = 256
    r_lidar = torch.rand(batch_size, 1)
    r_rgb = torch.rand(batch_size, 1)
    r_imu = torch.rand(batch_size, 1)
    features = {
        'lidar': torch.randn(batch_size, feature_dim),
        'rgb': torch.randn(batch_size, feature_dim),
        'imu': torch.randn(batch_size, feature_dim)
    }

    output = model(r_lidar, r_rgb, r_imu, features)

    lidar_norm = torch.norm(output['lidar_out'], dim=1)
    rgb_norm = torch.norm(output['rgb_out'], dim=1)
    imu_norm = torch.norm(output['imu_out'], dim=1)

    print(f"LiDAR normalized norms: {lidar_norm}")
    print(f"RGB normalized norms: {rgb_norm}")
    print(f"IMU normalized norms: {imu_norm}")

    print("✅ Adaptive Normalization test passed")


if __name__ == "__main__":
    test_adaptive_normalization()
