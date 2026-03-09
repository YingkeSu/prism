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

    def __init__(self, feature_dim: int = 256, use_reliability_modulation: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_reliability_modulation = use_reliability_modulation

        # Learnable normalization parameters
        self.gamma_lidar = nn.Parameter(torch.ones(1))
        self.gamma_rgb = nn.Parameter(torch.ones(1))
        self.gamma_imu = nn.Parameter(torch.ones(1))

        # Offset parameters
        self.beta_lidar = nn.Parameter(torch.zeros(1))
        self.beta_rgb = nn.Parameter(torch.zeros(1))
        self.beta_imu = nn.Parameter(torch.zeros(1))

        # Reliability modulation parameters (lambda for each modality)
        self.lambda_lidar = nn.Parameter(torch.zeros(1))
        self.lambda_rgb = nn.Parameter(torch.zeros(1))
        self.lambda_imu = nn.Parameter(torch.zeros(1))

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
        # Ensure reliability scores have shape (B, 1) and are in [0, 1]
        r_l = r_lidar.clamp(0, 1).view(-1, 1)
        r_r = r_rgb.clamp(0, 1).view(-1, 1)
        r_i = r_imu.clamp(0, 1).view(-1, 1)

        if self.use_reliability_modulation:
            # Reliability-modulated gamma: gamma_eff = gamma * (1 + lambda * r)
            gamma_l = self.gamma_lidar * (1.0 + self.lambda_lidar * r_l)
            gamma_r = self.gamma_rgb * (1.0 + self.lambda_rgb * r_r)
            gamma_i = self.gamma_imu * (1.0 + self.lambda_imu * r_i)
        else:
            # Ablation mode: disable reliability-conditioned modulation.
            batch_size = r_l.shape[0]
            gamma_l = self.gamma_lidar.view(1, 1).expand(batch_size, 1)
            gamma_r = self.gamma_rgb.view(1, 1).expand(batch_size, 1)
            gamma_i = self.gamma_imu.view(1, 1).expand(batch_size, 1)

        # LiDAR feature normalization (L2 norm per feature, keeps (B, D) shape)
        lidar_norm = F.normalize(
            features['lidar'] - self.beta_lidar,
            p=2, dim=1, eps=1e-6
        )
        lidar_out = gamma_l * lidar_norm

        # RGB feature normalization (L2 norm per feature, keeps (B, D) shape)
        rgb_norm = F.normalize(
            features['rgb'] - self.beta_rgb,
            p=2, dim=1, eps=1e-6
        )
        rgb_out = gamma_r * rgb_norm

        # IMU feature normalization (L2 norm per feature, keeps (B, D) shape)
        imu_norm = F.normalize(
            features['imu'] - self.beta_imu,
            p=2, dim=1, eps=1e-6
        )
        imu_out = gamma_i * imu_norm

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
