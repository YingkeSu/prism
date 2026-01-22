"""
Reliability-Aware Fusion Module

Complete fusion module integrating:
1. Reliability estimation
2. Dynamic weighting
3. Adaptive normalization
4. Feature fusion
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from networks.reliability_predictor import ReliabilityPredictor
from networks.dynamic_weighting_layer import DynamicWeightingLayer
from networks.adaptive_normalization import AdaptiveNormalization


class ReliabilityAwareFusionModule(nn.Module):
    """
    Reliability-Aware Fusion Module

    Integrates all sub-modules for complete reliability-aware fusion.

    Args:
        feature_dim: Feature dimension (default 256)
        num_heads: Number of attention heads (default 8)

    Input:
        lidar_points: (B, N, 3) LiDAR point cloud
        rgb_image: (B, 3, H, W) RGB image
        imu_data: (B, T, 6) IMU sequence

    Output:
        Dict: Fusion output + intermediate info (for visualization)
    """

    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()

        # Sub-modules
        self.reliability_estimator = ReliabilityPredictor()
        self.dynamic_weighting = DynamicWeightingLayer(
            feature_dim=feature_dim,
            num_heads=num_heads
        )
        self.adaptive_norm = AdaptiveNormalization(feature_dim=feature_dim)

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True)
        )

        # Encoders (from fused features to output)
        self.lidar_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 3),
            nn.LayerNorm(feature_dim // 3),
            nn.ReLU(inplace=True)
        )

        self.rgb_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 3),
            nn.LayerNorm(feature_dim // 3),
            nn.ReLU(inplace=True)
        )

        self.imu_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 3),
            nn.LayerNorm(feature_dim // 3),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        lidar_points: torch.Tensor,
        rgb_image: torch.Tensor,
        imu_data: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass

        Args:
            lidar_points: (B, N, 3) LiDAR point cloud
            rgb_image: (B, 3, H, W) RGB image
            imu_data: (B, T, 6) IMU sequence

        Returns:
            Dict: Fusion output + intermediate info (for visualization)
        """
        batch_size = lidar_points.shape[0]

        # 1. Reliability estimation
        reliability_output = self.reliability_estimator(
            lidar_points, rgb_image, imu_data
        )

        r_lidar = reliability_output['r_lidar']
        r_rgb = reliability_output['r_rgb']
        r_imu = reliability_output['r_imu']
        features = reliability_output['features']

        # 2. Encode modality features
        lidar_feat = self.lidar_encoder(reliability_output['features'])
        rgb_feat = self.rgb_encoder(reliability_output['features'])
        imu_feat = self.imu_encoder(reliability_output['features'])

        # 3. Dynamic weighting
        weights = self.dynamic_weighting(lidar_feat, rgb_feat, imu_feat)

        w_lidar = weights['w_lidar']
        w_rgb = weights['w_rgb']
        w_imu = weights['w_imu']

        # 4. Adaptive normalization
        raw_features = {
            'lidar': lidar_feat,
            'rgb': rgb_feat,
            'imu': imu_feat
        }
        normed_features = self.adaptive_norm(r_lidar, r_rgb, r_imu, raw_features)

        # 5. Weighted fusion
        fused = torch.cat([
            normed_features['lidar_out'] * w_lidar,
            normed_features['rgb_out'] * w_rgb,
            normed_features['imu_out'] * w_imu
        ], dim=1)  # (B, feature_dim)

        # 6. Pass through fusion network
        output = self.fusion_net(fused)  # (B, 64)

        return {
            'output': output,
            'reliability': {
                'lidar': r_lidar,
                'rgb': r_rgb,
                'imu': r_imu
            },
            'weights': weights,
            'normed_features': normed_features
        }

    def get_loss(self, predictions: Dict[str, Any], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Main loss (MSE)
        mse_loss = nn.MSELoss()(predictions['output'], targets.float())

        # Reliability regularization (encourage balanced modality usage)
        reliability = predictions['reliability']
        r_lidar = reliability['lidar']
        r_rgb = reliability['rgb']
        r_imu = reliability['imu']

        # Variance regularization (encourage balance)
        mean_r = (r_lidar + r_rgb + r_imu) / 3.0
        var_r = ((r_lidar - mean_r)**2 + (r_rgb - mean_r)**2 + (r_imu - mean_r)**2) / 3.0
        reliability_reg = 0.01 * var_r.mean()

        # Total loss
        total_loss = mse_loss + reliability_reg

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'reliability_reg': reliability_reg
        }


def test_reliability_aware_fusion():
    """Test Reliability-Aware Fusion Module"""
    model = ReliabilityAwareFusionModule(feature_dim=256, num_heads=8)

    # Create test data
    batch_size = 4
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)
    targets = torch.randn(batch_size, 64)

    # Forward pass
    output = model(lidar_points, rgb_image, imu_data)

    print(f"Output shape: {output['output'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1000:.1f}K")

    # Compute loss
    loss_dict = model.get_loss(output, targets)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"MSE loss: {loss_dict['mse_loss'].item():.4f}")
    print(f"Reliability reg: {loss_dict['reliability_reg'].item():.4f}")

    print("✅ Reliability-Aware Fusion Module test passed")


if __name__ == "__main__":
    test_reliability_aware_fusion()
