"""
Reliability Predictor Network

Fuses quality metrics from 3 modalities to predict reliability scores for each modality.
"""

import torch
import torch.nn as nn
from typing import Dict
from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.imu_consistency_checker import IMUConsistencyChecker


class ReliabilityPredictor(nn.Module):
    """
    Lightweight Reliability Prediction Network

    Fuses quality metrics from 3 modalities, predicts reliability scores for each modality.

    Args:
        lidar_dim: LiDAR feature dimension (default 64)
        rgb_dim: RGB feature dimension (default 256)
        imu_dim: IMU feature dimension (default 64)
        imu_window_size: IMU temporal window used by consistency checker (default 16)
        hidden_dim: Hidden layer dimension (default 128)
        output_dim: Output feature dimension (default 256)

    Input:
        lidar_points: (B, N, 3) LiDAR point cloud
        rgb_image: (B, 3, H, W) RGB image
        imu_data: (B, T, 6) IMU sequence

    Output:
        Dict: {
            'r_lidar': (B, 1),      # LiDAR reliability [0, 1]
            'r_rgb': (B, 1),        # RGB reliability [0, 1]
            'r_imu': (B, 1),         # IMU reliability [0, 1]
            'features': (B, output_dim) # Fused features
        }
    """

    def __init__(
        self,
        lidar_dim: int = 64,
        rgb_dim: int = 256,
        imu_dim: int = 64,
        imu_window_size: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 256
    ):
        super().__init__()

        # 3 reliability estimators
        self.lidar_estimator = LiDARSNREstimator(point_dim=3, feature_dim=lidar_dim)
        self.rgb_estimator = ImageQualityEstimator(input_channels=3)
        self.imu_estimator = IMUConsistencyChecker(
            imu_dim=6,
            window_size=max(1, int(imu_window_size)),
            feature_dim=imu_dim
        )

        # Encoders - simplified 2-layer design to reduce parameters
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.rgb_encoder = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        # Feature fusion - simplified
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2, output_dim),
            nn.ReLU(inplace=True)
        )

        # Reliability score prediction heads - simplified
        self.lidar_reliability = nn.Linear(output_dim, 1)
        self.rgb_reliability = nn.Linear(output_dim, 1)
        self.imu_reliability = nn.Linear(output_dim, 1)

    def forward(
        self,
        lidar_points: torch.Tensor,
        rgb_image: torch.Tensor,
        imu_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            lidar_points: (B, N, 3)
            rgb_image: (B, 3, H, W)
            imu_data: (B, T, 6)

        Returns:
            Dict: Reliability scores and fused features
        """
        # 1. Estimate quality metrics for each modality
        lidar_output = self.lidar_estimator(lidar_points)
        rgb_output = self.rgb_estimator(rgb_image)
        imu_output = self.imu_estimator(imu_data)

        # 2. Extract features from each estimator
        lidar_snr = lidar_output['snr']  # (B, 1)
        rgb_quality = rgb_output['overall_quality']  # (B, 1)
        imu_consistency = imu_output['consistency']  # (B, 1)

        # 3. Encode modality features (use raw features from estimators)
        lidar_feat = self.lidar_encoder(lidar_output['features'])
        rgb_feat = self.rgb_encoder(rgb_output['features'])
        imu_feat = self.imu_encoder(imu_output['features'])

        # 4. Fuse features
        fused = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
        features = self.feature_fusion(fused)  # (B, output_dim)

        # 5. Predict reliability scores with sigmoid activation
        r_lidar = torch.sigmoid(self.lidar_reliability(features))  # (B, 1)
        r_rgb = torch.sigmoid(self.rgb_reliability(features))  # (B, 1)
        r_imu = torch.sigmoid(self.imu_reliability(features))  # (B, 1)

        return {
            'r_lidar': r_lidar,
            'r_rgb': r_rgb,
            'r_imu': r_imu,
            'features': features
        }

    def count_parameters(self) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_reliability_predictor():
    """Test Reliability Predictor"""
    model = ReliabilityPredictor(
        lidar_dim=64,
        rgb_dim=256,
        imu_dim=64,
        hidden_dim=128,
        output_dim=256
    )

    # Create test data
    batch_size = 4
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)

    # Forward pass
    output = model(lidar_points, rgb_image, imu_data)

    # Verify output dimensions
    assert output['r_lidar'].shape == (batch_size, 1)
    assert output['r_rgb'].shape == (batch_size, 1)
    assert output['r_imu'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 256)

    # Verify output range
    for key in ['r_lidar', 'r_rgb', 'r_imu']:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), f"{key} range error"

    # Verify parameter count
    num_params = model.count_parameters()
    assert num_params < 500000, f"Parameter count {num_params} exceeds 500K"

    print("✅ Reliability Predictor test passed")
    print(f"Total parameters: {num_params / 1000:.1f}K")
    print(f"LiDAR reliability: {output['r_lidar'][0].item():.3f}")
    print(f"RGB reliability: {output['r_rgb'][0].item():.3f}")
    print(f"IMU reliability: {output['r_imu'][0].item():.3f}")


if __name__ == "__main__":
    test_reliability_predictor()
