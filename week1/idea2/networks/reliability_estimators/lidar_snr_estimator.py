"""
LiDAR Point Cloud SNR Estimator

Evaluates point cloud quality metrics:
- Point cloud density
- Point cloud distribution uniformity
- Signal-to-noise ratio (SNR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LiDARSNREstimator(nn.Module):
    """
    LiDAR Point Cloud SNR Estimator

    Evaluates point cloud quality metrics:
    - Point cloud density
    - Point cloud distribution uniformity
    - Signal-to-noise ratio (SNR)

    Args:
        point_dim: Point cloud dimension (default 3: x, y, z)
        feature_dim: Feature dimension (default 64)

    Input:
        lidar_points: (B, N, 3) LiDAR point cloud

    Output:
        Dict: {
            'snr': (B, 1),           # SNR [0, 1]
            'density': (B, 1),        # Point density [0, 1]
            'uniformity': (B, 1),     # Uniformity [0, 1]
            'features': (B, feature_dim) # Extracted features
        }
    """

    def __init__(self, point_dim: int = 3, feature_dim: int = 64):
        super().__init__()
        self.point_dim = point_dim
        self.feature_dim = feature_dim

        # Statistical feature extraction network
        self.point_stats = nn.Sequential(
            # Conv1d: (B, 3, N) -> (B, 32, N)
            nn.Conv1d(point_dim, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # Conv1d: (B, 32, N) -> (B, 64, N)
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # AdaptiveAvgPool1d: (B, 64, N) -> (B, 64, 1)
            nn.AdaptiveAvgPool1d(1)
        )

        # SNR prediction head
        self.snr_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Density prediction head
        self.density_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Uniformity prediction head
        self.uniformity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, lidar_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            lidar_points: (B, N, 3) LiDAR point cloud
                          B: batch_size
                          N: num_points (variable, e.g., 1000)
                          3: (x, y, z)

        Returns:
            Dict: Contains snr, density, uniformity, features
        """
        # Transpose: (B, N, 3) -> (B, 3, N) for Conv1d
        x = lidar_points.transpose(1, 2)

        # Extract statistical features
        stats = self.point_stats(x)  # (B, 64, 1) -> (B, 64)
        stats = stats.squeeze(-1)  # (B, 64)

        # Predict metrics
        snr = self.snr_head(stats)  # (B, 1)
        density = self.density_head(stats)  # (B, 1)
        uniformity = self.uniformity_head(stats)  # (B, 1)

        return {
            'snr': snr,
            'density': density,
            'uniformity': uniformity,
            'features': stats
        }

    def compute_traditional_metrics(self, lidar_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute traditional SNR metrics (for validation)

        Args:
            lidar_points: (B, N, 3)

        Returns:
            Dict: Traditional SNR metrics
        """
        B, N, _ = lidar_points.shape

        # 1. Point cloud density
        density = N / 1000.0  # Normalize to [0, 1]

        # 2. Point cloud distribution uniformity
        center = lidar_points.mean(dim=1, keepdim=True)  # (B, 1, 3)
        distances = torch.norm(lidar_points - center, dim=-1)  # (B, N)
        std_distance = distances.std(dim=1, keepdim=True)  # (B, 1)
        uniformity = torch.exp(-std_distance / 5.0)  # Lower std = more uniform

        # 3. Overall SNR
        snr = 0.4 * density + 0.6 * uniformity

        return {
            'density': density,
            'uniformity': uniformity,
            'snr': snr,
            'std_distance': std_distance
        }


def test_lidar_snr_estimator():
    """Test LiDAR SNR Estimator"""
    model = LiDARSNREstimator(point_dim=3, feature_dim=64)

    # Create test data
    batch_size = 4
    num_points = 1000
    lidar_points = torch.randn(batch_size, num_points, 3)

    # Simulate different quality point clouds
    # High quality: dense, uniform
    lidar_points[0] = torch.randn(num_points, 3) * 0.1  # Dense point cloud
    # Low quality: sparse, non-uniform
    lidar_points[1] = torch.cat([
        torch.randn(100, 3) * 0.1,  # Dense region
        torch.randn(900, 3) * 10.0   # Sparse region
    ], dim=0)

    # Forward pass
    output = model(lidar_points)

    # Verify output dimensions
    assert output['snr'].shape == (batch_size, 1)
    assert output['density'].shape == (batch_size, 1)
    assert output['uniformity'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 64)

    # Verify output range
    assert torch.all(output['snr'] >= 0) and torch.all(output['snr'] <= 1)
    assert torch.all(output['density'] >= 0) and torch.all(output['density'] <= 1)
    assert torch.all(output['uniformity'] >= 0) and torch.all(output['uniformity'] <= 1)

    # High quality point cloud should have higher SNR
    assert output['snr'][0] > output['snr'][1]

    print("✅ LiDAR SNR Estimator test passed")
    print(f"High SNR: {output['snr'][0].item():.3f}")
    print(f"Low SNR: {output['snr'][1].item():.3f}")


if __name__ == "__main__":
    test_lidar_snr_estimator()
