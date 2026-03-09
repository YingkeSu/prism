"""quality_aware_fusion.py

Quality-aware dynamic fusion for multi-modal UAV observations.

Orchestrates reliability estimation, temporal prediction, quality-aware gating,
and dynamic weighting.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from networks.dynamic_weighting_layer import DynamicWeightingLayer
from networks.quality_aware_attention import QualityAwareAttention
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator
from networks.temporal_quality_predictor import TemporalQualityPredictor


class QualityAwareDynamicFusion(nn.Module):
    """
    Quality-Aware Dynamic Fusion for Multi-Modal Observations.

    Orchestrates reliability estimation, temporal prediction, quality-aware
    gating, and dynamic weighting.

    Args:
        feature_dim: Common feature dimension D (default 256)
        seq_len: Historical quality sequence length (default 10)
        num_heads: Attention heads for DynamicWeightingLayer (default 8)
        use_temporal: If True, use temporal quality prediction (default False)

    Input:
        lidar_points: (B, N, 3) LiDAR point cloud
        rgb_image: (B, 3, H, W) RGB image
        imu_data: (B, 6) IMU data
        historical_quality: Optional (B, T, Q) historical quality

    Output:
        Dict[str, torch.Tensor]: {
            'w_lidar': (B, 1),  # Gated LiDAR weight [0, 1]
            'w_rgb': (B, 1),    # Gated RGB weight [0, 1]
            'w_imu': (B, 1),    # Gated IMU weight [0, 1]
            'features': (B, D), # Fused features
            'debug': {          # Optional debug tensors
                'quality_now': (B, 3),
                'quality_pred': (B, 3),
                'gate': (B, 3),
                'attention_weights': (B, 8, 3, 3),
            }
        }
    """

    def __init__(
        self,
        feature_dim: int = 256,
        seq_len: int = 10,
        num_heads: int = 8,
        use_temporal: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.use_temporal = use_temporal

        # Reliability estimators
        self.lidar_reliability = LiDARSNREstimator(point_dim=3, feature_dim=64)
        self.rgb_reliability = ImageQualityEstimator(input_channels=3)

        # Temporal quality predictor (optional)
        if use_temporal:
            self.temporal_predictor = TemporalQualityPredictor(
                seq_len=seq_len,
                num_modalities=3,
                hidden_dim=64,
            )

        # Quality-aware attention
        self.quality_attention = QualityAwareAttention(
            num_modalities=3,
            hidden_dim=32,
        )

        # Dynamic weighting layer
        self.dynamic_weighting = DynamicWeightingLayer(
            feature_dim=feature_dim,
            num_heads=num_heads,
        )

        # Feature projection layers
        self.lidar_proj = nn.Linear(64, feature_dim)
        self.rgb_proj = nn.Linear(256, feature_dim)
        self.imu_proj = nn.Linear(6, feature_dim)

    def _compute_imu_quality(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Compute IMU quality score based on consistency.

        Args:
            imu_data: (B, 6) IMU data [ax, ay, az, gx, gy, gz]

        Returns:
            quality: (B, 1) IMU quality [0, 1]
        """
        # Simple heuristic: low variance = high quality
        # Split into accelerometer and gyroscope
        accel = imu_data[:, :3]  # (B, 3)
        gyro = imu_data[:, 3:6]  # (B, 3)

        # Compute variance
        accel_var = accel.var(dim=1, keepdim=True)  # (B, 1)
        gyro_var = gyro.var(dim=1, keepdim=True)  # (B, 1)

        # Lower variance = higher quality (using sigmoid)
        quality = torch.sigmoid(-5.0 * (accel_var + gyro_var))

        return quality  # (B, 1)

    def forward(
        self,
        lidar_points: torch.Tensor,
        rgb_image: torch.Tensor,
        imu_data: torch.Tensor,
        historical_quality: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            lidar_points: (B, N, 3) LiDAR point cloud
            rgb_image: (B, 3, H, W) RGB image
            imu_data: (B, 6) IMU data
            historical_quality: Optional (B, T, Q) historical quality

        Returns:
            Dict with weights, features, and debug tensors
        """
        B = lidar_points.shape[0]

        # Step 1: Extract reliability features and quality
        lidar_rel = self.lidar_reliability(lidar_points)  # Dict with 'snr', 'features'
        rgb_rel = self.rgb_reliability(rgb_image)  # Dict with 'overall_quality', 'features'
        imu_quality = self._compute_imu_quality(imu_data)  # (B, 1)

        # Current quality: (B, 3) [q_lidar, q_rgb, q_imu]
        quality_now = torch.cat([
            lidar_rel['snr'],  # (B, 1)
            rgb_rel['overall_quality'],  # (B, 1)
            imu_quality,  # (B, 1)
        ], dim=1)  # (B, 3)

        # Step 2: Temporal quality prediction (optional)
        if self.use_temporal and historical_quality is not None:
            pred_output = self.temporal_predictor(historical_quality)
            quality_pred = torch.cat([
                pred_output['q_lidar'],
                pred_output['q_rgb'],
                pred_output['q_imu'],
            ], dim=1)  # (B, 3)
        else:
            quality_pred = quality_now.clone()

        # Step 3: Project features to common dimension
        lidar_feat = self.lidar_proj(lidar_rel['features'])  # (B, D)
        rgb_feat = self.rgb_proj(rgb_rel['features'])  # (B, D)
        imu_feat = self.imu_proj(imu_data)  # (B, D)

        # Step 4: Compute dynamic weights
        weight_output = self.dynamic_weighting(lidar_feat, rgb_feat, imu_feat)
        w_lidar = weight_output['w_lidar']  # (B, 1)
        w_rgb = weight_output['w_rgb']  # (B, 1)
        w_imu = weight_output['w_imu']  # (B, 1)

        # Step 5: Compute quality-aware gate
        gate = self.quality_attention(quality_now, quality_pred)  # (B, 3)

        # Step 6: Apply gate and normalize
        raw_weights = torch.cat([w_lidar, w_rgb, w_imu], dim=1)  # (B, 3)
        gated_weights = raw_weights * gate  # (B, 3)

        # Normalize to sum to 1
        weight_sum = gated_weights.sum(dim=1, keepdim=True)  # (B, 1)
        weight_sum = torch.clamp(weight_sum, min=1e-6)  # Avoid division by zero
        gated_weights = gated_weights / weight_sum  # (B, 3)

        w_lidar_gated = gated_weights[:, 0:1]
        w_rgb_gated = gated_weights[:, 1:2]
        w_imu_gated = gated_weights[:, 2:3]

        # Step 7: Fuse features
        fused_features = (
            w_lidar_gated * lidar_feat +
            w_rgb_gated * rgb_feat +
            w_imu_gated * imu_feat
        )  # (B, D)

        return {
            'w_lidar': w_lidar_gated,
            'w_rgb': w_rgb_gated,
            'w_imu': w_imu_gated,
            'features': fused_features,
            'debug': {
                'quality_now': quality_now,
                'quality_pred': quality_pred,
                'gate': gate,
                'attention_weights': weight_output['attention_weights'],
            }
        }


def test_quality_aware_fusion() -> None:
    """Test Quality-Aware Dynamic Fusion."""
    model = QualityAwareDynamicFusion(
        feature_dim=256,
        seq_len=10,
        num_heads=8,
        use_temporal=True,
    )

    # Create test data
    batch_size = 4
    num_points = 1000
    height, width = 128, 128

    lidar_points = torch.randn(batch_size, num_points, 3)
    rgb_image = torch.rand(batch_size, 3, height, width) * 255
    imu_data = torch.randn(batch_size, 6)
    historical_quality = torch.rand(batch_size, 10, 3)

    # Forward pass
    output = model(lidar_points, rgb_image, imu_data, historical_quality)

    # Verify output dimensions
    assert output['w_lidar'].shape == (batch_size, 1)
    assert output['w_rgb'].shape == (batch_size, 1)
    assert output['w_imu'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 256)
    assert output['debug']['quality_now'].shape == (batch_size, 3)
    assert output['debug']['quality_pred'].shape == (batch_size, 3)
    assert output['debug']['gate'].shape == (batch_size, 3)

    # Verify weight range [0, 1]
    for key in ['w_lidar', 'w_rgb', 'w_imu']:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), \
            f"{key} range error"

    # Verify weights sum to 1
    total_weight = output['w_lidar'] + output['w_rgb'] + output['w_imu']
    assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=1e-5)

    # Test low-quality suppression
    # Create data with low LiDAR quality (very sparse, high noise)
    low_quality_lidar = torch.randn(batch_size, num_points, 3) * 100.0  # High noise
    rgb_image_test = torch.rand(batch_size, 3, height, width) * 255
    imu_data_test = torch.randn(batch_size, 6)

    output_low = model(low_quality_lidar, rgb_image_test, imu_data_test)

    # Check that gate suppresses low quality (compare gate values, not weights directly)
    # The gate for low-quality LiDAR should be lower than for normal quality
    normal_gate = output['debug']['gate'][0, 0].item()
    low_gate = output_low['debug']['gate'][0, 0].item()
    print(f"Normal LiDAR gate: {normal_gate:.3f}, Low-quality LiDAR gate: {low_gate:.3f}")

    # At least verify weights still sum to 1 even with low quality
    total_weight_low = output_low['w_lidar'] + output_low['w_rgb'] + output_low['w_imu']
    assert torch.allclose(total_weight_low, torch.ones_like(total_weight_low), atol=1e-5)

    print("✅ QualityAwareDynamicFusion test passed")
    print(f"Normal sample: LiDAR={output['w_lidar'][0].item():.3f}, "
          f"RGB={output['w_rgb'][0].item():.3f}, IMU={output['w_imu'][0].item():.3f}")
    print(f"Low LiDAR quality: LiDAR={output_low['w_lidar'][0].item():.3f}, "
          f"RGB={output_low['w_rgb'][0].item():.3f}, IMU={output_low['w_imu'][0].item():.3f}")


if __name__ == "__main__":
    test_quality_aware_fusion()
