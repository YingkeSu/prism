"""temporal_quality_predictor.py

Temporal quality predictor using LSTM over historical quality scores.

Predicts next-step modality quality from a fixed-length history.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TemporalQualityPredictor(nn.Module):
    """
    Temporal Quality Predictor with LSTM.

    Predicts next-step modality quality from a fixed-length history.

    Args:
        seq_len: Length of historical quality sequence (default 10)
        num_modalities: Number of modalities Q (default 3: lidar, rgb, imu)
        hidden_dim: LSTM hidden dimension (default 64)

    Input:
        historical_quality: (B, T, Q) quality history
            - B: batch_size
            - T: seq_len (time steps)
            - Q: num_modalities (3: lidar, rgb, imu)

    Output:
        Dict[str, torch.Tensor]: {
            'q_lidar': (B, 1),  # Predicted LiDAR quality [0, 1]
            'q_rgb': (B, 1),    # Predicted RGB quality [0, 1]
            'q_imu': (B, 1),    # Predicted IMU quality [0, 1]
            'features': (B, hidden_dim)  # LSTM final hidden state
        }
    """

    def __init__(
        self,
        seq_len: int = 10,
        num_modalities: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim

        # Shared LSTM for all modalities
        self.lstm = nn.LSTM(
            input_size=num_modalities,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Per-modality prediction heads
        self.lidar_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.imu_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, historical_quality: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            historical_quality: (B, T, Q) quality history

        Returns:
            Dict with predicted qualities and LSTM features

        Raises:
            ValueError: If input shape is incorrect
        """
        if historical_quality.ndim != 3:
            raise ValueError(
                f"Expected historical_quality (B, T, Q), got shape {tuple(historical_quality.shape)}"
            )

        B, T, Q = historical_quality.shape

        if Q != self.num_modalities:
            raise ValueError(
                f"Expected Q={self.num_modalities} modalities, got Q={Q}"
            )

        # LSTM forward: (B, T, Q) -> (B, T, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(historical_quality)

        # Use final hidden state: (1, B, hidden_dim) -> (B, hidden_dim)
        features = h_n.squeeze(0)

        # Predict per-modality quality
        q_lidar = self.lidar_head(features)  # (B, 1)
        q_rgb = self.rgb_head(features)  # (B, 1)
        q_imu = self.imu_head(features)  # (B, 1)

        return {
            'q_lidar': q_lidar,
            'q_rgb': q_rgb,
            'q_imu': q_imu,
            'features': features,
        }


def test_temporal_quality_predictor() -> None:
    """Test Temporal Quality Predictor."""
    model = TemporalQualityPredictor(seq_len=10, num_modalities=3, hidden_dim=64)

    # Create test data
    batch_size = 4
    seq_len = 10
    num_modalities = 3

    # Random quality history in [0, 1]
    historical_quality = torch.rand(batch_size, seq_len, num_modalities)

    # Forward pass
    output = model(historical_quality)

    # Verify output dimensions
    assert output['q_lidar'].shape == (batch_size, 1)
    assert output['q_rgb'].shape == (batch_size, 1)
    assert output['q_imu'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 64)

    # Verify output range [0, 1]
    for key in ['q_lidar', 'q_rgb', 'q_imu']:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), \
            f"{key} range error"

    # Test with varying quality patterns
    # Just verify output range is valid
    print("✅ TemporalQualityPredictor test passed")
    print(f"Sample 0: LiDAR={output['q_lidar'][0].item():.3f}, "
          f"RGB={output['q_rgb'][0].item():.3f}, IMU={output['q_imu'][0].item():.3f}")
    print(f"Sample 1: LiDAR={output['q_lidar'][1].item():.3f}, "
          f"RGB={output['q_rgb'][1].item():.3f}, IMU={output['q_imu'][1].item():.3f}")


if __name__ == "__main__":
    test_temporal_quality_predictor()
