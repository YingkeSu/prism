"""quality_aware_attention.py

Quality-aware attention gate for suppressing low-quality modalities.

Produces a gate that suppresses low-quality modalities so they cannot dominate fusion.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class QualityAwareAttention(nn.Module):
    """
    Quality-Aware Attention Gate.

    Produces a gate that suppresses low-quality modalities so they cannot
    dominate fusion.

    Args:
        num_modalities: Number of modalities Q (default 3: lidar, rgb, imu)
        hidden_dim: MLP hidden dimension (default 32)

    Input:
        quality_now: (B, Q) current quality scores [0, 1]
        quality_pred: Optional (B, Q) predicted quality scores [0, 1]

    Output:
        gate: (B, Q) attention gate values [0, 1]
    """

    def __init__(
        self,
        num_modalities: int = 3,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim

        # Input dimension: 2*Q (current + predicted) or Q (current only)
        input_dim = num_modalities * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Sigmoid(),
        )

    def forward(
        self,
        quality_now: torch.Tensor,
        quality_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            quality_now: (B, Q) current quality [0, 1]
            quality_pred: Optional (B, Q) predicted quality [0, 1]

        Returns:
            gate: (B, Q) attention gate [0, 1]

        Raises:
            ValueError: If input shapes are incorrect
        """
        if quality_now.ndim != 2:
            raise ValueError(
                f"Expected quality_now (B, Q), got shape {tuple(quality_now.shape)}"
            )

        B, Q = quality_now.shape

        if Q != self.num_modalities:
            raise ValueError(
                f"Expected Q={self.num_modalities} modalities, got Q={Q}"
            )

        # If no predicted quality, use current quality twice
        if quality_pred is None:
            quality_pred = quality_now.clone()
        else:
            if quality_pred.shape != quality_now.shape:
                raise ValueError(
                    f"quality_pred shape {tuple(quality_pred.shape)} "
                    f"must match quality_now shape {tuple(quality_now.shape)}"
                )

        # Concatenate current and predicted quality: (B, 2*Q)
        quality_concat = torch.cat([quality_now, quality_pred], dim=1)

        # MLP forward + sigmoid: (B, 2*Q) -> (B, Q)
        gate = self.mlp(quality_concat)

        return gate


def test_quality_aware_attention() -> None:
    """Test Quality-Aware Attention."""
    model = QualityAwareAttention(num_modalities=3, hidden_dim=32)

    # Create test data
    batch_size = 4
    num_modalities = 3

    # Random current quality
    quality_now = torch.rand(batch_size, num_modalities)

    # Case 1: Without predicted quality
    gate_no_pred = model(quality_now)

    # Verify output dimensions
    assert gate_no_pred.shape == (batch_size, num_modalities)

    # Verify output range [0, 1]
    assert torch.all(gate_no_pred >= 0) and torch.all(gate_no_pred <= 1)

    # Case 2: With predicted quality
    quality_pred = torch.rand(batch_size, num_modalities)
    gate_with_pred = model(quality_now, quality_pred)

    assert gate_with_pred.shape == (batch_size, num_modalities)
    assert torch.all(gate_with_pred >= 0) and torch.all(gate_with_pred <= 1)

    # Case 3: Test low-quality suppression
    # High quality for lidar/rgb, low for imu
    high_low_quality = torch.tensor([
        [0.9, 0.8, 0.1],  # Sample 0
        [0.95, 0.85, 0.05],  # Sample 1
    ])
    gate = model(high_low_quality, high_low_quality)

    # IMU gate should be lower than others
    assert gate[0, 2] < gate[0, 0], "Low quality IMU should have lower gate"
    assert gate[0, 2] < gate[0, 1], "Low quality IMU should have lower gate"
    assert gate[1, 2] < gate[1, 0], "Low quality IMU should have lower gate"
    assert gate[1, 2] < gate[1, 1], "Low quality IMU should have lower gate"

    print("✅ QualityAwareAttention test passed")
    print(f"Sample 0: LiDAR gate={gate[0, 0].item():.3f}, "
          f"RGB gate={gate[0, 1].item():.3f}, IMU gate={gate[0, 2].item():.3f}")
    print(f"Sample 1: LiDAR gate={gate[1, 0].item():.3f}, "
          f"RGB gate={gate[1, 1].item():.3f}, IMU gate={gate[1, 2].item():.3f}")


if __name__ == "__main__":
    test_quality_aware_attention()
