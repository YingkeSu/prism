"""
Dynamic Weighting Layer

Multi-head attention based dynamic weighting for multi-modal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DynamicWeightingLayer(nn.Module):
    """
    Dynamic Weighting Layer with Attention-based Multi-Modal Fusion

    Args:
        feature_dim: Feature dimension (default 256)
        num_heads: Number of attention heads (default 8)

    Input:
        lidar_feat: (B, D) LiDAR feature
        rgb_feat: (B, D) RGB feature
        imu_feat: (B, D) IMU feature
        temperature: Optional temperature scaling parameter

    Output:
        Dict: {
            'w_lidar': (B, 1),            # LiDAR weight [0, 1]
            'w_rgb': (B, 1),              # RGB weight [0, 1]
            'w_imu': (B, 1),               # IMU weight [0, 1]
            'attention_scores': (B, 3),      # Attention scores
            'attention_weights': (B, 8, 3)  # Multi-head attention weights
        }
    """

    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Gating mechanism (temperature scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

        # Learnable weight biases
        self.bias_lidar = nn.Parameter(torch.zeros(1))
        self.bias_rgb = nn.Parameter(torch.zeros(1))
        self.bias_imu = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        lidar_feat: torch.Tensor,
        rgb_feat: torch.Tensor,
        imu_feat: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B = lidar_feat.shape[0]

        # 1. Stack multi-modal features: (B, 3, D)
        multimodal_feat = torch.stack([lidar_feat, rgb_feat, imu_feat], dim=1)

        # 2. Use multi-head attention to compute attention weights
        attention_output, attention_weights = self.multi_head_attention(
            multimodal_feat, multimodal_feat, multimodal_feat
        )

        # 3. Compute average attention scores across heads
        # attention_weights shape: (B, L, S) where L=3 (modalities), S=3
        attention_scores = attention_weights.mean(dim=2)  # (B, L)

        # 4. Apply temperature scaling
        if temperature is None:
            temperature = torch.sigmoid(self.temperature)
        attention_scores = attention_scores / temperature

        # 5. Add bias and compute weights via softmax
        lidar_logits = attention_scores[:, 0:1] + self.bias_lidar
        rgb_logits = attention_scores[:, 1:2] + self.bias_rgb
        imu_logits = attention_scores[:, 2:3] + self.bias_imu

        # Stack and apply softmax jointly to ensure sum to 1
        logits = torch.cat([lidar_logits, rgb_logits, imu_logits], dim=1)
        weights = F.softmax(logits, dim=1)

        w_lidar = weights[:, 0].unsqueeze(-1)
        w_rgb = weights[:, 1].unsqueeze(-1)
        w_imu = weights[:, 2].unsqueeze(-1)

        # Ensure weights sum to 1 (relaxed tolerance for numerical stability)
        total_weight = torch.stack([w_lidar, w_rgb, w_imu], dim=0).sum(dim=0)
        # assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=1e-5)  # Disabled for now

        return {
            'w_lidar': w_lidar,
            'w_rgb': w_rgb,
            'w_imu': w_imu,
            'attention_scores': attention_scores,
            'attention_weights': attention_weights
        }


def test_dynamic_weighting_layer():
    """Test Dynamic Weighting Layer"""
    model = DynamicWeightingLayer(feature_dim=128, num_heads=8)

    # Create test data
    batch_size = 4
    feature_dim = 128
    lidar_feat = torch.randn(batch_size, feature_dim)
    rgb_feat = torch.randn(batch_size, feature_dim)
    imu_feat = torch.randn(batch_size, feature_dim)

    # Forward pass
    output = model(lidar_feat, rgb_feat, imu_feat)

    # Debug: print actual shape
    print(f"Debug: attention_weights shape = {output['attention_weights'].shape}")

    # Verify output dimensions
    assert output['w_lidar'].shape == (batch_size, 1)
    assert output['w_rgb'].shape == (batch_size, 1)
    assert output['w_imu'].shape == (batch_size, 1)
    assert output['attention_scores'].shape == (batch_size, 3)
    # Multi-head attention returns (B, L, S) where L=sequence length (3 modalities), S=3
    assert output['attention_weights'].shape == (batch_size, 3, 3)

    # Verify weight range
    for key in ['w_lidar', 'w_rgb', 'w_imu']:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), f"{key} range error"

    # Verify weights sum to 1
    total_weight = output['w_lidar'] + output['w_rgb'] + output['w_imu']
    assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=1e-5)

    print("✅ Dynamic Weighting Layer test passed")
    for i in range(batch_size):
        print(f"Sample {i}: LiDAR={output['w_lidar'][i].item():.3f}, RGB={output['w_rgb'][i].item():.3f}, IMU={output['w_imu'][i].item():.3f}, Sum={total_weight[i].item():.3f}")


if __name__ == "__main__":
    test_dynamic_weighting_layer()
