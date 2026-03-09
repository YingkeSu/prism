"""uav_multimodal_extractor.py

Stable-Baselines3 feature extractor for UAV multi-modal observations.

Expected observation dict keys:
- lidar: (B, N, 3)
- rgb: (B, H, W, 3) or (B, 3, H, W)
- imu: (B, 6)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from networks.dynamic_weighting_layer import DynamicWeightingLayer
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator


class _LidarEncoder(nn.Module):
    """Point-cloud encoder producing a single (B, D) feature."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, lidar_points: torch.Tensor) -> torch.Tensor:
        if lidar_points.ndim != 3:
            raise ValueError(f"Expected lidar_points (B, N, 3), got shape {tuple(lidar_points.shape)}.")

        if lidar_points.shape[-1] == 3:
            x = lidar_points.transpose(1, 2)  # (B, 3, N)
        elif lidar_points.shape[1] == 3:
            x = lidar_points
        else:
            raise ValueError(
                f"Expected lidar_points last dim=3 (B,N,3) or channel dim=3 (B,3,N), got {tuple(lidar_points.shape)}."
            )

        x = self.net(x)  # (B, 128, N)
        x = x.mean(dim=-1)  # (B, 128)
        return self.proj(x)  # (B, D)


class _RgbEncoder(nn.Module):
    """Lightweight CNN producing a single (B, D) feature."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, rgb_chw: torch.Tensor) -> torch.Tensor:
        if rgb_chw.ndim != 4 or rgb_chw.shape[1] != 3:
            raise ValueError(f"Expected rgb (B, 3, H, W), got shape {tuple(rgb_chw.shape)}.")
        x = self.conv(rgb_chw)
        return self.proj(x)


class _ImuEncoder(nn.Module):
    def __init__(self, imu_dim: int, feature_dim: int):
        super().__init__()
        self.imu_dim = imu_dim
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(imu_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.ndim != 2 or imu.shape[1] != self.imu_dim:
            raise ValueError(f"Expected imu (B, {self.imu_dim}), got shape {tuple(imu.shape)}.")
        return self.net(imu)


class UAVMultimodalExtractor(BaseFeaturesExtractor):
    """
    SB3 feature extractor for dict observations from `UAVMultimodalEnv`.

    Args:
        observation_space: Gymnasium Dict space with 'lidar'/'rgb'/'imu'.
        features_dim: Output feature dimension.
        use_reliability: If True, compute modality qualities and gate fusion weights.
        num_heads: Attention heads for `DynamicWeightingLayer`.
        fixed_weights: Optional (w_lidar, w_rgb, w_imu) that sum to 1.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 256,
        use_reliability: bool = False,
        num_heads: int = 8,
        fixed_weights: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(f"Expected Dict observation_space, got {type(observation_space)}")

        for key in ("lidar", "rgb", "imu"):
            if key not in observation_space.spaces:
                raise ValueError(f"Missing key {key!r} in observation_space.")

        self.use_reliability = bool(use_reliability)
        self.fixed_weights = fixed_weights

        self.lidar_encoder = _LidarEncoder(feature_dim=features_dim)
        self.rgb_encoder = _RgbEncoder(feature_dim=features_dim)
        self.imu_encoder = _ImuEncoder(imu_dim=6, feature_dim=features_dim)

        self.dynamic_weighting = DynamicWeightingLayer(feature_dim=features_dim, num_heads=num_heads)

        if self.use_reliability:
            self.lidar_rel = LiDARSNREstimator(point_dim=3, feature_dim=64)
            self.rgb_rel = ImageQualityEstimator(input_channels=3)

    @staticmethod
    def _to_tensor(x: torch.Tensor | object) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    @staticmethod
    def _rgb_to_chw(rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4:
            raise ValueError(f"Expected rgb 4D tensor, got {tuple(rgb.shape)}")

        # (B, H, W, 3) -> (B, 3, H, W)
        if rgb.shape[-1] == 3:
            return rgb.permute(0, 3, 1, 2)
        # Already (B, 3, H, W)
        if rgb.shape[1] == 3:
            return rgb

        raise ValueError(f"Expected rgb in (B,H,W,3) or (B,3,H,W), got {tuple(rgb.shape)}")

    def _compute_fixed_weights(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        if self.fixed_weights is None:
            raise RuntimeError("fixed_weights is None")

        w_l, w_r, w_i = self.fixed_weights
        total = float(w_l + w_r + w_i)
        if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
            raise ValueError(f"fixed_weights must sum to 1; got sum={total}")

        w = torch.tensor([w_l, w_r, w_i], device=device, dtype=dtype).view(1, 3).repeat(batch_size, 1)
        return {"w_lidar": w[:, 0:1], "w_rgb": w[:, 1:2], "w_imu": w[:, 2:3]}

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        lidar = self._to_tensor(observations["lidar"]).to(dtype=torch.float32)
        rgb = self._to_tensor(observations["rgb"]).to(dtype=torch.float32)
        imu = self._to_tensor(observations["imu"]).to(dtype=torch.float32)

        if lidar.ndim != 3:
            raise ValueError(f"Expected lidar (B, N, 3), got {tuple(lidar.shape)}")
        if imu.ndim != 2:
            raise ValueError(f"Expected imu (B, 6), got {tuple(imu.shape)}")

        rgb_chw = self._rgb_to_chw(rgb)

        # Encoder input normalization: keep quality estimator on [0,255], CNN on [0,1]
        rgb_max = float(rgb_chw.detach().max().cpu()) if rgb_chw.numel() > 0 else 0.0
        if rgb_max <= 1.5:
            rgb_for_quality = rgb_chw * 255.0
            rgb_for_cnn = rgb_chw
        else:
            rgb_for_quality = rgb_chw
            rgb_for_cnn = rgb_chw / 255.0

        lidar_feat = self.lidar_encoder(lidar)
        rgb_feat = self.rgb_encoder(rgb_for_cnn)
        imu_feat = self.imu_encoder(imu)

        batch_size = int(lidar_feat.shape[0])
        device = lidar_feat.device
        dtype = lidar_feat.dtype

        if self.fixed_weights is not None:
            weights = self._compute_fixed_weights(batch_size=batch_size, device=device, dtype=dtype)
        else:
            weights = self.dynamic_weighting(lidar_feat, rgb_feat, imu_feat)

            if self.use_reliability:
                q_lidar = self.lidar_rel(lidar)["snr"]  # (B,1)
                q_rgb = self.rgb_rel(rgb_for_quality)["overall_quality"]  # (B,1)
                q_imu = torch.sigmoid(-imu.std(dim=1, keepdim=True))  # (B,1)

                gate = torch.cat([q_lidar, q_rgb, q_imu], dim=1).clamp(0.0, 1.0)  # (B,3)
                raw = torch.cat([weights["w_lidar"], weights["w_rgb"], weights["w_imu"]], dim=1)
                gated = raw * gate
                denom = gated.sum(dim=1, keepdim=True).clamp_min(1e-6)
                gated = gated / denom

                weights["w_lidar"] = gated[:, 0:1]
                weights["w_rgb"] = gated[:, 1:2]
                weights["w_imu"] = gated[:, 2:3]

        fused = (
            weights["w_lidar"] * lidar_feat
            + weights["w_rgb"] * rgb_feat
            + weights["w_imu"] * imu_feat
        )

        return fused


def test_uav_multimodal_extractor() -> None:
    """Minimal CPU-only smoke test."""
    obs_space = gym.spaces.Dict(
        {
            "lidar": gym.spaces.Box(-50.0, 50.0, shape=(1000, 3), dtype=float),
            "rgb": gym.spaces.Box(0.0, 255.0, shape=(128, 128, 3), dtype=float),
            "imu": gym.spaces.Box(-20.0, 20.0, shape=(6,), dtype=float),
        }
    )
    extractor = UAVMultimodalExtractor(obs_space, features_dim=256, use_reliability=True)

    batch_obs = {
        "lidar": torch.randn(4, 1000, 3),
        "rgb": torch.rand(4, 128, 128, 3) * 255.0,
        "imu": torch.randn(4, 6),
    }
    features = extractor(batch_obs)
    assert features.shape == (4, 256)
    print("✅ UAVMultimodalExtractor test passed")


if __name__ == "__main__":
    test_uav_multimodal_extractor()
