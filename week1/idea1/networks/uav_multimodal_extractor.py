"""
UAV Multimodal Feature Extractor

SB3-compatible custom feature extractor that integrates reliability-aware fusion.
"""

from typing import Dict, Optional, Sequence, Union, cast

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from networks.reliability_aware_fusion import ReliabilityAwareFusionModule


def _to_channel_first_rgb(rgb_image: torch.Tensor) -> torch.Tensor:
    """
    Normalize RGB tensors to (B, 3, H, W).

    SB3 may already provide channel-first tensors after preprocessing, while
    ad-hoc tests often use channel-last tensors.
    """
    rgb_image = rgb_image.float() if rgb_image.dtype != torch.float32 else rgb_image

    if rgb_image.ndim != 4:
        raise ValueError(f"Expected RGB tensor with 4 dims, got shape={tuple(rgb_image.shape)}")
    if rgb_image.shape[1] == 3:
        return rgb_image
    if rgb_image.shape[-1] == 3:
        return rgb_image.permute(0, 3, 1, 2)

    raise ValueError(
        "RGB tensor must be channel-first (B,3,H,W) or channel-last (B,H,W,3), "
        f"got shape={tuple(rgb_image.shape)}"
    )


def _resolve_rgb_hw(rgb_shape: Sequence[int]) -> tuple[int, int]:
    """Resolve RGB height/width from either HWC or CHW observation space shape."""
    if len(rgb_shape) != 3:
        raise ValueError(f"RGB shape must have 3 dims, got {rgb_shape}")

    # Channel-last: (H, W, 3)
    if rgb_shape[-1] == 3:
        return int(rgb_shape[0]), int(rgb_shape[1])

    # Channel-first: (3, H, W)
    if rgb_shape[0] == 3:
        return int(rgb_shape[1]), int(rgb_shape[2])

    raise ValueError(
        "RGB shape must be HWC (..., ..., 3) or CHW (3, ..., ...), "
        f"got {rgb_shape}"
    )


class BaselineExtractor(BaseFeaturesExtractor):
    """
    Baseline RGB Encoder for UAV Multimodal Feature Extraction

    Original UAVMultimodalExtractor implementation with 16.9M RGB encoder parameters.
    Renamed from UAVMultimodalExtractor for clarity in encoder comparison.

    Compatible with Stable-Baselines3 BaseFeaturesExtractor.
    Integrates reliability-aware fusion for multi-modal inputs.

    Args:
        observation_space: Dict observation space from gym.Env
        features_dim: Output feature dimension (default 256)
        use_reliability: Whether to use reliability-aware fusion (default True)
        num_heads: Number of attention heads (default 8)
        fixed_weights: Optional fixed weights (lidar, rgb, imu). Requires
            use_reliability=True.
        debug_mode: If True, cache intermediate results for visualization (default False)

    Input:
        observations: Dict with keys 'lidar', 'rgb', 'imu'

    Output:
        features: (B, features_dim) Fused feature vector
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        use_reliability: bool = True,
        num_heads: int = 8,
        fixed_weights: Optional[Sequence[float]] = None,
        use_reliability_modulation: bool = True,
        imu_history_len: int = 16,
        debug_mode: bool = False
    ):
        super().__init__(observation_space, features_dim)

        if observation_space is None:
            raise ValueError("observation_space is required")

        obs_space = cast(spaces.Dict, observation_space)

        if not use_reliability and fixed_weights is not None:
            raise ValueError("fixed_weights requires use_reliability=True")

        self.use_reliability = use_reliability
        self.num_heads = num_heads
        self.fixed_weights = fixed_weights
        self.use_reliability_modulation = use_reliability_modulation
        self.imu_history_len = max(1, int(imu_history_len))
        self.debug_mode = debug_mode

        # Cached intermediate results for visualization (populated when debug_mode=True)
        self.last_fusion_output = None

        # Extract modality dimensions from observation space
        lidar_shape = obs_space['lidar'].shape  # (N, 3)
        rgb_shape = obs_space['rgb'].shape  # (H, W, 3)
        imu_shape = obs_space['imu'].shape  # (6,)

        if lidar_shape is None or rgb_shape is None or imu_shape is None:
            raise ValueError("observation_space shapes must be defined")

        assert lidar_shape is not None
        assert rgb_shape is not None
        assert imu_shape is not None
        rgb_h, rgb_w = _resolve_rgb_hw(rgb_shape)

        # Base encoders (encode raw data to features)
        self.lidar_base_encoder = nn.Sequential(
            nn.Linear(lidar_shape[1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        self.rgb_base_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * (rgb_h // 4) * (rgb_w // 4), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

        self.imu_base_encoder = nn.Sequential(
            nn.Linear(imu_shape[0], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        # Reliability-aware fusion module
        if use_reliability:
            self.reliability_fusion = ReliabilityAwareFusionModule(
                feature_dim=features_dim,
                num_heads=num_heads,
                fixed_weights=fixed_weights,
                imu_window_size=self.imu_history_len,
                use_reliability_modulation=use_reliability_modulation
            )

        # Output projection
        if not use_reliability:
            # Simple concatenation when not using reliability
            self.output_projection = nn.Sequential(
                nn.Linear(64 + 256 + 6, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, features_dim)
            )
        else:
            self.output_projection = nn.Linear(64, features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            observations: Dict with keys 'lidar', 'rgb', 'imu'

        Returns:
            features: (B, features_dim) Fused feature vector
        """
        # Get modality data
        lidar_points = observations['lidar']  # (B, N, 3)
        rgb_image = observations['rgb']  # (B, H, W, 3)
        imu_data = observations['imu']  # (B, 6)

        # Normalize RGB layout to channel-first for CNN/reliability branches.
        rgb_image = _to_channel_first_rgb(rgb_image)

        # Use reliability-aware fusion or simple concatenation
        if self.use_reliability:
            # Expand IMU to a short sequence (simulated history) using zero-copy expand.
            imu_sequence = imu_data.unsqueeze(1).expand(-1, self.imu_history_len, -1)

            # Use reliability-aware fusion module
            fusion_output = self.reliability_fusion(
                lidar_points,
                rgb_image,
                imu_sequence
            )

            # Cache intermediate results for visualization when in debug mode
            if self.debug_mode:
                self.last_fusion_output = {
                    'reliability': fusion_output['reliability'],
                    'weights': fusion_output['weights']
                }

            # Project to output dimension
            features = self.output_projection(fusion_output['output'])
        else:
            # Simple concatenation (baseline)
            lidar_feat = self.lidar_base_encoder(lidar_points.mean(dim=1))  # (B, 64)
            rgb_feat = self.rgb_base_encoder(rgb_image)  # (B, 256)
            imu_feat = self.imu_base_encoder(imu_data)  # (B, 6)
            combined = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
            features = self.output_projection(combined)

            # Clear cached output when not using reliability
            if self.debug_mode:
                self.last_fusion_output = None

        return features


class GAPEncoder(BaseFeaturesExtractor):
    """
    GAP-Optimized RGB Encoder for UAV Multimodal Feature Extraction

    Uses Global Average Pooling (GAP) to reduce parameters from 16.9M to ~117K (99.3% reduction).
    Maintains (B, 256) output dimension for compatibility with fusion module.

    Compatible with Stable-Baselines3 BaseFeaturesExtractor.
    Integrates reliability-aware fusion for multi-modal inputs.

    Args:
        observation_space: Dict observation space from gym.Env
        features_dim: Output feature dimension (default 256)
        use_reliability: Whether to use reliability-aware fusion (default True)
        num_heads: Number of attention heads (default 8)
        fixed_weights: Optional fixed weights (lidar, rgb, imu). Requires
            use_reliability=True.
        debug_mode: If True, cache intermediate results for visualization (default False)

    Input:
        observations: Dict with keys 'lidar', 'rgb', 'imu'

    Output:
        features: (B, features_dim) Fused feature vector
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        use_reliability: bool = True,
        num_heads: int = 8,
        fixed_weights: Optional[Sequence[float]] = None,
        use_reliability_modulation: bool = True,
        imu_history_len: int = 16,
        debug_mode: bool = False
    ):
        super().__init__(observation_space, features_dim)

        if observation_space is None:
            raise ValueError("observation_space is required")

        obs_space = cast(spaces.Dict, observation_space)

        if not use_reliability and fixed_weights is not None:
            raise ValueError("fixed_weights requires use_reliability=True")

        self.use_reliability = use_reliability
        self.num_heads = num_heads
        self.fixed_weights = fixed_weights
        self.use_reliability_modulation = use_reliability_modulation
        self.imu_history_len = max(1, int(imu_history_len))
        self.debug_mode = debug_mode

        # Cached intermediate results for visualization (populated when debug_mode=True)
        self.last_fusion_output = None

        # Extract modality dimensions from observation space
        lidar_shape = obs_space['lidar'].shape  # (N, 3)
        rgb_shape = obs_space['rgb'].shape  # (H, W, 3)
        imu_shape = obs_space['imu'].shape  # (6,)

        if lidar_shape is None or rgb_shape is None or imu_shape is None:
            raise ValueError("observation_space shapes must be defined")

        assert lidar_shape is not None
        assert rgb_shape is not None
        assert imu_shape is not None

        # Base encoders (encode raw data to features)
        self.lidar_base_encoder = nn.Sequential(
            nn.Linear(lidar_shape[1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        # GAP-Optimized RGB encoder with AdaptiveAvgPool2d
        self.rgb_base_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (B, 3, 128, 128) -> (B, 32, 64, 64)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 32, 64, 64) -> (B, 64, 32, 32)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 64, 32, 32) -> (B, 64, 1, 1)
            nn.Flatten(),  # (B, 64, 1, 1) -> (B, 64)
            nn.Linear(64, 256),  # 64 * 256 = 16,384 params
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)  # 256 * 256 = 65,536 params
        )

        self.imu_base_encoder = nn.Sequential(
            nn.Linear(imu_shape[0], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        # Reliability-aware fusion module
        if use_reliability:
            self.reliability_fusion = ReliabilityAwareFusionModule(
                feature_dim=features_dim,
                num_heads=num_heads,
                fixed_weights=fixed_weights,
                imu_window_size=self.imu_history_len,
                use_reliability_modulation=use_reliability_modulation
            )

        # Output projection
        if not use_reliability:
            # Simple concatenation when not using reliability
            self.output_projection = nn.Sequential(
                nn.Linear(64 + 256 + 6, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, features_dim)
            )
        else:
            self.output_projection = nn.Linear(64, features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            observations: Dict with keys 'lidar', 'rgb', 'imu'

        Returns:
            features: (B, features_dim) Fused feature vector
        """
        # Get modality data
        lidar_points = observations['lidar']  # (B, N, 3)
        rgb_image = observations['rgb']  # (B, H, W, 3)
        imu_data = observations['imu']  # (B, 6)

        # Normalize RGB layout to channel-first for CNN/reliability branches.
        rgb_image = _to_channel_first_rgb(rgb_image)

        # Use reliability-aware fusion or simple concatenation
        if self.use_reliability:
            # Expand IMU to a short sequence (simulated history) using zero-copy expand.
            imu_sequence = imu_data.unsqueeze(1).expand(-1, self.imu_history_len, -1)

            # Use reliability-aware fusion module
            fusion_output = self.reliability_fusion(
                lidar_points,
                rgb_image,
                imu_sequence
            )

            # Cache intermediate results for visualization when in debug mode
            if self.debug_mode:
                self.last_fusion_output = {
                    'reliability': fusion_output['reliability'],
                    'weights': fusion_output['weights']
                }

            # Project to output dimension
            features = self.output_projection(fusion_output['output'])
        else:
            # Simple concatenation (baseline)
            lidar_feat = self.lidar_base_encoder(lidar_points.mean(dim=1))  # (B, 64)
            rgb_feat = self.rgb_base_encoder(rgb_image)  # (B, 256)
            imu_feat = self.imu_base_encoder(imu_data)  # (B, 6)
            combined = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
            features = self.output_projection(combined)

            # Clear cached output when not using reliability
            if self.debug_mode:
                self.last_fusion_output = None

        return features


class MobileNetV3Encoder(BaseFeaturesExtractor):
    """
    MobileNetV3-Optimized RGB Encoder for UAV Multimodal Feature Extraction

    Uses MobileNetV3 small backbone as feature extractor (~2.5M params for 85% reduction vs baseline).
    Maintains (B, 256) output dimension for compatibility with fusion module.

    Compatible with Stable-Baselines3 BaseFeaturesExtractor.
    Integrates reliability-aware fusion for multi-modal inputs.

    Args:
        observation_space: Dict observation space from gym.Env
        features_dim: Output feature dimension (default 256)
        use_reliability: Whether to use reliability-aware fusion (default True)
        num_heads: Number of attention heads (default 8)
        fixed_weights: Optional fixed weights (lidar, rgb, imu). Requires
            use_reliability=True.
        debug_mode: If True, cache intermediate results for visualization (default False)

    Input:
        observations: Dict with keys 'lidar', 'rgb', 'imu'

    Output:
        features: (B, features_dim) Fused feature vector
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        use_reliability: bool = True,
        num_heads: int = 8,
        fixed_weights: Optional[Sequence[float]] = None,
        use_reliability_modulation: bool = True,
        imu_history_len: int = 16,
        debug_mode: bool = False
    ):
        super().__init__(observation_space, features_dim)

        if observation_space is None:
            raise ValueError("observation_space is required")

        obs_space = cast(spaces.Dict, observation_space)

        if not use_reliability and fixed_weights is not None:
            raise ValueError("fixed_weights requires use_reliability=True")

        self.use_reliability = use_reliability
        self.num_heads = num_heads
        self.fixed_weights = fixed_weights
        self.use_reliability_modulation = use_reliability_modulation
        self.imu_history_len = max(1, int(imu_history_len))
        self.debug_mode = debug_mode

        # Cached intermediate results for visualization (populated when debug_mode=True)
        self.last_fusion_output = None

        # Extract modality dimensions from observation space
        lidar_shape = obs_space['lidar'].shape  # (N,3)
        rgb_shape = obs_space['rgb'].shape  # (H, W, 3)
        imu_shape = obs_space['imu'].shape  # (6,)

        if lidar_shape is None or rgb_shape is None or imu_shape is None:
            raise ValueError("observation_space shapes must be defined")

        assert lidar_shape is not None
        assert rgb_shape is not None
        assert imu_shape is not None

        # Import torchvision for MobileNetV3
        import torchvision.models as models

        # Base encoders (encode raw data to features)
        self.lidar_base_encoder = nn.Sequential(
            nn.Linear(lidar_shape[1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        # Keep default offline-safe behavior: do not require internet downloads.
        mobilenet = models.mobilenet_v3_small(weights=None)
        self.rgb_base_encoder = mobilenet.features  # Pretrained feature extractor (~2.5M params)
        self.rgb_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, 256),  # MobileNetV3 small output is 576 channels
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

        self.imu_base_encoder = nn.Sequential(
            nn.Linear(imu_shape[0], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        # Reliability-aware fusion module
        if use_reliability:
            self.reliability_fusion = ReliabilityAwareFusionModule(
                feature_dim=features_dim,
                num_heads=num_heads,
                fixed_weights=fixed_weights,
                imu_window_size=self.imu_history_len,
                use_reliability_modulation=use_reliability_modulation
            )

        # Output projection
        if not use_reliability:
            # Simple concatenation when not using reliability
            self.output_projection = nn.Sequential(
                nn.Linear(64 + 256 + 6, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, features_dim)
            )
        else:
            self.output_projection = nn.Linear(64, features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            observations: Dict with keys 'lidar', 'rgb', 'imu'

        Returns:
            features: (B, features_dim) Fused feature vector
        """
        # Get modality data
        lidar_points = observations['lidar']  # (B, N, 3)
        rgb_image = observations['rgb']  # (B, H, W, 3)
        imu_data = observations['imu']  # (B, 6)

        # Normalize RGB layout to channel-first for CNN/reliability branches.
        rgb_image = _to_channel_first_rgb(rgb_image)

        # Resize RGB to 224x224 for MobileNetV3 standard input size
        import torch.nn.functional as F
        rgb_image = F.interpolate(rgb_image, size=(224, 224), mode='bilinear', align_corners=False)

        # Use reliability-aware fusion or simple concatenation
        if self.use_reliability:
            # Expand IMU to a short sequence (simulated history) using zero-copy expand.
            imu_sequence = imu_data.unsqueeze(1).expand(-1, self.imu_history_len, -1)

            # Use reliability-aware fusion module
            fusion_output = self.reliability_fusion(
                lidar_points,
                rgb_image,
                imu_sequence
            )

            # Cache intermediate results for visualization when in debug mode
            if self.debug_mode:
                self.last_fusion_output = {
                    'reliability': fusion_output['reliability'],
                    'weights': fusion_output['weights']
                }

            # Project to output dimension
            features = self.output_projection(fusion_output['output'])
        else:
            # Simple concatenation (baseline)
            lidar_feat = self.lidar_base_encoder(lidar_points.mean(dim=1))  # (B, 64)
            rgb_feat = self.rgb_projection(self.rgb_base_encoder(rgb_image))  # (B, 256)
            imu_feat = self.imu_base_encoder(imu_data)  # (B, 6)
            combined = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
            features = self.output_projection(combined)

            # Clear cached output when not using reliability
            if self.debug_mode:
                self.last_fusion_output = None

        return features


def create_rgb_encoder(
    encoder_type: str,
    observation_space: gym.spaces.Dict,
    **kwargs
) -> Union[BaselineExtractor, GAPEncoder, MobileNetV3Encoder]:
    """
    Factory function to create RGB encoder based on type.

    Args:
        encoder_type: Type of encoder ('baseline', 'gap', 'mobilenet')
        observation_space: Dict observation space from gym.Env
        **kwargs: Additional arguments passed to encoder constructor, including:
            - features_dim: Output feature dimension
            - use_reliability: Whether to use reliability-aware fusion
            - num_heads: Number of attention heads
            - fixed_weights: Optional fixed weights
            - debug_mode: If True, cache intermediate results for visualization

    Returns:
        encoder: Instance of BaselineExtractor, GAPEncoder, or MobileNetV3Encoder

    Raises:
        ValueError: If encoder_type is not recognized
    """
    if encoder_type == 'baseline':
        return BaselineExtractor(observation_space, **kwargs)
    elif encoder_type == 'gap':
        return GAPEncoder(observation_space, **kwargs)
    elif encoder_type == 'mobilenet':
        return MobileNetV3Encoder(observation_space, **kwargs)
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            "Must be 'baseline', 'gap', or 'mobilenet'"
        )


class UAVMultimodalExtractor(BaselineExtractor):
    """
    Backward-compatible alias for legacy imports.

    Historical scripts/tests still import `UAVMultimodalExtractor`; the
    implementation was renamed to `BaselineExtractor`.
    """

    pass


def test_uav_multimodal_extractor():
    """Test Baseline Extractor (formerly UAVMultimodalExtractor)"""
    # Create test observation space
    obs_space = spaces.Dict({
        'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
        'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
    })

    # Test with reliability-aware fusion
    extractor = BaselineExtractor(obs_space, features_dim=256, use_reliability=True)

    # Create test observations
    batch_size = 4
    observations = {
        'lidar': torch.randn(batch_size, 1000, 3),
        'rgb': torch.randint(0, 255, (batch_size, 128, 128, 3)),
        'imu': torch.randn(batch_size, 6)
    }

    # Forward pass
    features = extractor(observations)

    print(f"Output features shape: {features.shape}")
    print(f"Expected shape: ({batch_size}, 256)")

    assert features.shape == (batch_size, 256), "Output dimension incorrect"
    print("✅ Baseline Extractor (with reliability) test passed")

    # Test without reliability-aware fusion
    extractor_baseline = BaselineExtractor(obs_space, features_dim=256, use_reliability=False)
    features_baseline = extractor_baseline(observations)

    print(f"Baseline output shape: {features_baseline.shape}")
    assert features_baseline.shape == (batch_size, 256), "Baseline output dimension incorrect"
    print("✅ Baseline Extractor (baseline mode) test passed")


if __name__ == "__main__":
    import numpy as np
    test_uav_multimodal_extractor()
