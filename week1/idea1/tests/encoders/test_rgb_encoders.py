"""
Test Suite for RGB Encoder Optimization Schemes

Tests for BaselineExtractor, GAPEncoder, and MobileNetV3Encoder.
Includes parameter count verification, output dimension checks, and factory function tests.
"""

import pytest
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import encoders (will fail initially - RED phase of TDD)
from networks import BaselineExtractor, GAPEncoder, MobileNetV3Encoder
from networks.uav_multimodal_extractor import create_rgb_encoder


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def observation_space():
    """Create observation space matching UAVMultimodalEnv."""
    return spaces.Dict({
        'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
        'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
    })


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def sample_observations(batch_size):
    """Create sample observations for testing."""
    return {
        'lidar': torch.randn(batch_size, 1000, 3),
        'rgb': torch.randint(0, 255, (batch_size, 128, 128, 3)),
        'imu': torch.randn(batch_size, 6)
    }


# =============================================================================
# BaselineExtractor Tests
# =============================================================================

def test_baseline_encoder_parameter_count(observation_space):
    """Test BaselineExtractor has ~16.9M parameters (±5% tolerance)."""
    encoder = BaselineExtractor(observation_space, features_dim=256, use_reliability=False)

    # Count RGB encoder parameters specifically
    rgb_encoder_params = sum(p.numel() for p in encoder.rgb_base_encoder.parameters()) / 1e6

    # Expected: ~16.9M, tolerance: ±5%
    expected_params = 16.9
    tolerance = 0.05 * expected_params  # 5% tolerance

    assert abs(rgb_encoder_params - expected_params) <= tolerance, \
        f"RGB encoder params: {rgb_encoder_params:.3f}M, expected: {expected_params}M ±{tolerance:.3f}M"


def test_baseline_encoder_output_shape(observation_space, sample_observations, batch_size):
    """Test BaselineExtractor outputs correct shape (B, 256)."""
    encoder = BaselineExtractor(observation_space, features_dim=256, use_reliability=False)

    features = encoder(sample_observations)

    assert features.shape == (batch_size, 256), \
        f"Output shape: {features.shape}, expected: ({batch_size}, 256)"


def test_baseline_encoder_forward_pass(observation_space, sample_observations):
    """Test BaselineEncoder forward pass completes without errors."""
    encoder = BaselineExtractor(observation_space, features_dim=256, use_reliability=False)

    # Should not raise any exceptions
    features = encoder(sample_observations)

    assert features.dtype == torch.float32, f"Output dtype: {features.dtype}, expected: torch.float32"
    assert not torch.isnan(features).any(), "Output contains NaN values"
    assert not torch.isinf(features).any(), "Output contains Inf values"


# =============================================================================
# GAPEncoder Tests
# =============================================================================

def test_gap_encoder_parameter_count(observation_space):
    """Test GAPEncoder has ~102K parameters (actual: 101.8K, ±5% tolerance)."""
    encoder = GAPEncoder(observation_space, features_dim=256, use_reliability=False)

    # Count RGB encoder parameters specifically
    rgb_encoder_params = sum(p.numel() for p in encoder.rgb_base_encoder.parameters()) / 1e3

    # Expected: ~102K (actual: 101.8K), tolerance: ±5%
    expected_params = 102
    tolerance = 0.05 * expected_params  # 5% tolerance

    assert abs(rgb_encoder_params - expected_params) <= tolerance, \
        f"RGB encoder params: {rgb_encoder_params:.1f}K, expected: {expected_params}K ±{tolerance:.1f}K"


def test_gap_encoder_output_shape(observation_space, sample_observations, batch_size):
    """Test GAPEncoder outputs correct shape (B, 256)."""
    encoder = GAPEncoder(observation_space, features_dim=256, use_reliability=False)

    features = encoder(sample_observations)

    assert features.shape == (batch_size, 256), \
        f"Output shape: {features.shape}, expected: ({batch_size}, 256)"


def test_gap_encoder_forward_pass(observation_space, sample_observations):
    """Test GAPEncoder forward pass completes without errors."""
    encoder = GAPEncoder(observation_space, features_dim=256, use_reliability=False)

    # Should not raise any exceptions
    features = encoder(sample_observations)

    assert features.dtype == torch.float32, f"Output dtype: {features.dtype}, expected: torch.float32"
    assert not torch.isnan(features).any(), "Output contains NaN values"
    assert not torch.isinf(features).any(), "Output contains Inf values"


# =============================================================================
# MobileNetV3Encoder Tests
# =============================================================================

def test_mobilenet_encoder_parameter_count(observation_space):
    """Test MobileNetV3Encoder has ~1.14M parameters (features + projection)."""
    encoder = MobileNetV3Encoder(observation_space, features_dim=256, use_reliability=False)

    # Count RGB encoder parameters (features + projection)
    rgb_params = sum(p.numel() for p in encoder.rgb_base_encoder.parameters())
    rgb_proj_params = sum(p.numel() for p in encoder.rgb_projection.parameters())
    rgb_encoder_params = (rgb_params + rgb_proj_params) / 1e6

    # Expected: ~1.14M (mobilenet.features ~927K + projection ~213K)
    expected_params = 1.14
    tolerance = 0.05 * expected_params  # 5% tolerance

    assert abs(rgb_encoder_params - expected_params) <= tolerance, \
        f"RGB encoder params: {rgb_encoder_params:.3f}M, expected: {expected_params}M ±{tolerance:.3f}M"


def test_mobilenet_encoder_output_shape(observation_space, sample_observations, batch_size):
    """Test MobileNetV3Encoder outputs correct shape (B, 256)."""
    encoder = MobileNetV3Encoder(observation_space, features_dim=256, use_reliability=False)

    features = encoder(sample_observations)

    assert features.shape == (batch_size, 256), \
        f"Output shape: {features.shape}, expected: ({batch_size}, 256)"


def test_mobilenet_encoder_forward_pass(observation_space, sample_observations):
    """Test MobileNetV3Encoder forward pass completes without errors."""
    encoder = MobileNetV3Encoder(observation_space, features_dim=256, use_reliability=False)

    # Should not raise any exceptions
    features = encoder(sample_observations)

    assert features.dtype == torch.float32, f"Output dtype: {features.dtype}, expected: torch.float32"
    assert not torch.isnan(features).any(), "Output contains NaN values"
    assert not torch.isinf(features).any(), "Output contains Inf values"


# =============================================================================
# Factory Function Tests
# =============================================================================

def test_factory_function_baseline(observation_space):
    """Test factory function returns BaselineExtractor for 'baseline' type."""
    encoder = create_rgb_encoder('baseline', observation_space, features_dim=256, use_reliability=False)

    assert isinstance(encoder, BaselineExtractor), \
        f"Factory returned {type(encoder).__name__}, expected: BaselineExtractor"


def test_factory_function_gap(observation_space):
    """Test factory function returns GAPEncoder for 'gap' type."""
    encoder = create_rgb_encoder('gap', observation_space, features_dim=256, use_reliability=False)

    assert isinstance(encoder, GAPEncoder), \
        f"Factory returned {type(encoder).__name__}, expected: GAPEncoder"


def test_factory_function_mobilenet(observation_space):
    """Test factory function returns MobileNetV3Encoder for 'mobilenet' type."""
    encoder = create_rgb_encoder('mobilenet', observation_space, features_dim=256, use_reliability=False)

    assert isinstance(encoder, MobileNetV3Encoder), \
        f"Factory returned {type(encoder).__name__}, expected: MobileNetV3Encoder"


def test_factory_function_unknown_type(observation_space):
    """Test factory function raises ValueError for unknown encoder type."""
    with pytest.raises(ValueError, match="Unknown encoder type"):
        create_rgb_encoder('unknown_type', observation_space, features_dim=256, use_reliability=False)


# =============================================================================
# Consistency Tests
# =============================================================================

def test_all_encoders_same_output_shape(observation_space, sample_observations, batch_size):
    """Test all encoders produce output shape (B, 256)."""
    baseline = BaselineExtractor(observation_space, features_dim=256, use_reliability=False)
    gap = GAPEncoder(observation_space, features_dim=256, use_reliability=False)
    mobilenet = MobileNetV3Encoder(observation_space, features_dim=256, use_reliability=False)

    baseline_out = baseline(sample_observations)
    gap_out = gap(sample_observations)
    mobilenet_out = mobilenet(sample_observations)

    assert baseline_out.shape == (batch_size, 256)
    assert gap_out.shape == (batch_size, 256)
    assert mobilenet_out.shape == (batch_size, 256)
