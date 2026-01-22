"""
RGB Image Quality Estimator

Evaluates image quality metrics:
- Sharpness (Laplacian operator)
- Contrast (local standard deviation)
- Brightness (histogram analysis)
- Texture complexity (gradient variance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ImageQualityEstimator(nn.Module):
    """
    RGB Image Quality Estimator

    Evaluates image quality metrics:
    - Sharpness (Laplacian operator)
    - Contrast (local standard deviation)
    - Brightness (histogram analysis)
    - Texture complexity (gradient variance)

    Args:
        input_channels: Input channel count (default 3: RGB)

    Input:
        rgb_image: (B, 3, H, W) RGB image

    Output:
        Dict: {
            'sharpness': (B, 1),      # Sharpness [0, 1]
            'contrast': (B, 1),       # Contrast [0, 1]
            'brightness': (B, 1),     # Brightness [0, 1]
            'texture': (B, 1),        # Texture [0, 1]
            'overall_quality': (B, 1) # Overall quality [0, 1]
        }
    """

    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels

        # Differentiable Laplacian kernel
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([
                [[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]]
            ], dtype=torch.float32).view(1, 1, 3, 3)
        )

        # Quality fusion network
        self.quality_fusion = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Feature extraction network (for encoding images to feature vectors)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * (128 // 4) * (128 // 4), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )

    def forward(self, rgb_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            rgb_image: (B, 3, H, W) RGB image

        Returns:
            Dict: Quality metrics
        """
        # 1. Convert to grayscale
        gray = 0.299 * rgb_image[:, 0:1, :, :] + \
                0.587 * rgb_image[:, 1:2, :, :] + \
                0.114 * rgb_image[:, 2:3, :, :]  # (B, 1, H, W)

        # 2. Sharpness (Laplacian operator)
        sharpness = self._compute_sharpness(gray)  # (B, 1)

        # 3. Contrast (local standard deviation)
        contrast = self._compute_contrast(gray)  # (B, 1)

        # 4. Brightness (histogram analysis)
        brightness = self._compute_brightness(rgb_image)  # (B, 1)

        # 5. Texture (gradient variance)
        texture = self._compute_texture(gray)  # (B, 1)

        # 6. Overall quality score
        overall_quality = self.quality_fusion(quality_features)  # (B, 1)

        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'texture': texture,
            'overall_quality': overall_quality
        }

    def _compute_sharpness(self, gray_image: torch.Tensor) -> torch.Tensor:
        """
        Compute sharpness using Laplacian operator

        Args:
            gray_image: (B, 1, H, W)

        Returns:
            sharpness: (B, 1) [0, 1]
        """
        # Apply Laplacian convolution
        laplacian = F.conv2d(gray_image, self.laplacian_kernel, padding=1)

        # Sharpness = variance (higher edge response = sharper)
        sharpness = torch.var(laplacian, dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
        sharpness = sharpness.squeeze(-1).squeeze(-1)  # (B, 1)

        # Normalize to [0, 1]
        sharpness = torch.sigmoid(sharpness * 0.1)

        return sharpness

    def _compute_contrast(self, gray_image: torch.Tensor) -> torch.Tensor:
        """
        Compute contrast using local standard deviation

        Args:
            gray_image: (B, 1, H, W)

        Returns:
            contrast: (B, 1) [0, 1]
        """
        # Use 7x7 local window
        kernel_size = 7
        padding = kernel_size // 2

        # Pad boundaries
        padded = F.pad(gray_image, (padding, padding, padding, padding), mode='reflect')

        # Sliding window compute standard deviation
        patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        # patches: (B, 1, H', W', 7, 7)

        local_std = patches.std(dim=(-2, -1))  # (B, 1, H', W')

        # Global contrast
        contrast = local_std.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
        contrast = contrast.squeeze(-1).squeeze(-1)  # (B, 1)

        # Normalize to [0, 1]
        contrast = torch.sigmoid(contrast * 0.5)

        return contrast

    def _compute_brightness(self, rgb_image: torch.Tensor) -> torch.Tensor:
        mean_brightness = rgb_image.mean(dim=[2, 3], keepdim=True) / 255.0
        mean_brightness = mean_brightness.squeeze(-1).squeeze(-1)
        mean_brightness = mean_brightness.mean(dim=1, keepdim=True)

        ideal_brightness = 0.5
        brightness_quality = 1.0 - torch.abs(mean_brightness - ideal_brightness)

        return brightness_quality

    def _compute_texture(self, gray_image: torch.Tensor) -> torch.Tensor:
        """
        Compute texture complexity (gradient variance)

        Args:
            gray_image: (B, 1, H, W)

        Returns:
            texture: (B, 1) [0, 1]
        """
        # Compute gradient
        grad_x = torch.diff(gray_image, dim=-1)  # (B, 1, H, W-1)
        grad_y = torch.diff(gray_image, dim=-2)  # (B, 1, H-1, W)

        # Gradient variance
        grad_var = torch.var(grad_x, dim=[2, 3], keepdim=True) + \
                   torch.var(grad_y, dim=[2, 3], keepdim=True)
        grad_var = grad_var.squeeze(-1).squeeze(-1)  # (B, 1)

        # Texture complexity is best when moderate
        texture_quality = torch.sigmoid(-torch.abs(grad_var - 0.5) * 5.0)

        return texture_quality


def test_image_quality_estimator():
    """Test Image Quality Estimator"""
    model = ImageQualityEstimator(input_channels=3)

    # Create test images
    batch_size = 4
    height, width = 128, 128
    rgb_image = torch.rand(batch_size, 3, height, width) * 255

    # Simulate different quality
    # Clear image
    rgb_image[0] = torch.rand(3, height, width) * 255
    # Blurred image (low-pass filter)
    rgb_image[1] = F.avg_pool2d(rgb_image[1:2], kernel_size=5, stride=1, padding=2)
    rgb_image[1] = F.interpolate(rgb_image[1].unsqueeze(0), size=(height, width), mode='bilinear').squeeze(0)
    # Low contrast
    rgb_image[2] = rgb_image[2] * 0.3 + 128
    # Dark image
    rgb_image[3] = rgb_image[3] * 0.2

    # Forward pass
    output = model(rgb_image)

    # Verify output dimensions
    assert output['sharpness'].shape == (batch_size, 1)
    assert output['contrast'].shape == (batch_size, 1)
    assert output['brightness'].shape == (batch_size, 1)
    assert output['texture'].shape == (batch_size, 1)
    assert output['overall_quality'].shape == (batch_size, 1)

    # Verify output range
    for key in output:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), f"{key} range error"

    # Clear image should have highest quality
    assert output['overall_quality'][0] > output['overall_quality'][1], "Clear image should have higher quality"
    # Blurred image should have lower sharpness
    assert output['sharpness'][0] > output['sharpness'][1], "Clear image should have higher sharpness"

    print("✅ Image Quality Estimator test passed")
    print(f"Clear image quality: {output['overall_quality'][0].item():.3f}")
    print(f"Blurred image quality: {output['overall_quality'][1].item():.3f}")
    print(f"Low contrast image: {output['overall_quality'][2].item():.3f}")
    print(f"Dark image: {output['overall_quality'][3].item():.3f}")


if __name__ == "__main__":
    test_image_quality_estimator()
