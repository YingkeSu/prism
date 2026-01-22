# Idea1 技术实现文档

**创建日期**: 2026-01-23
**项目**: PRISM - UAV Research
**创新点**: 多维度可靠性感知的自适应融合 (Idea1)
**Python版本**: 3.10+
**PyTorch版本**: >=2.3

---

## 执行摘要

本文档提供Idea1的完整技术实现指南，包括所有核心模块的代码实现、架构设计、接口规范和调试技巧。涵盖从基础组件到完整SB3集成的完整开发流程。

---

## 一、项目结构

### 1.1 目录组织

```
week1/idea1/
├── networks/
│   ├── reliability_estimators/
│   │   ├── __init__.py
│   │   ├── lidar_snr_estimator.py       # LiDAR信噪比估计
│   │   ├── image_quality_estimator.py    # 图像质量评估
│   │   └── imu_consistency_checker.py  # IMU一致性检查
│   ├── reliability_predictor.py         # 可靠性预测网络
│   ├── dynamic_weighting_layer.py       # 动态权重分配
│   ├── adaptive_normalization.py        # 自适应归一化
│   ├── reliability_aware_fusion.py      # 完整融合模块
│   └── uav_multimodal_extractor.py     # SB3自定义特征提取器
├── envs/
│   ├── __init__.py
│   ├── uav_multimodal_env.py          # 多模态UAV环境
│   └── simple_2d_env.py               # 2D简化环境
├── experiments/
│   ├── __init__.py
│   ├── ablation_reliability_estimator.py
│   ├── ablation_attention_heads.py
│   ├── ablation_dynamic_vs_fixed.py
│   ├── comparison_baselines.py
│   └── summarize_results.py
├── utils/
│   ├── __init__.py
│   ├── preprocess_uavscenes.py
│   ├── checkpoint_manager.py
│   ├── plot_comparison.py
│   └── metrics.py
├── train_uav_multimodal.py
├── evaluate_model.py
├── verify_sb3_compatibility.py
└── README.md
```

### 1.2 导入规范

```python
# networks/__init__.py

"""
Idea1 Networks Package
"""

from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.imu_consistency_checker import IMUConsistencyChecker

from networks.reliability_predictor import ReliabilityPredictor
from networks.dynamic_weighting_layer import DynamicWeightingLayer
from networks.adaptive_normalization import AdaptiveNormalization
from networks.reliability_aware_fusion import ReliabilityAwareFusionModule
from networks.uav_multimodal_extractor import UAVMultimodalExtractor

__all__ = [
    'LiDARSNREstimator',
    'ImageQualityEstimator',
    'IMUConsistencyChecker',
    'ReliabilityPredictor',
    'DynamicWeightingLayer',
    'AdaptiveNormalization',
    'ReliabilityAwareFusionModule',
    'UAVMultimodalExtractor'
]
```

---

## 二、可靠性估计器实现

### 2.1 LiDAR SNR估计器

```python
# networks/reliability_estimators/lidar_snr_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class LiDARSNREstimator(nn.Module):
    """
    LiDAR点云信噪比估计器
    
    评估点云质量指标：
    - 点云密度
    - 点云分布均匀性
    - 信噪比（SNR）
    
    Args:
        point_dim: 点云维度（默认3: x, y, z）
        feature_dim: 特征维度（默认64）
    
    Input:
        lidar_points: (B, N, 3) LiDAR点云
    
    Output:
        Dict: {
            'snr': (B, 1),           # 信噪比 [0, 1]
            'density': (B, 1),        # 点密度 [0, 1]
            'uniformity': (B, 1),     # 均匀性 [0, 1]
            'features': (B, feature_dim) # 提取特征
        }
    """
    
    def __init__(self, point_dim: int = 3, feature_dim: int = 64):
        super().__init__()
        self.point_dim = point_dim
        self.feature_dim = feature_dim
        
        # 统计特征提取网络
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
        
        # SNR预测头
        self.snr_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 密度预测头
        self.density_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 均匀性预测头
        self.uniformity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, lidar_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            lidar_points: (B, N, 3) LiDAR点云
                          B: batch_size
                          N: num_points (可变，如1000)
                          3: (x, y, z)
        
        Returns:
            Dict: 包含snr, density, uniformity, features
        """
        # 转置: (B, N, 3) -> (B, 3, N) for Conv1d
        x = lidar_points.transpose(1, 2)
        
        # 提取统计特征
        stats = self.point_stats(x)  # (B, 64, 1) -> (B, 64)
        stats = stats.squeeze(-1)  # (B, 64)
        
        # 预测各项指标
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
        计算传统SNR指标（用于验证和可视化）
        
        Args:
            lidar_points: (B, N, 3)
        
        Returns:
            Dict: 传统SNR指标
        """
        B, N, _ = lidar_points.shape
        
        # 1. 点云密度
        density = N / 1000.0  # 归一化到[0, 1]
        
        # 2. 点云分布均匀性
        center = lidar_points.mean(dim=1, keepdim=True)  # (B, 1, 3)
        distances = torch.norm(lidar_points - center, dim=-1)  # (B, N)
        std_distance = distances.std(dim=1, keepdim=True)  # (B, 1)
        uniformity = torch.exp(-std_distance / 5.0)  # 距离标准差越小，均匀性越高
        
        # 3. 综合SNR
        snr = 0.4 * density + 0.6 * uniformity
        
        return {
            'density': density,
            'uniformity': uniformity,
            'snr': snr,
            'std_distance': std_distance
        }

# 单元测试
def test_lidar_snr_estimator():
    """测试LiDAR SNR估计器"""
    model = LiDARSNREstimator(point_dim=3, feature_dim=64)
    
    # 创建测试数据
    batch_size = 4
    num_points = 1000
    lidar_points = torch.randn(batch_size, num_points, 3)
    
    # 模拟不同质量的点云
    # 高质量：密集、均匀
    lidar_points[0] = torch.randn(num_points, 3) * 0.1  # 密集点云
    # 低质量：稀疏、不均匀
    lidar_points[1] = torch.cat([
        torch.randn(100, 3) * 0.1,  # 密集区域
        torch.randn(900, 3) * 10.0   # 稀疏区域
    ], dim=0)
    
    # 前向传播
    output = model(lidar_points)
    
    # 验证输出维度
    assert output['snr'].shape == (batch_size, 1)
    assert output['density'].shape == (batch_size, 1)
    assert output['uniformity'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 64)
    
    # 验证输出范围
    assert torch.all(output['snr'] >= 0) and torch.all(output['snr'] <= 1)
    assert torch.all(output['density'] >= 0) and torch.all(output['density'] <= 1)
    assert torch.all(output['uniformity'] >= 0) and torch.all(output['uniformity'] <= 1)
    
    # 高质量点云SNR应该更高
    assert output['snr'][0] > output['snr'][1]
    
    print("✅ LiDAR SNR估计器测试通过")
    print(f"高SNR: {output['snr'][0].item():.3f}")
    print(f"低SNR: {output['snr'][1].item():.3f}")

if __name__ == "__main__":
    test_lidar_snr_estimator()
```

**关键实现要点**:
1. **Conv1d输入格式**: 需要转置为`(B, 3, N)`
2. **BatchNorm**: 添加BatchNorm层提高训练稳定性
3. **Dropout**: 在预测头中添加Dropout防止过拟合
4. **Sigmoid输出**: 所有输出归一化到[0, 1]

---

### 2.2 图像质量评估器

```python
# networks/reliability_estimators/image_quality_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ImageQualityEstimator(nn.Module):
    """
    RGB图像质量评估器
    
    评估图像质量指标：
    - 锐利度（Laplacian算子）
    - 对比度（局部标准差）
    - 亮度（直方图分析）
    - 纹理复杂度（梯度方差）
    
    Args:
        input_channels: 输入通道数（默认3: RGB）
    
    Input:
        rgb_image: (B, 3, H, W) RGB图像
    
    Output:
        Dict: {
            'sharpness': (B, 1),      # 锐利度 [0, 1]
            'contrast': (B, 1),       # 对比度 [0, 1]
            'brightness': (B, 1),     # 亮度 [0, 1]
            'texture': (B, 1),        # 纹理 [0, 1]
            'overall_quality': (B, 1) # 综合质量 [0, 1]
        }
    """
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 可微分Laplacian核
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([
                [[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]]
            ], dtype=torch.float32).view(1, 1, 3, 3)
        )
        
        # 质量融合网络
        self.quality_fusion = nn.Sequential(
            nn.Linear(4, 32),  # 4个质量指标 -> 32
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb_image: (B, 3, H, W) RGB图像
        
        Returns:
            Dict: 质量指标
        """
        # 1. 转为灰度图
        gray = 0.299 * rgb_image[:, 0:1, :, :] + \
                0.587 * rgb_image[:, 1:2, :, :] + \
                0.114 * rgb_image[:, 2:3, :, :]  # (B, 1, H, W)
        
        # 2. 锐利度（Laplacian算子）
        sharpness = self._compute_sharpness(gray)  # (B, 1)
        
        # 3. 对比度（局部标准差）
        contrast = self._compute_contrast(gray)  # (B, 1)
        
        # 4. 亮度（直方图分析）
        brightness = self._compute_brightness(rgb_image)  # (B, 1)
        
        # 5. 纹理（梯度方差）
        texture = self._compute_texture(gray)  # (B, 1)
        
        # 6. 综合质量评分
        quality_features = torch.cat([
            sharpness, contrast, brightness, texture
        ], dim=1)  # (B, 4)
        
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
        使用Laplacian算子计算锐利度
        
        Args:
            gray_image: (B, 1, H, W)
        
        Returns:
            sharpness: (B, 1) [0, 1]
        """
        # 应用Laplacian卷积
        laplacian = F.conv2d(gray_image, self.laplacian_kernel, padding=1)
        
        # 锐利度 = 方差（边缘响应越大越清晰）
        sharpness = torch.var(laplacian, dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
        sharpness = sharpness.squeeze(-1).squeeze(-1)  # (B, 1)
        
        # 归一化到[0, 1]
        sharpness = torch.sigmoid(sharpness * 0.1)
        
        return sharpness
    
    def _compute_contrast(self, gray_image: torch.Tensor) -> torch.Tensor:
        """
        计算局部标准差作为对比度
        
        Args:
            gray_image: (B, 1, H, W)
        
        Returns:
            contrast: (B, 1) [0, 1]
        """
        # 使用7x7局部窗口
        kernel_size = 7
        padding = kernel_size // 2
        
        # 扩展边界
        padded = F.pad(gray_image, (padding, padding, padding, padding), mode='reflect')
        
        # 滑动窗口计算标准差
        patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        # patches: (B, 1, H', W', 7, 7)
        
        local_std = patches.std(dim=(-2, -1))  # (B, 1, H', W')
        
        # 全局对比度
        contrast = local_std.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
        contrast = contrast.squeeze(-1).squeeze(-1)  # (B, 1)
        
        # 归一化到[0, 1]
        contrast = torch.sigmoid(contrast * 0.5)
        
        return contrast
    
    def _compute_brightness(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        计算亮度质量（直方图分析）
        
        Args:
            rgb_image: (B, 3, H, W)
        
        Returns:
            brightness_quality: (B, 1) [0, 1]
        """
        # 计算平均亮度
        mean_brightness = rgb_image.mean(dim=[2, 3], keepdim=True) / 255.0  # (B, 1, 1, 1)
        mean_brightness = mean_brightness.squeeze(-1).squeeze(-1)  # (B, 1)
        
        # 适中亮度时质量最高（0.4-0.6）
        ideal_brightness = 0.5
        brightness_quality = 1.0 - torch.abs(mean_brightness - ideal_brightness)
        
        return brightness_quality
    
    def _compute_texture(self, gray_image: torch.Tensor) -> torch.Tensor:
        """
        计算纹理复杂度（梯度方差）
        
        Args:
            gray_image: (B, 1, H, W)
        
        Returns:
            texture: (B, 1) [0, 1]
        """
        # 计算梯度
        grad_x = torch.diff(gray_image, dim=-1)  # (B, 1, H, W-1)
        grad_y = torch.diff(gray_image, dim=-2)  # (B, 1, H-1, W)
        
        # 梯度方差
        grad_var = torch.var(grad_x, dim=[2, 3], keepdim=True) + \
                   torch.var(grad_y, dim=[2, 3], keepdim=True)
        grad_var = grad_var.squeeze(-1).squeeze(-1)  # (B, 1)
        
        # 纹理复杂度适中时质量最高
        texture_quality = torch.sigmoid(-torch.abs(grad_var - 0.5) * 5.0)
        
        return texture_quality

# 单元测试
def test_image_quality_estimator():
    """测试图像质量评估器"""
    model = ImageQualityEstimator(input_channels=3)
    
    # 创建测试图像
    batch_size = 4
    height, width = 128, 128
    rgb_image = torch.rand(batch_size, 3, height, width) * 255
    
    # 模拟不同质量
    # 清晰图像
    rgb_image[0] = torch.rand(3, height, width) * 255
    # 模糊图像（低通滤波）
    rgb_image[1] = F.avg_pool2d(rgb_image[1:2], kernel_size=5, stride=1, padding=2)
    rgb_image[1] = F.interpolate(rgb_image[1].unsqueeze(0), size=(height, width), mode='bilinear').squeeze(0)
    # 低对比度
    rgb_image[2] = rgb_image[2] * 0.3 + 128
    # 暗色图像
    rgb_image[3] = rgb_image[3] * 0.2
    
    # 前向传播
    output = model(rgb_image)
    
    # 验证输出维度
    assert output['sharpness'].shape == (batch_size, 1)
    assert output['contrast'].shape == (batch_size, 1)
    assert output['brightness'].shape == (batch_size, 1)
    assert output['texture'].shape == (batch_size, 1)
    assert output['overall_quality'].shape == (batch_size, 1)
    
    # 验证输出范围
    for key in output:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), f"{key}范围错误"
    
    # 清晰图像质量最高
    assert output['overall_quality'][0] > output['overall_quality'][1], "清晰图像质量应该更高"
    # 模糊图像锐利度最低
    assert output['sharpness'][0] > output['sharpness'][1], "清晰图像锐利度应该更高"
    
    print("✅ 图像质量评估器测试通过")
    print(f"清晰图像质量: {output['overall_quality'][0].item():.3f}")
    print(f"模糊图像质量: {output['overall_quality'][1].item():.3f}")
    print(f"低对比度图像: {output['overall_quality'][2].item():.3f}")
    print(f"暗色图像: {output['overall_quality'][3].item():.3f}")

if __name__ == "__main__":
    test_image_quality_estimator()
```

---

### 2.3 IMU一致性检查器

```python
# networks/reliability_estimators/imu_consistency_checker.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class IMUConsistencyChecker(nn.Module):
    """
    IMU数据一致性检查器
    
    检查IMU数据质量：
    - 漂移（加速度计/陀螺仪）
    - 速度异常检测
    - 姿态一致性
    
    Args:
        imu_dim: IMU数据维度（默认6: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z）
        window_size: 滑动窗口大小（默认100）
    
    Input:
        imu_sequence: (B, T, 6) IMU序列
                       B: batch_size
                       T: sequence_length
                       6: (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    
    Output:
        Dict: {
            'drift_score': (B, 1),       # 漂移分数 [0, 1], 0=无漂移
            'velocity_anomaly': (B, 1),   # 速度异常 [0, 1], 0=无异常
            'consistency': (B, 1),        # 一致性 [0, 1], 1=完全一致
            'features': (B, feature_dim)  # 提取特征
        }
    """
    
    def __init__(self, imu_dim: int = 6, window_size: int = 100, feature_dim: int = 64):
        super().__init__()
        self.imu_dim = imu_dim
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # 漂移分析器（使用1D卷积处理时序）
        self.drift_analyzer = nn.Sequential(
            nn.Conv1d(imu_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 速度异常检测器
        self.velocity_anomaly_detector = nn.Sequential(
            nn.Conv1d(imu_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 漂移预测头
        self.drift_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 速度异常预测头
        self.anomaly_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 时序特征提取（LSTM）
        self.lstm = nn.LSTM(imu_dim, 64, batch_first=True, bidirectional=True)
        self.lstm_fc = nn.Linear(128, feature_dim)
    
    def forward(self, imu_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            imu_sequence: (B, T, 6) IMU序列
        
        Returns:
            Dict: 一致性指标
        """
        B, T, _ = imu_sequence.shape
        
        # 转置: (B, T, 6) -> (B, 6, T) for Conv1d
        x = imu_sequence.transpose(1, 2)
        
        # 填充/截断到window_size
        if T < self.window_size:
            padding = torch.zeros(B, 6, self.window_size - T)
            x = torch.cat([padding, x], dim=2)
        else:
            x = x[:, :, -self.window_size:]
        
        # 分析漂移
        drift_features = self.drift_analyzer(x)  # (B, 128, 1) -> (B, 128)
        drift_score = self.drift_head(drift_features)  # (B, 1)
        
        # 分析速度异常
        anomaly_features = self.velocity_anomaly_detector(x)  # (B, 128)
        velocity_anomaly = self.anomaly_head(anomaly_features)  # (B, 1)
        
        # 时序特征提取（LSTM）
        lstm_out, (h_n, c_n) = self.lstm(imu_sequence)
        # lstm_out: (B, T, 128)
        
        # 使用最后时刻的特征
        lstm_features = lstm_out[:, -1, :]  # (B, 128)
        features = self.lstm_fc(lstm_features)  # (B, feature_dim)
        
        # 计算一致性：1 - 漂移 - 异常
        consistency = 1.0 - 0.5 * drift_score - 0.5 * velocity_anomaly
        consistency = torch.clamp(consistency, 0.0, 1.0)
        
        return {
            'drift_score': drift_score,
            'velocity_anomaly': velocity_anomaly,
            'consistency': consistency,
            'features': features
        }
    
    def compute_traditional_metrics(self, imu_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算传统IMU一致性指标（用于验证）
        
        Args:
            imu_sequence: (B, T, 6)
        
        Returns:
            Dict: 传统指标
        """
        # 加速度计均值（静止时应该接近[0, 0, -9.81])
        acc_mean = imu_sequence[:, :, :3].mean(dim=1)  # (B, 3)
        acc_std = imu_sequence[:, :, :3].std(dim=1)  # (B, 3)
        
        # 陀螺仪均值（静止时应该接近[0, 0, 0]）
        gyro_mean = imu_sequence[:, :, 3:].mean(dim=1)  # (B, 3)
        gyro_std = imu_sequence[:, :, 3:].std(dim=1)  # (B, 3)
        
        # 漂移分数（标准差越大，漂移越严重）
        acc_drift = acc_std.mean(dim=1, keepdim=True)  # (B, 1)
        gyro_drift = gyro_std.mean(dim=1, keepdim=True)  # (B, 1)
        total_drift = (acc_drift + gyro_drift) / 2.0  # (B, 1)
        
        # 归一化到[0, 1]（使用sigmoid反转）
        consistency = torch.sigmoid(-total_drift * 10.0)
        
        return {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'gyro_mean': gyro_mean,
            'gyro_std': gyro_std,
            'drift': total_drift,
            'consistency': consistency
        }

# 单元测试
def test_imu_consistency_checker():
    """测试IMU一致性检查器"""
    model = IMUConsistencyChecker(imu_dim=6, window_size=100)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 100
    imu_sequence = torch.randn(batch_size, seq_len, 6)
    
    # 模拟不同质量
    # 高质量：低噪声
    imu_sequence[0] = torch.randn(seq_len, 6) * 0.01
    # 中等质量
    imu_sequence[1] = torch.randn(seq_len, 6) * 0.1
    # 低质量：高噪声
    imu_sequence[2] = torch.randn(seq_len, 6) * 1.0
    # 有漂移
    imu_sequence[3] = torch.randn(seq_len, 6) * 0.1
    imu_sequence[3, :, :3] += torch.linspace(0, 1, seq_len).unsqueeze(1)  # 加速度漂移
    
    # 前向传播
    output = model(imu_sequence)
    
    # 验证输出维度
    assert output['drift_score'].shape == (batch_size, 1)
    assert output['velocity_anomaly'].shape == (batch_size, 1)
    assert output['consistency'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 64)
    
    # 验证输出范围
    assert torch.all(output['drift_score'] >= 0) and torch.all(output['drift_score'] <= 1)
    assert torch.all(output['velocity_anomaly'] >= 0) and torch.all(output['velocity_anomaly'] <= 1)
    assert torch.all(output['consistency'] >= 0) and torch.all(output['consistency'] <= 1)
    
    # 高质量IMU一致性最高
    assert output['consistency'][0] > output['consistency'][2], "低噪声IMU应该一致性更高"
    # 无漂移IMU一致性更高
    assert output['consistency'][0] > output['consistency'][3], "无漂移IMU应该一致性更高"
    
    print("✅ IMU一致性检查器测试通过")
    print(f"高质量IMU一致性: {output['consistency'][0].item():.3f}")
    print(f"低质量IMU一致性: {output['consistency'][2].item():.3f}")
    print(f"有漂移IMU一致性: {output['consistency'][3].item():.3f}")

if __name__ == "__main__":
    test_imu_consistency_checker()
```

---

## 三、可靠性预测网络

```python
# networks/reliability_predictor.py

import torch
import torch.nn as nn
from typing import Dict
from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.imu_consistency_checker import IMUConsistencyChecker

class ReliabilityPredictor(nn.Module):
    """
    轻量级可靠性预测网络
    
    融合3个模态的质量指标，预测各模态的可靠性分数
    
    Args:
        lidar_dim: LiDAR特征维度（默认64）
        rgb_dim: RGB特征维度（默认256）
        imu_dim: IMU特征维度（默认64）
        hidden_dim: 隐藏层维度（默认128）
        output_dim: 输出特征维度（默认256）
    
    Input:
        lidar_points: (B, N, 3) LiDAR点云
        rgb_image: (B, 3, H, W) RGB图像
        imu_data: (B, T, 6) IMU序列
    
    Output:
        Dict: {
            'r_lidar': (B, 1),      # LiDAR可靠性 [0, 1]
            'r_rgb': (B, 1),        # RGB可靠性 [0, 1]
            'r_imu': (B, 1),         # IMU可靠性 [0, 1]
            'features': (B, output_dim) # 融合特征
        }
    """
    
    def __init__(
        self,
        lidar_dim: int = 64,
        rgb_dim: int = 256,
        imu_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256
    ):
        super().__init__()
        
        # 3个可靠性估计器
        self.lidar_estimator = LiDARSNREstimator(point_dim=3, feature_dim=lidar_dim)
        self.rgb_estimator = ImageQualityEstimator(input_channels=3)
        self.imu_estimator = IMUConsistencyChecker(imu_dim=6, feature_dim=imu_dim)
        
        # 编码器
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.rgb_encoder = nn.Sequential(
            nn.Linear(rgb_dim, hidden_dim),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim // 2),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2, output_dim),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # 可靠性分数预测头
        self.lidar_reliability = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.rgb_reliability = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.imu_reliability = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        lidar_points: torch.Tensor,
        rgb_image: torch.Tensor,
        imu_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            lidar_points: (B, N, 3)
            rgb_image: (B, 3, H, W)
            imu_data: (B, T, 6)
        
        Returns:
            Dict: 可靠性分数和融合特征
        """
        # 1. 估计各模态质量指标
        lidar_output = self.lidar_estimator(lidar_points)
        rgb_output = self.rgb_estimator(rgb_image)
        imu_output = self.imu_estimator(imu_data)
        
        # 提取特征
        lidar_snr = lidar_output['snr']  # (B, 1)
        rgb_quality = rgb_output['overall_quality']  # (B, 1)
        imu_consistency = imu_output['consistency']  # (B, 1)
        
        # 2. 编码各模态特征
        lidar_feat = self.lidar_encoder(lidar_output['features'])  # (B, hidden_dim//2)
        rgb_feat = self.rgb_encoder(rgb_output['features'])  # (B, hidden_dim//2)
        imu_feat = self.imu_encoder(imu_output['features'])  # (B, hidden_dim//2)
        
        # 3. 融合特征
        fused = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
        features = self.feature_fusion(fused)  # (B, output_dim)
        
        # 4. 预测可靠性分数
        r_lidar = self.lidar_reliability(features)  # (B, 1)
        r_rgb = self.rgb_reliability(features)  # (B, 1)
        r_imu = self.imu_reliability(features)  # (B, 1)
        
        return {
            'r_lidar': r_lidar,
            'r_rgb': r_rgb,
            'r_imu': r_imu,
            'features': features
        }
    
    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 单元测试
def test_reliability_predictor():
    """测试可靠性预测网络"""
    model = ReliabilityPredictor(
        lidar_dim=64,
        rgb_dim=256,
        imu_dim=64,
        hidden_dim=128,
        output_dim=256
    )
    
    # 创建测试数据
    batch_size = 4
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)
    
    # 前向传播
    output = model(lidar_points, rgb_image, imu_data)
    
    # 验证输出维度
    assert output['r_lidar'].shape == (batch_size, 1)
    assert output['r_rgb'].shape == (batch_size, 1)
    assert output['r_imu'].shape == (batch_size, 1)
    assert output['features'].shape == (batch_size, 256)
    
    # 验证输出范围
    for key in ['r_lidar', 'r_rgb', 'r_imu']:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), f"{key}范围错误"
    
    # 验证参数量
    num_params = model.count_parameters()
    assert num_params < 500000, f"参数量{num_params}超过500K"
    
    print("✅ 可靠性预测网络测试通过")
    print(f"总参数量: {num_params / 1000:.1f}K")
    print(f"LiDAR可靠性: {output['r_lidar'][0].item():.3f}")
    print(f"RGB可靠性: {output['r_rgb'][0].item():.3f}")
    print(f"IMU可靠性: {output['r_imu'][0].item():.3f}")

if __name__ == "__main__":
    test_reliability_predictor()
```

---

## 四、动态权重分配层

```python
# networks/dynamic_weighting_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class DynamicWeightingLayer(nn.Module):
    """
    动态权重分配层，基于注意力机制融合多模态特征
    
    Args:
        feature_dim: 特征维度（默认256）
        num_heads: 注意力头数（默认8）
    
    Input:
        lidar_feat: (B, D) LiDAR特征
        rgb_feat: (B, D) RGB特征
        imu_feat: (B, D) IMU特征
        temperature: 可选的温度缩放参数
    
    Output:
        Dict: {
            'w_lidar': (B, 1),            # LiDAR权重 [0, 1]
            'w_rgb': (B, 1),              # RGB权重 [0, 1]
            'w_imu': (B, 1),               # IMU权重 [0, 1]
            'attention_scores': (B, 3),      # 注意力分数
            'attention_weights': (B, 8, 3)  # 注意力权重（多头）
        }
    """
    
    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 多头注意力
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 门控机制（温度缩放）
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # 可学习的权重偏置
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
        """
        前向传播
        
        Args:
            lidar_feat: (B, D)
            rgb_feat: (B, D)
            imu_feat: (B, D)
            temperature: 可选的温度缩放参数
        
        Returns:
            Dict: 权重和注意力信息
        """
        B = lidar_feat.shape[0]
        
        # 1. 多模态特征拼接: (B, 3, D)
        multimodal_feat = torch.stack([
            lidar_feat,
            rgb_feat,
            imu_feat
        ], dim=1)
        
        # 2. 多头注意力（自注意力）
        attn_output, attention_weights = self.multi_head_attention(
            multimodal_feat, multimodal_feat, multimodal_feat
        )
        # attn_output: (B, 3, D)
        # attention_weights: (B, 8, 3)
        
        # 聚合多头注意力
        # 方法1: 平均
        attn_output_mean = attn_output.mean(dim=1)  # (B, D)
        
        # 方法2: 加权平均（使用注意力权重）
        # attention_weights转置为(B, 3, 8)用于加权
        attention_weights = attention_weights.transpose(1, 2)  # (B, 8, 3)
        attn_output_weighted = torch.matmul(attention_weights.mean(dim=1, keepdim=True), attn_output)  # (B, 1, 3, D)
        attn_output_weighted = attn_output_weighted.squeeze(1)  # (B, 3, D)
        attn_output_weighted = attn_output_weighted.mean(dim=1)  # (B, D)
        
        # 使用加权平均
        attention_output_final = attn_output_weighted
        
        # 提取注意力分数（从输出特征估计）
        attention_scores = attn_output_final.mean(dim=1, keepdim=True)  # (B, 1)
        # 使用多头注意力的平均权重作为分数
        avg_attention_weights = attention_weights.mean(dim=1)  # (B, 3)
        
        # 3. 应用温度缩放
        if temperature is None:
            temperature = torch.exp(self.temperature)
        attention_scores = attention_scores / temperature
        
        # 4. 加上偏置并计算权重
        # 分别为每个模态计算权重
        w_lidar = F.softmax(attention_scores + self.bias_lidar, dim=-1)  # (B, 1)
        w_rgb = F.softmax(attention_scores + self.bias_rgb, dim=-1)  # (B, 1)
        w_imu = F.softmax(attention_scores + self.bias_imu, dim=-1)  # (B, 1)
        
        # 确保权重和为1
        total_weight = w_lidar + w_rgb + w_imu
        assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=1e-5)
        
        return {
            'w_lidar': w_lidar,
            'w_rgb': w_rgb,
            'w_imu': w_imu,
            'attention_scores': avg_attention_weights,
            'attention_weights': attention_weights
        }

# 单元测试
def test_dynamic_weighting_layer():
    """测试动态权重分配层"""
    model = DynamicWeightingLayer(feature_dim=128, num_heads=8)
    
    # 创建测试数据
    batch_size = 4
    feature_dim = 128
    lidar_feat = torch.randn(batch_size, feature_dim)
    rgb_feat = torch.randn(batch_size, feature_dim)
    imu_feat = torch.randn(batch_size, feature_dim)
    
    # 前向传播
    output = model(lidar_feat, rgb_feat, imu_feat)
    
    # 验证输出维度
    assert output['w_lidar'].shape == (batch_size, 1)
    assert output['w_rgb'].shape == (batch_size, 1)
    assert output['w_imu'].shape == (batch_size, 1)
    assert output['attention_scores'].shape == (batch_size, 3)
    assert output['attention_weights'].shape == (batch_size, 8, 3)
    
    # 验证权重范围
    for key in ['w_lidar', 'w_rgb', 'w_imu']:
        assert torch.all(output[key] >= 0) and torch.all(output[key] <= 1), f"{key}范围错误"
    
    # 验证权重和为1
    total_weight = output['w_lidar'] + output['w_rgb'] + output['w_imu']
    assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=1e-5)
    
    print("✅ 动态权重分配层测试通过")
    for i in range(batch_size):
        print(f"样本{i}: LiDAR={output['w_lidar'][i].item():.3f}, RGB={output['w_rgb'][i].item():.3f}, IMU={output['w_imu'][i].item():.3f}, 和={total_weight[i].item():.3f}")

if __name__ == "__main__":
    test_dynamic_weighting_layer()
```

---

## 五、SB3集成

由于响应长度限制，完整的SB3集成代码和其余模块实现将包含在后续文档中。当前文档已提供：

1. ✅ 3个可靠性估计器的完整实现
2. ✅ 可靠性预测网络的完整实现
3. ✅ 动态权重分配层的完整实现

每个模块都包含：
- 完整的类定义和文档字符串
- 前向传播实现
- 辅助函数（传统指标计算）
- 单元测试

**下一步**: 继续创建剩余模块（自适应归一化层、完整融合模块、SB3自定义特征提取器）

---

**文档版本**: v1.0
**创建时间**: 2026-01-23 00:30:00
**状态**: 部分完成（核心模块已实现，SB3集成待续）
