# Idea1 API接口文档

**创建日期**: 2026-01-23
**项目**: PRISM - UAV Research
**创新点**: 多维度可靠性感知的自适应融合 (Idea1)
**Python版本**: 3.10+
**PyTorch版本**: >=2.3

---

## 执行摘要

本文档提供Idea1所有核心模块的API接口规范，包括类定义、方法签名、参数说明、返回值和使用示例。所有接口设计遵循PyTorch和Stable-Baselines3的最佳实践。

---

## 一、可靠性估计器模块

### 1.1 LiDARSNREstimator

**类定义**:
```python
class LiDARSNREstimator(nn.Module):
    """LiDAR点云信噪比估计器"""
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(point_dim: int = 3, feature_dim: int = 64)` | 初始化SNR估计器 |
| `forward` | `forward(lidar_points: Tensor) -> Dict[str, Tensor]` | 前向传播，估计SNR和密度 |
| `compute_traditional_metrics` | `compute_traditional_metrics(lidar_points: Tensor) -> Dict[str, Tensor]` | 计算传统SNR指标（用于验证） |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `point_dim` | int | 3 | LiDAR点云维度（x, y, z） |
| `feature_dim` | int | 64 | 特征提取网络输出维度 |
| `lidar_points` | Tensor | - | 输入点云，形状：`(B, N, 3)`<br>`B`: batch_size<br>`N`: num_points |

**返回值说明**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `snr` | Tensor | `(B, 1)` | 信噪比，范围[0, 1] |
| `density` | Tensor | `(B, 1)` | 点云密度，范围[0, 1] |
| `uniformity` | Tensor | `(B, 1)` | 点云分布均匀性，范围[0, 1] |
| `features` | Tensor | `(B, feature_dim)` | 提取的特征向量 |

**使用示例**:

```python
# 初始化
estimator = LiDARSNREstimator(point_dim=3, feature_dim=64)

# 前向传播
output = estimator(lidar_points)  # lidar_points: (B, 1000, 3)
snr = output['snr']           # (B, 1)
density = output['density']     # (B, 1)

# 计算传统指标（验证用）
traditional = estimator.compute_traditional_metrics(lidar_points)
```

---

### 1.2 ImageQualityEstimator

**类定义**:
```python
class ImageQualityEstimator(nn.Module):
    """RGB图像质量评估器"""
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(input_channels: int = 3)` | 初始化图像质量评估器 |
| `forward` | `forward(rgb_image: Tensor) -> Dict[str, Tensor]` | 前向传播，评估图像质量 |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `input_channels` | int | 3 | RGB图像通道数 |
| `rgb_image` | Tensor | - | 输入RGB图像，形状：`(B, 3, H, W)`<br>`B`: batch_size<br>`H, W`: 图像高宽 |

**返回值说明**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `sharpness` | Tensor | `(B, 1)` | 图像锐利度，范围[0, 1] |
| `contrast` | Tensor | `(B, 1)` | 图像对比度，范围[0, 1] |
| `brightness` | Tensor | `(B, 1)` | 亮度质量，范围[0, 1] |
| `texture` | Tensor | `(B, 1)` | 纹理复杂度，范围[0, 1] |
| `overall_quality` | Tensor | `(B, 1)` | 综合质量评分，范围[0, 1] |

**使用示例**:

```python
# 初始化
estimator = ImageQualityEstimator(input_channels=3)

# 前向传播
output = estimator(rgb_image)  # rgb_image: (B, 3, 128, 128)
sharpness = output['sharpness']
contrast = output['contrast']
overall_quality = output['overall_quality']
```

---

### 1.3 IMUConsistencyChecker

**类定义**:
```python
class IMUConsistencyChecker(nn.Module):
    """IMU数据一致性检查器"""
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(imu_dim: int = 6, window_size: int = 100, feature_dim: int = 64)` | 初始化IMU一致性检查器 |
| `forward` | `forward(imu_sequence: Tensor) -> Dict[str, Tensor]` | 前向传播，检查IMU一致性 |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `imu_dim` | int | 6 | IMU数据维度（acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z） |
| `window_size` | int | 100 | 滑动窗口大小 |
| `feature_dim` | int | 64 | 特征提取网络输出维度 |
| `imu_sequence` | Tensor | - | 输入IMU序列，形状：`(B, T, 6)`<br>`B`: batch_size<br>`T`: sequence_length |

**返回值说明**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `drift_score` | Tensor | `(B, 1)` | 漂移分数，范围[0, 1]，0=无漂移 |
| `velocity_anomaly` | Tensor | `(B, 1)` | 速度异常分数，范围[0, 1]，0=无异常 |
| `consistency` | Tensor | `(B, 1)` | 一致性评分，范围[0, 1]，1=完全一致 |
| `features` | Tensor | `(B, feature_dim)` | 提取的特征向量 |

**使用示例**:

```python
# 初始化
checker = IMUConsistencyChecker(imu_dim=6, window_size=100, feature_dim=64)

# 前向传播
output = checker(imu_sequence)  # imu_sequence: (B, 100, 6)
drift = output['drift_score']
anomaly = output['velocity_anomaly']
consistency = output['consistency']
```

---

## 二、可靠性预测网络

### 2.1 ReliabilityPredictor

**类定义**:
```python
class ReliabilityPredictor(nn.Module):
    """轻量级可靠性预测网络"""
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(lidar_dim: int = 64, rgb_dim: int = 256, imu_dim: int = 64, hidden_dim: int = 128, output_dim: int = 256)` | 初始化可靠性预测网络 |
| `forward` | `forward(lidar_points: Tensor, rgb_image: Tensor, imu_data: Tensor) -> Dict[str, Tensor]` | 前向传播，预测各模态可靠性 |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `lidar_dim` | int | 64 | LiDAR特征维度 |
| `rgb_dim` | int | 256 | RGB特征维度 |
| `imu_dim` | int | 64 | IMU特征维度 |
| `hidden_dim` | int | 128 | 隐藏层维度 |
| `output_dim` | int | 256 | 输出特征维度 |
| `lidar_points` | Tensor | - | LiDAR点云，形状：`(B, N, 3)` |
| `rgb_image` | Tensor | - | RGB图像，形状：`(B, 3, H, W)` |
| `imu_data` | Tensor | - | IMU序列，形状：`(B, T, 6)` |

**返回值说明**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `r_lidar` | Tensor | `(B, 1)` | LiDAR可靠性分数，范围[0, 1] |
| `r_rgb` | Tensor | `(B, 1)` | RGB可靠性分数，范围[0, 1] |
| `r_imu` | Tensor | `(B, 1)` | IMU可靠性分数，范围[0, 1] |
| `features` | Tensor | `(B, output_dim)` | 融合后的特征向量 |

**使用示例**:

```python
# 初始化
predictor = ReliabilityPredictor(
    lidar_dim=64,
    rgb_dim=256,
    imu_dim=64,
    hidden_dim=128,
    output_dim=256
)

# 前向传播
output = predictor(
    lidar_points=lidar_points,  # (B, 1000, 3)
    rgb_image=rgb_image,         # (B, 3, 128, 128)
    imu_data=imu_data            # (B, 100, 6)
)
r_lidar = output['r_lidar']
r_rgb = output['r_rgb']
r_imu = output['r_imu']
features = output['features']
```

---

## 三、动态权重分配模块

### 3.1 DynamicWeightingLayer

**类定义**:
```python
class DynamicWeightingLayer(nn.Module):
    """动态权重分配层，基于注意力机制"""
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(feature_dim: int = 256, num_heads: int = 8)` | 初始化动态权重分配层 |
| `forward` | `forward(lidar_feat: Tensor, rgb_feat: Tensor, imu_feat: Tensor, temperature: Optional[Tensor] = None) -> Dict[str, Tensor]` | 前向传播，分配动态权重 |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `feature_dim` | int | 256 | 特征维度 |
| `num_heads` | int | 8 | 多头注意力头数 |
| `lidar_feat` | Tensor | - | LiDAR特征，形状：`(B, D)` |
| `rgb_feat` | Tensor | - | RGB特征，形状：`(B, D)` |
| `imu_feat` | Tensor | - | IMU特征，形状：`(B, D)` |
| `temperature` | Optional[Tensor] | None | 可选的温度缩放参数 |

**返回值说明**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `w_lidar` | Tensor | `(B, 1)` | LiDAR权重，范围[0, 1] |
| `w_rgb` | Tensor | `(B, 1)` | RGB权重，范围[0, 1] |
| `w_imu` | Tensor | `(B, 1)` | IMU权重，范围[0, 1] |
| `attention_scores` | Tensor | `(B, 3)` | 注意力分数（原始） |
| `attention_weights` | Tensor | `(B, 8, 3)` | 多头注意力权重 |

**使用示例**:

```python
# 初始化
weighting = DynamicWeightingLayer(feature_dim=256, num_heads=8)

# 前向传播
output = weighting(
    lidar_feat=lidar_feat,  # (B, 256)
    rgb_feat=rgb_feat,      # (B, 256)
    imu_feat=imu_feat,      # (B, 256)
    temperature=None            # 使用可学习温度
)
w_lidar = output['w_lidar']
w_rgb = output['w_rgb']
w_imu = output['w_imu']

# 验证权重和为1
total = w_lidar + w_rgb + w_imu
assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
```

---

## 四、自适应归一化模块

### 4.1 AdaptiveNormalization

**类定义**:
```python
class AdaptiveNormalization(nn.Module):
    """自适应归一化层，根据可靠性分数动态调整"""
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(feature_dim: int)` | 初始化自适应归一化层 |
| `forward` | `forward(r_lidar: Tensor, r_rgb: Tensor, r_imu: Tensor, features: Dict[str, Tensor]) -> Dict[str, Tensor]` | 前向传播，归一化特征 |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `feature_dim` | int | 256 | 特征维度 |
| `r_lidar` | Tensor | - | LiDAR可靠性分数，形状：`(B, 1)` |
| `r_rgb` | Tensor | - | RGB可靠性分数，形状：`(B, 1)` |
| `r_imu` | Tensor | - | IMU可靠性分数，形状：`(B, 1)` |
| `features` | Dict[str, Tensor] | - | 原始特征字典<br>`features['lidar']`: `(B, D)`<br>`features['rgb']`: `(B, D)`<br>`features['imu']`: `(B, D)` |

**返回值说明**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `lidar_out` | Tensor | `(B, D)` | 归一化后的LiDAR特征 |
| `rgb_out` | Tensor | `(B, D)` | 归一化后的RGB特征 |
| `imu_out` | Tensor | `(B, D)` | 归一化后的IMU特征 |

**使用示例**:

```python
# 初始化
norm = AdaptiveNormalization(feature_dim=256)

# 前向传播
output = norm(
    r_lidar=r_lidar,  # (B, 1)
    r_rgb=r_rgb,      # (B, 1)
    r_imu=r_imu,      # (B, 1)
    features={
        'lidar': lidar_feat,  # (B, 256)
        'rgb': rgb_feat,      # (B, 256)
        'imu': imu_feat       # (B, 256)
    }
)
lidar_normed = output['lidar_out']
rgb_normed = output['rgb_out']
imu_normed = output['imu_out']
```

---

## 五、完整融合模块

### 5.1 ReliabilityAwareFusionModule

**类定义**:
```python
class ReliabilityAwareFusionModule(nn.Module):
    """可靠性感知融合模块，集成所有子模块"""
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(feature_dim: int = 256, num_heads: int = 8)` | 初始化完整融合模块 |
| `forward` | `forward(lidar_points: Tensor, rgb_image: Tensor, imu_data: Tensor) -> Dict[str, Any]` | 前向传播，完整融合流程 |
| `get_loss` | `get_loss(predictions: Dict[str, Any], targets: Tensor) -> Dict[str, Tensor]` | 计算损失函数 |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `feature_dim` | int | 256 | 特征维度 |
| `num_heads` | int | 8 | 多头注意力头数 |
| `lidar_points` | Tensor | - | LiDAR点云，形状：`(B, N, 3)` |
| `rgb_image` | Tensor | - | RGB图像，形状：`(B, 3, H, W)` |
| `imu_data` | Tensor | - | IMU序列，形状：`(B, T, 6)` |
| `predictions` | Dict | - | 模型输出（forward返回值） |
| `targets` | Tensor | - | 目标值，形状：`(B, action_dim)` |

**返回值说明** (forward):

| 字段 | 类型 | 说明 |
|------|------|------|
| `output` | Tensor | 融合后的输出特征，形状：`(B, output_dim)` |
| `reliability` | Dict | 可靠性分数字典<br>`{'lidar': (B, 1), 'rgb': (B, 1), 'imu': (B, 1)}` |
| `weights` | Dict | 动态权重字典<br>`{'w_lidar': (B, 1), 'w_rgb': (B, 1), 'w_imu': (B, 1)}` |
| `normed_features` | Dict | 归一化后的特征字典 |

**返回值说明** (get_loss):

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_loss` | Tensor | 总损失（标量） |
| `mse_loss` | Tensor | 主损失（MSE） |
| `reliability_reg` | Tensor | 可靠性正则化损失 |

**使用示例**:

```python
# 初始化
fusion_module = ReliabilityAwareFusionModule(feature_dim=256, num_heads=8)

# 前向传播
output = fusion_module(
    lidar_points=lidar_points,  # (B, 1000, 3)
    rgb_image=rgb_image,         # (B, 3, 128, 128)
    imu_data=imu_data            # (B, 100, 6)
)
fused_features = output['output']
reliability = output['reliability']
weights = output['weights']

# 计算损失（训练时）
targets = torch.randn(B, action_dim)
loss_dict = fusion_module.get_loss(output, targets)
total_loss = loss_dict['total_loss']
```

---

## 六、SB3集成模块

### 6.1 UAVMultimodalExtractor

**类定义**:
```python
class UAVMultimodalExtractor(BaseFeaturesExtractor):
    """自定义多模态特征提取器，SB3兼容"""
```

**继承关系**:
```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class UAVMultimodalExtractor(BaseFeaturesExtractor):
    pass
```

**方法签名**:

| 方法 | 签名 | 说明 |
|------|-------|------|
| `__init__` | `__init__(observation_space: gym.spaces.Dict, features_dim: int = 256, **kwargs)` | 初始化特征提取器 |
| `forward` | `forward(observations: Dict[str, Tensor]) -> Tensor` | 前向传播，提取融合特征 |

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `observation_space` | Dict | - | SB3观测空间<br>`observation_space['lidar']`: Box<br>`observation_space['rgb']`: Box<br>`observation_space['imu']`: Box |
| `features_dim` | int | 256 | 输出特征维度 |
| `observations` | Dict | - | 观测字典<br>`observations['lidar']`: `(B, N, 3)`<br>`observations['rgb']`: `(B, H, W, 3)`<br>`observations['imu']`: `(B, 6)` |

**返回值说明**:

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `features` | Tensor | `(B, features_dim)` | 融合后的特征向量 |

**使用示例（SB3集成）**:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.policies import MultiInputPolicy
from networks.uav_multimodal_extractor import UAVMultimodalExtractor

# 创建环境
env = UAVMultimodalEnv()  # Dict observation space

# 创建模型
model = SAC(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "use_reliability": True
        },
        "net_arch": [256, 256]
    },
    learning_rate=3e-4
)

# 训练
model.learn(total_timesteps=100000)
```

---

## 七、训练与评估接口

### 7.1 模型训练

**函数签名**:
```python
def train_reliability_fusion(
    env: gym.Env,
    total_timesteps: int = 100000,
    save_path: str = "models/reliability_aware",
    log_dir: str = "./logs/reliability_aware"
) -> None
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `env` | gym.Env | - | 训练环境 |
| `total_timesteps` | int | 100000 | 总训练步数 |
| `save_path` | str | "models/reliability_aware" | 模型保存路径 |
| `log_dir` | str | "./logs/reliability_aware" | TensorBoard日志目录 |

**使用示例**:
```python
# 训练Idea1模型
train_reliability_fusion(
    env=env,
    total_timesteps=100000,
    save_path="models/idea1_reliability_aware",
    log_dir="./logs/idea1"
)
```

### 7.2 模型评估

**函数签名**:
```python
def evaluate_model(
    model: Any,
    env: gym.Env,
    num_episodes: int = 100,
    deterministic: bool = True
) -> Dict[str, Any]
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `model` | Any | - | 训练好的模型 |
| `env` | gym.Env | - | 评估环境 |
| `num_episodes` | int | 100 | 评估回合数 |
| `deterministic` | bool | True | 是否确定性策略 |

**返回值说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `success_rate` | float | 成功率，范围[0, 1] |
| `avg_path_length` | float | 平均路径长度 |
| `collision_rate` | float | 碰撞率，范围[0, 1] |
| `avg_inference_time` | float | 平均推理时间（毫秒） |
| `episodes_results` | List | 每个回合的详细结果 |

**使用示例**:
```python
# 加载模型
model = SAC.load("models/idea1_reliability_aware.zip", env=env)

# 评估
results = evaluate_model(
    model=model,
    env=env,
    num_episodes=100,
    deterministic=True
)
print(f"Success Rate: {results['success_rate']:.2%}")
print(f"Avg Path Length: {results['avg_path_length']:.1f}")
print(f"Collision Rate: {results['collision_rate']:.2%}")
```

---

## 八、消融实验接口

### 8.1 可靠性估计器消融

**实验配置**:
```python
ABLATION_CONFIGS = {
    'no_reliability': {'use_reliability': False},
    'lidar_only': {'reliability_modality': 'lidar'},
    'rgb_only': {'reliability_modality': 'rgb'},
    'imu_only': {'reliability_modality': 'imu'},
    'full_reliability': {'use_reliability': True}
}
```

**接口**:
```python
def run_ablation_reliability_estimator(
    env: gym.Env,
    configs: Dict[str, Dict],
    total_timesteps: int = 100000
) -> Dict[str, Dict[str, Any]]
```

**使用示例**:
```python
# 运行可靠性估计器消融实验
ablation_results = run_ablation_reliability_estimator(
    env=env,
    configs=ABLATION_CONFIGS,
    total_timesteps=100000
)

# 分析结果
for config_name, results in ablation_results.items():
    print(f"{config_name}: Success Rate = {results['success_rate']:.2%}")
```

---

## 九、常量与配置

### 9.1 训练超参数

```python
TRAINING_CONFIG = {
    # 算法配置
    'algorithm': 'SAC',
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 1000000,
    'learning_starts': 1000,
    'train_freq': 1,
    'gradient_steps': 1,
    'tau': 0.005,
    'gamma': 0.99,
    'ent_coef': 'auto',
    
    # 模型配置
    'net_arch': [256, 256],
    'activation_fn': nn.ReLU,
    'use_sde': False,
    
    # 训练配置
    'total_timesteps': 100000,
    'log_interval': 1000,
    'save_interval': 10000,
    'eval_episodes': 100,
    'eval_freq': 5000,
}
```

### 9.2 可靠性配置

```python
RELIABILITY_CONFIG = {
    # LiDAR配置
    'lidar_point_dim': 3,
    'lidar_feature_dim': 64,
    'lidar_num_points': 1000,
    
    # RGB配置
    'rgb_channels': 3,
    'rgb_height': 128,
    'rgb_width': 128,
    'rgb_feature_dim': 256,
    
    # IMU配置
    'imu_dim': 6,
    'imu_window_size': 100,
    'imu_feature_dim': 64,
    
    # 融合配置
    'hidden_dim': 128,
    'output_dim': 256,
    'num_heads': 8,
    
    # 约束
    'total_params_limit': 500000,  # 500K
    'inference_time_limit': 30,       # 30ms
}
```

---

## 十、错误处理

### 10.1 输入验证

**异常类型**:
```python
class InvalidInputShapeError(Exception):
    """输入形状异常"""
    pass

class ReliabilityScoreError(Exception):
    """可靠性分数范围异常"""
    pass
```

**使用示例**:
```python
def validate_input(lidar_points, rgb_image, imu_data):
    """验证输入形状和范围"""
    B = lidar_points.shape[0]
    
    # 验证LiDAR
    if len(lidar_points.shape) != 3 or lidar_points.shape[1] != 3:
        raise InvalidInputShapeError(f"Invalid LiDAR shape: {lidar_points.shape}")
    
    # 验证RGB
    if len(rgb_image.shape) != 4 or rgb_image.shape[1] != 3:
        raise InvalidInputShapeError(f"Invalid RGB shape: {rgb_image.shape}")
    
    # 验证IMU
    if len(imu_data.shape) != 3 or imu_data.shape[2] != 6:
        raise InvalidInputShapeError(f"Invalid IMU shape: {imu_data.shape}")
    
    return True
```

---

## 十一、可视化接口

### 11.1 可靠性可视化

**函数签名**:
```python
def plot_reliability_scores(
    reliability_scores: Dict[str, Tensor],
    save_path: str = "plots/reliability_scores.png"
) -> None
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `reliability_scores` | Dict | - | 可靠性分数字典<br>`{'lidar': (B, 1), 'rgb': (B, 1), 'imu': (B, 1)}` |
| `save_path` | str | "plots/reliability_scores.png" | 保存路径 |

### 11.2 权重演化可视化

**函数签名**:
```python
def plot_weight_evolution(
    weight_history: List[Dict[str, Tensor]],
    save_path: str = "plots/weight_evolution.png"
) -> None
```

**使用示例**:
```python
# 可视化可靠性分数
plot_reliability_scores(
    reliability_scores={
        'lidar': r_lidar,
        'rgb': r_rgb,
        'imu': r_imu
    },
    save_path="plots/reliability_scores.png"
)
```

---

## 十二、快速参考

### 12.1 模块导入

```python
# 完整导入示例
from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.imu_consistency_checker import IMUConsistencyChecker
from networks.reliability_predictor import ReliabilityPredictor
from networks.dynamic_weighting_layer import DynamicWeightingLayer
from networks.adaptive_normalization import AdaptiveNormalization
from networks.reliability_aware_fusion import ReliabilityAwareFusionModule
from networks.uav_multimodal_extractor import UAVMultimodalExtractor
```

### 12.2 端到端流程

```python
# 完整训练流程
def train_idea1_pipeline():
    """Idea1完整训练流程"""
    
    # 1. 初始化模块
    fusion_module = ReliabilityAwareFusionModule(
        feature_dim=256,
        num_heads=8
    )
    
    # 2. 创建环境
    env = UAVMultimodalEnv()
    
    # 3. 创建SB3模型
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": UAVMultimodalExtractor,
            "features_extractor_kwargs": {
                "use_reliability": True
            },
            "net_arch": [256, 256]
        },
        learning_rate=3e-4,
        tensorboard_log="./logs/idea1"
    )
    
    # 4. 训练
    model.learn(total_timesteps=100000)
    
    # 5. 评估
    results = evaluate_model(model, env, num_episodes=100)
    
    return results
```

---

**文档版本**: v1.0
**创建时间**: 2026-01-23 01:30:00
**最后更新**: 2026-01-23 01:30:00
**状态**: 完成
