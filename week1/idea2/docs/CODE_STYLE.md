# Idea2 代码风格规范 (Code Style Guide)

**创建日期**: 2026-02-04
**项目**: PRISM - UAV Research
**基于**: Idea1代码规范 + PEP 8 + PyTorch最佳实践
**Python版本**: 3.10+

## 外部规范参考 (External Standards)

本项目的本地规范以本文档为准；同时，我们将权威外部规范的链接与项目级工程约定集中在：

- `docs/ENGINEERING_STANDARDS.md`

---

## 目录

1. [命名约定](#1-命名约定)
2. [导入规范](#2-导入规范)
3. [类型注解](#3-类型注解)
4. [文档字符串](#4-文档字符串)
5. [代码格式化](#5-代码格式化)
6. [PyTorch模块规范](#6-pytorch模块规范)
7. [Gymnasium环境规范](#7-gymnasium环境规范)
8. [测试规范](#8-测试规范)
9. [错误处理](#9-错误处理)

---

## 1. 命名约定

### 1.1 基本规则

| 类型 | 规则 | 示例 | 说明 |
|------|------|------|------|
| **类名** | `PascalCase` | `QualityPredictor`, `UAVEnv` | 首字母大写，驼峰命名 |
| **函数名** | `snake_case` | `compute_quality`, `fuse_features` | 小写，下划线分隔 |
| **变量名** | `snake_case` | `lidar_points`, `fusion_weights` | 小写，下划线分隔 |
| **常量** | `UPPER_CASE` | `MAX_STEPS`, `DEFAULT_LR` | 全大写，下划线分隔 |
| **私有方法** | `_snake_case` | `_internal_method`, `_parse_obs` | 前缀单下划线 |
| **私有类** | `_PascalCase` | `_InternalHelper` | 前缀单下划线 |

### 1.2 命名示例

```python
# ✅ 正确
class QualityPredictor(nn.Module):
    pass

def compute_fusion_weights(features: torch.Tensor) -> torch.Tensor:
    pass

lidar_points = torch.randn(4, 1000, 3)
fusion_weights = {'lidar': 0.5, 'rgb': 0.3, 'imu': 0.2}

MAX_EPISODE_STEPS = 1000
DEFAULT_LEARNING_RATE = 3e-4

def _internal_helper(data: np.ndarray) -> Dict:
    pass

# ❌ 错误
class qualityPredictor:  # 应为PascalCase
    pass

def ComputeWeights():     # 应为snake_case
    pass

LidarPoints = []          # 应为小写
```

### 1.3 特殊命名约定

```python
# 测试函数：test_ 前缀
def test_quality_predictor():
    pass

# 工厂函数：create_ 或 make_ 前缀
def create_rgb_encoder(encoder_type: str) -> nn.Module:
    pass

# 布尔变量：is_/has_/can_ 前缀
is_training = True
has_imu_data = False
can_fuse = True

# 回调函数：_on_ 前缀
def _on_training_start():
    pass

def _on_step():
    pass
```

---

## 2. 导入规范

### 2.1 导入顺序（严格遵循）

```python
# 1. 标准库导入
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 2. 第三方库导入
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import SAC

# 3. 本地模块导入（使用绝对导入）
from networks.quality_predictor import QualityPredictor
from networks.fusion_layer import FusionLayer
from envs.uav_env import UAVEnvironment
```

### 2.2 导入规范

```python
# ✅ 正确 - 每行一个导入
import os
import sys
import torch

# ✅ 正确 - 从同一模块导入多个
from typing import Dict, List, Optional, Tuple, Any

# ❌ 错误 - 一行导入多个模块
import os, sys, torch

# ✅ 正确 - 使用别名
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ❌ 错误 - 通配符导入
from networks.quality_predictor import *
```

### 2.3 绝对导入 vs 相对导入

```python
# ✅ 正确 - 使用绝对导入（从项目根目录）
from networks.quality_predictor import QualityPredictor
from envs.uav_env import UAVEnvironment

# ❌ 错误 - 避免相对导入
from ..networks.quality_predictor import QualityPredictor
from .envs.uav_env import UAVEnvironment
```

---

## 3. 类型注解

### 3.1 函数类型注解（必须）

```python
# ✅ 正确 - 完整类型注解
def forward(
    self,
    lidar_points: torch.Tensor,
    rgb_image: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    多模态特征融合

    Args:
        lidar_points: (B, N, 3) LiDAR点云
        rgb_image: (B, 3, H, W) RGB图像

    Returns:
        融合特征字典
    """
    pass

# ❌ 错误 - 缺少类型注解
def forward(self, lidar_points, rgb_image):
    pass
```

### 3.2 常用类型

```python
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import numpy as np

# PyTorch张量
torch.Tensor

# NumPy数组
np.ndarray

# 容器类型
Dict[str, torch.Tensor]          # 字典
List[int]                         # 列表
Tuple[torch.Tensor, torch.Tensor] # 元组
Optional[int]                     # 可选类型
Union[int, float]                 # 联合类型

# Gymnasium类型
obs_type = Dict[str, np.ndarray]  # 观测类型
```

### 3.3 类属性类型注解

```python
class QualityPredictor(nn.Module):
    """质量预测器"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # 类型注解（可选）
        self.input_dim: int = input_dim
        self.encoder: nn.Module = nn.Linear(input_dim, hidden_dim)
```

---

## 4. 文档字符串

### 4.1 格式（Google风格）

```python
class QualityPredictor(nn.Module):
    """
    质量预测网络

    融合多模态质量指标，预测各模态可靠性分数。

    Args:
        lidar_dim: LiDAR特征维度（默认64）
        rgb_dim: RGB特征维度（默认256）
        hidden_dim: 隐藏层维度（默认128）

    Input:
        lidar_points: (B, N, 3) LiDAR点云
        rgb_image: (B, 3, H, W) RGB图像

    Output:
        Dict: {
            'q_lidar': (B, 1),   # LiDAR质量 [0, 1]
            'q_rgb': (B, 1),     # RGB质量 [0, 1]
            'features': (B, 256) # 融合特征
        }

    Attributes:
        encoder: 特征编码器
        predictor: 质量预测器
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (B, D) 输入张量

        Returns:
            (B, output_dim) 输出张量

        Raises:
            ValueError: 如果输入形状不匹配
        """
        pass
```

### 4.2 形状标注规范

```python
# ✅ 正确
lidar_points: (B, N, 3) - B=batch_size, N=num_points

# ✅ 正确（详细说明）
Returns:
    Dict[str, Tensor]: 融合结果
        - 'features': (B, 256) 融合特征向量
        - 'weights': Dict[str, Tensor] 各模态权重
            - 'w_lidar': (B, 1) LiDAR权重 [0, 1]
```

---

## 5. 代码格式化

### 5.1 基本格式

```python
# ✅ 使用4个空格缩进
def compute_features(data):
    if data is not None:
        result = process(data)
        return result

# ✅ 运算符前后加空格
x = a + b * c

# ✅ 逗号后加空格
func(a, b, c)

# ✅ 行长度 < 100字符
long_variable_name = some_function(
    parameter1, parameter2, parameter3
)
```

### 5.2 格式化工具

```bash
# 格式化
ruff format .
# 或
black .

# Linting
ruff check .
```

---

## 6. PyTorch模块规范

### 6.1 nn.Module模板

```python
import torch
import torch.nn as nn
from typing import Dict

class MyModule(nn.Module):
    """
    模块文档字符串

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度（默认256）
        dropout: Dropout比例（默认0.1）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 层定义
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),      # inplace节省内存
            nn.LayerNorm(hidden_dim),   # LayerNorm更适合RL
            nn.Dropout(dropout)
        )

        print(f"Initialized with {self.count_parameters() / 1000:.1f}K parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (B, input_dim) 输入

        Returns:
            (B, hidden_dim) 输出
        """
        return self.encoder(x)

    def count_parameters(self) -> int:
        """统计可训练参数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 测试函数
def test_my_module():
    """测试模块"""
    model = MyModule(input_dim=64)
    x = torch.randn(4, 64)
    output = model(x)
    assert output.shape == (4, 256)
    print("✅ MyModule test passed")

if __name__ == "__main__":
    test_my_module()
```

### 6.2 常用超参数

```python
# 默认配置
FEATURE_DIM = 256
HIDDEN_DIM = 128
NUM_HEADS = 8
DROPOUT = 0.1
LEARNING_RATE = 3e-4
```

---

## 7. Gymnasium环境规范

### 7.1 环境模板

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

class UAVEnvironment(gym.Env):
    """
    UAV导航环境

    观测: Dict[str, np.ndarray]
        - 'lidar': (N, 3) 点云
        - 'rgb': (H, W, 3) RGB图像
        - 'imu': (6,) IMU数据

    动作: Box(-1, 1, (4,))
        - (vx, vy, vz, omega)

    Args:
        max_steps: 最大步数
        render_mode: 渲染模式
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        max_steps: int = 1000,
        render_mode: str | None = None
    ):
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode

        # 观测空间
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
            'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.float32),
            'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
        })

        # 动作空间
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        observation = self._get_obs()
        return observation, {}

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """环境步进"""
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        info = {}
        return observation, reward, terminated, truncated, info
```

---

## 8. 测试规范

### 8.1 测试文件命名

```python
# ✅ 正确
tests/
├── unit/
│   ├── test_quality_predictor.py
│   └── test_fusion_layer.py
├── integration/
│   ├── test_quality_aware_fusion.py
│   └── test_sb3_integration.py
└── training/
    └── train_minimal_test.py
```

### 8.2 测试函数命名

```python
# ✅ 正确 - test_ 前缀
def test_quality_predictor():
    """测试质量预测器"""
    model = QualityPredictor()
    x = torch.randn(4, 64)
    output = model(x)
    assert output['features'].shape == (4, 256)
    print("✅ Quality Predictor test passed")
```

### 8.3 测试结构

```python
def test_module():
    """测试模块"""
    # 1. Setup
    model = MyModule(param=128)
    data = torch.randn(4, 64)

    # 2. Execute
    output = model(data)

    # 3. Assert
    assert output.shape == (4, 128)
    assert torch.all(output >= 0)

    print("✅ Test passed")
```

---

## 9. 错误处理

### 9.1 异常处理

```python
# ✅ 测试代码使用assert
assert features.shape == (batch_size, 256)

# ✅ 生产代码使用try/except
try:
    result = model.forward(obs)
except RuntimeError as e:
    print(f"Forward failed: {e}")
    raise

# ❌ 避免空的except
try:
    result = model.forward(obs)
except:
    pass  # 错误！
```

### 9.2 输入验证

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    前向传播

    Args:
        x: (B, D) 输入张量

    Raises:
        TypeError: 如果输入类型错误
        ValueError: 如果输入形状错误
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected Tensor, got {type(x)}")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {x.ndim}D")

    return self.network(x)
```

---

## 附录A: 代码检查清单

提交代码前检查：

- [ ] 命名符合规范
- [ ] 类型注解完整
- [ ] 文档字符串完整
- [ ] 导入顺序正确
- [ ] 运行 `ruff format .`
- [ ] 运行 `ruff check .`
- [ ] 测试通过
- [ ] 参数量检查（<500K）

---

## 附录B: 常用代码片段

### 参数计数

```python
def count_parameters(model: nn.Module) -> int:
    """统计可训练参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 设置随机种子

```python
def set_seed(seed: int) -> None:
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

---

**文档版本**: v1.0
**最后更新**: 2026-02-04
**基于**: Idea1代码规范 + PEP 8 + PyTorch最佳实践
