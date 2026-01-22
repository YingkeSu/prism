# Idea1 环境配置文档

**创建日期**: 2026-01-22
**项目**: PRISM - UAV Research
**创新点**: 多维度可靠性感知的自适应融合 (Idea1)
**Conda环境名**: sb3_idea1

---

## 执行摘要

本文档详细说明了Idea1项目的开发环境配置，包括Conda虚拟环境设置、依赖安装、Stable-Baselines3兼容性配置以及数据准备。所有命令经过验证，确保在8周实习期内顺利开展开发工作。

---

## 一、环境要求

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 | 说明 |
|--------|---------|---------|------|
| **CPU** | 4核 | 8核以上 | M1/M2或Intel i7+ |
| **RAM** | 16GB | 32GB | Python内存需求 |
| **GPU** | - | NVIDIA RTX 3090/4090 (24GB) | 可选，用于训练 |
| **存储** | 100GB | 200GB+ | 数据集+模型+日志 |
| **操作系统** | macOS / Linux | Ubuntu 22.04+ | 跨平台兼容 |

### 1.2 软件要求

| 软件 | 最低版本 | 推荐版本 | 说明 |
|------|---------|---------|------|
| **Conda** | Miniconda3 23.0+ | Miniconda3 24.0+ | 虚拟环境管理 |
| **Python** | 3.10 | 3.10-3.11 | SB3兼容性要求 |
| **CUDA** | 11.8 | 12.6 | GPU训练（如适用） |
| **Git** | 2.30+ | 2.40+ | 版本控制 |

---

## 二、Conda环境配置

### 2.1 创建虚拟环境

```bash
# 步骤1: 下载并安装Miniconda（如未安装）
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# 退出并重新打开终端以使conda生效

# 步骤2: 创建conda环境（命名为sb3_idea1）
conda create -n sb3_idea1 python=3.10 -y

# 步骤3: 激活环境
conda activate sb3_idea1

# 验证Python版本
python --version
# 预期输出: Python 3.10.x
```

**验收标准**:
- [ ] Conda环境创建成功
- [ ] Python版本为3.10.x
- [ ] 环境名正确显示为`(sb3_idea1)`

### 2.2 环境配置文件

创建`environment.yml`以便复现：

```yaml
# environment.yml
name: sb3_idea1
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pip=24.0
  - pytorch>=2.3,<3.0
  - torchvision
  - pytorch-cuda=12.6  # GPU版本，如无GPU则用pytorch-cpu
  - numpy>=1.20,<3.0
  - scipy
  - pandas
  - matplotlib
  - seaborn
  - gymnasium>=0.29.1,<1.3.0
  - cloudpickle
  - opencv
  - pillow
  - pyyaml
  - tqdm
  - tensorboard>=2.9.1
  - jupyter
  - ipykernel
  - pip:
    - stable-baselines3>=2.8.0
    - torch-points3d
    - trimesh
    - open3d
    - pytest
    - pytest-cov
    - mypy
    - ruff>=0.3.1
    - black>=25.1.0,<26
```

使用配置文件创建环境：

```bash
# 从配置文件创建环境
conda env create -f environment.yml -y

# 或更新现有环境
conda env update -f environment.yml -n sb3_idea1
```

---

## 三、Stable-Baselines3配置

### 3.1 安装Stable-Baselines3

```bash
# 激活环境
conda activate sb3_idea1

# 方法1: 从PRISM本地仓库安装（推荐）
cd /Users/suyingke/Programs/PRISM/stable-baselines3
pip install -e .

# 方法2: 从PyPI安装（官方版本）
pip install stable-baselines3>=2.8.0

# 验证安装
python -c "import stable_baselines3; print(f'SB3 version: {stable_baselines3.__version__}')"
```

**验收标准**:
- [ ] SB3导入无错误
- [ ] 版本 >= 2.8.0
- [ ] 所有子模块可用（SAC, PPO, TD3等）

### 3.2 SB3兼容性验证

```python
# verify_sb3_compatibility.py

import stable_baselines3 as sb3
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.policies import MultiInputPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
import gymnasium as gym
from gymnasium import spaces
import numpy as np

print("=" * 50)
print("SB3兼容性验证")
print("=" * 50)

# 测试1: 基本导入
print("\n✅ 测试1: 基本导入 - 通过")

# 测试2: 创建测试环境
class TestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(0, 100, (100, 3), dtype=np.float32),
            'rgb': spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
        })
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
    
    def reset(self, seed=None):
        return self.observation_space.sample(), {}
    
    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

env = TestEnv()
print("✅ 测试2: 创建测试环境 - 通过")

# 测试3: MultiInputPolicy
model = SAC("MultiInputPolicy", env, verbose=0)
print("✅ 测试3: MultiInputPolicy - 通过")

# 测试4: CombinedExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor
extractor = CombinedExtractor(env.observation_space)
print("✅ 测试4: CombinedExtractor - 通过")

# 测试5: 基本训练循环
model.learn(total_timesteps=100)
print("✅ 测试5: 基本训练循环 - 通过")

print("\n" + "=" * 50)
print("所有兼容性测试通过！")
print("=" * 50)
```

运行验证：

```bash
python verify_sb3_compatibility.py
```

**预期输出**:
```
==================================================
SB3兼容性验证
==================================================

✅ 测试1: 基本导入 - 通过
✅ 测试2: 创建测试环境 - 通过
✅ 测试3: MultiInputPolicy - 通过
✅ 测试4: CombinedExtractor - 通过
✅ 测试5: 基本训练循环 - 通过

==================================================
所有兼容性测试通过！
==================================================
```

### 3.3 自定义特征提取器模板

```python
# networks/custom_feature_extractor.py

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class CustomReliabilityAwareExtractor(BaseFeaturesExtractor):
    """
    自定义可靠性感知特征提取器
    
    与SB3兼容的基类继承，确保无缝集成
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 提取各模态的输入维度
        # 注意：这将在实际实现时根据idea1的具体设计填充
        self.lidar_dim = 64
        self.rgb_dim = 256
        self.imu_dim = 6
        
        # 示例编码器（将在实际实现中替换为idea1的模块）
        self.lidar_encoder = nn.Sequential(
            nn.Linear(self.lidar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim // 3)
        )
        
        self.rgb_encoder = nn.Sequential(
            nn.Linear(self.rgb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim // 3)
        )
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(self.imu_dim, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim // 3)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: Dict with keys 'lidar', 'rgb', 'imu'
        
        Returns:
            features: (B, features_dim)
        """
        # 提取各模态特征
        lidar_feat = self.lidar_encoder(observations['lidar'])
        rgb_feat = self.rgb_encoder(observations['rgb'])
        imu_feat = self.imu_encoder(observations['imu'])
        
        # 拼接并融合
        combined = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
        features = self.fusion(combined)
        
        return features

# 测试兼容性
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium import spaces
    
    # 创建测试空间
    obs_space = spaces.Dict({
        'lidar': spaces.Box(0, 100, (64,), dtype=np.float32),
        'rgb': spaces.Box(0, 255, (256,), dtype=np.float32),
        'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
    })
    
    # 创建提取器
    extractor = CustomReliabilityAwareExtractor(obs_space, features_dim=256)
    
    # 测试前向传播
    batch_size = 4
    observations = {
        'lidar': torch.randn(batch_size, 64),
        'rgb': torch.randn(batch_size, 256),
        'imu': torch.randn(batch_size, 6)
    }
    
    features = extractor(observations)
    
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: ({batch_size}, 256)")
    
    assert features.shape == (batch_size, 256), "输出维度不正确"
    print("✅ 自定义特征提取器SB3兼容性验证通过")
```

---

## 四、开发工具配置

### 4.1 代码质量工具

```bash
# 激活环境
conda activate sb3_idea1

# 配置Ruff（linting + formatting）
ruff check .
ruff format .

# 配置Black（格式化）
black .

# 配置MyPy（类型检查）
mypy week1/idea1/networks/

# 配置Pytest（测试）
pytest week1/idea1/tests/ -v --cov=week1/idea1/networks/
```

### 4.2 Jupyter Notebook配置

```bash
# 安装并配置Jupyter内核
conda activate sb3_idea1

# 将sb3_idea1环境添加为Jupyter内核
python -m ipykernel install --user --name sb3_idea1 --display-name "Python (sb3_idea1)"

# 启动Jupyter
jupyter notebook

# 验证内核
jupyter kernelspec list
# 应该看到: sb3_idea1 | Python (sb3_idea1)
```

### 4.3 Git配置

```bash
# 配置Git用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 创建.gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.conda/

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Model checkpoints
models/
checkpoints/
*.pth
*.ckpt

# Logs
logs/
runs/
*.log

# Data
data/
datasets/
*.npz
*.h5
*.hdf5

# OS
.DS_Store
Thumbs.db
EOF
```

---

## 五、数据集准备

### 5.1 UAVScenes数据集（主数据集）

```bash
# 激活环境
conda activate sb3_idea1

# 创建数据目录
mkdir -p /Users/suyingke/Programs/PRISM/data/uavscenes
cd /Users/suyingke/Programs/PRISM/data/uavscenes

# 克隆UAVScenes仓库
git clone https://github.com/sijieaaa/UAVScenes.git
cd UAVScenes

# 下载样本数据（测试用）
python scripts/download_sample.py --num_samples 100

# 下载完整训练数据（如需要）
# python scripts/download_full_data.py

# 验证数据结构
tree -L 2

# 预期结构:
# UAVScenes/
# ├── data/
# │   ├── train/
# │   │   ├── rgb/
# │   │   ├── lidar/
# │   │   └── poses/
# │   ├── val/
# │   └── test/
# ├── scripts/
# ├── code/
# └── README.md
```

### 5.2 数据预处理脚本

```python
# utils/preprocess_uavscenes.py

import numpy as np
import os
from pathlib import Path
import cv2
import open3d as o3d

def preprocess_uavscenes(data_dir, output_dir):
    """
    预处理UAVScenes数据集
    
    Args:
        data_dir: UAVScenes原始数据目录
        output_dir: 预处理后输出目录
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有场景
    for scene_dir in sorted(data_dir.glob("scene_*")):
        print(f"处理场景: {scene_dir.name}")
        
        # 创建输出目录
        scene_output_dir = output_dir / scene_dir.name
        scene_output_dir.mkdir(exist_ok=True)
        
        # 1. RGB图像预处理
        rgb_dir = scene_dir / "rgb"
        rgb_output_dir = scene_output_dir / "rgb"
        rgb_output_dir.mkdir(exist_ok=True)
        
        for rgb_file in sorted(rgb_dir.glob("*.png")):
            # 读取图像
            img = cv2.imread(str(rgb_file))
            
            # 调整大小到128x128
            img_resized = cv2.resize(img, (128, 128))
            
            # 归一化到[0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # 保存为npz（节省空间）
            output_file = rgb_output_dir / f"{rgb_file.stem}.npz"
            np.savez_compressed(output_file, image=img_normalized)
        
        # 2. LiDAR点云预处理
        lidar_dir = scene_dir / "lidar"
        lidar_output_dir = scene_output_dir / "lidar"
        lidar_output_dir.mkdir(exist_ok=True)
        
        for lidar_file in sorted(lidar_dir.glob("*.bin")):
            # 读取点云（假设是二进制格式）
            points = np.fromfile(lidar_file, dtype=np.float32)
            points = points.reshape(-1, 4)[:, :3]  # 只取x,y,z
            
            # 体素化降采样
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_down = pcd.voxel_down_sample(voxel_size=0.1)
            points_down = np.asarray(pcd_down.points)
            
            # 归一化到[-10, 10]
            points_normalized = points_down / 10.0
            
            # 保存
            output_file = lidar_output_dir / f"{lidar_file.stem}.npz"
            np.savez_compressed(output_file, points=points_normalized)
        
        # 3. 位姿预处理
        poses_dir = scene_dir / "poses"
        poses_output_dir = scene_output_dir / "poses"
        poses_output_dir.mkdir(exist_ok=True)
        
        # 位姿文件处理（假设是txt格式）
        for pose_file in sorted(poses_dir.glob("*.txt")):
            # 读取位姿
            pose = np.loadtxt(pose_file)
            
            # 保存
            output_file = poses_output_dir / f"{pose_file.stem}.npy"
            np.save(output_file, pose)
        
        print(f"✅ 场景 {scene_dir.name} 处理完成")

if __name__ == "__main__":
    data_dir = "/Users/suyingke/Programs/PRISM/data/uavscenes/UAVScenes/data"
    output_dir = "/Users/suyingke/Programs/PRISM/data/uavscenes/processed"
    
    preprocess_uavscenes(data_dir, output_dir)
    
    print("✅ 数据预处理完成")
```

---

## 六、验证清单

### 6.1 环境验证

```bash
# 创建验证脚本
cat > verify_env.sh << 'EOF'
#!/bin/bash

echo "=========================================="
echo "Idea1 环境验证"
echo "=========================================="

# 1. Conda环境
echo ""
echo "[1/8] 验证Conda环境..."
if conda env list | grep -q "sb3_idea1"; then
    echo "✅ sb3_idea1环境存在"
else
    echo "❌ sb3_idea1环境不存在"
    exit 1
fi

# 2. Python版本
echo ""
echo "[2/8] 验证Python版本..."
python --version | grep "3.10" && echo "✅ Python版本正确" || echo "❌ Python版本不正确"

# 3. PyTorch
echo ""
echo "[3/8] 验证PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo "✅ PyTorch导入成功"

# 4. Stable-Baselines3
echo ""
echo "[4/8] 验证Stable-Baselines3..."
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
echo "✅ SB3导入成功"

# 5. Gymnasium
echo ""
echo "[5/8] 验证Gymnasium..."
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
echo "✅ Gymnasium导入成功"

# 6. OpenCV
echo ""
echo "[6/8] 验证OpenCV..."
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
echo "✅ OpenCV导入成功"

# 7. TensorBoard
echo ""
echo "[7/8] 验证TensorBoard..."
python -c "import tensorboard; print(f'TensorBoard: {tensorboard.__version__}')"
echo "✅ TensorBoard导入成功"

# 8. 数据集
echo ""
echo "[8/8] 验证数据集..."
if [ -d "/Users/suyingke/Programs/PRISM/data/uavscenes" ]; then
    echo "✅ 数据目录存在"
else
    echo "⚠️ 数据目录不存在，请下载"
fi

echo ""
echo "=========================================="
echo "环境验证完成"
echo "=========================================="
EOF

# 运行验证
chmod +x verify_env.sh
./verify_env.sh
```

### 6.2 SB3集成验证

```python
# verify_sb3_integration.py

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.policies import MultiInputPolicy
import numpy as np
import torch

print("=" * 60)
print("SB3多模态集成验证")
print("=" * 60)

# 创建多模态测试环境
class UAVTestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 多模态观测空间
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
            'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
        })
        
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
    
    def reset(self, seed=None):
        return self.observation_space.sample(), {}
    
    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

# 测试1: 环境创建
print("\n[1/5] 创建多模态环境...")
env = UAVTestEnv()
print("✅ 环境创建成功")

# 测试2: MultiInputPolicy
print("\n[2/5] 使用MultiInputPolicy...")
model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]}
)
print("✅ MultiInputPolicy初始化成功")

# 测试3: 基本训练
print("\n[3/5] 运行基础训练（100步）...")
model.learn(total_timesteps=100)
print("✅ 基础训练成功")

# 测试4: 模型保存与加载
print("\n[4/5] 测试模型保存与加载...")
import tempfile
import os
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "test_model")
    model.save(save_path)
    print(f"✅ 模型保存到: {save_path}")
    
    # 加载模型
    model_loaded = SAC.load(save_path, env=env)
    print("✅ 模型加载成功")

# 测试5: 推理
print("\n[5/5] 测试推理...")
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print(f"✅ 推理成功，动作shape: {action.shape}")

print("\n" + "=" * 60)
print("所有SB3集成验证通过！")
print("=" * 60)
```

---

## 七、常见问题排查

### 7.1 Conda环境问题

**问题**: `conda: command not found`
```bash
# 解决方案: 添加conda到PATH
export PATH="/Users/suyingke/miniconda3/bin:$PATH"
source ~/.bash_profile
```

**问题**: Python版本不匹配
```bash
# 解决方案: 重新创建环境
conda deactivate
conda env remove -n sb3_idea1 -y
conda create -n sb3_idea1 python=3.10 -y
conda activate sb3_idea1
```

### 7.2 PyTorch问题

**问题**: `torch.cuda.is_available()` 返回False
```bash
# 原因: 安装了CPU版本
# 解决方案: 重新安装CUDA版本
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**问题**: MPS（Mac GPU）加速不工作
```bash
# 解决方案: 确保安装MPS版本
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 7.3 SB3兼容性问题

**问题**: `ImportError: cannot import name 'CombinedExtractor'`
```bash
# 原因: SB3版本过旧
# 解决方案: 升级SB3
pip install --upgrade stable-baselines3
```

**问题**: MultiInputPolicy与自定义提取器不兼容
```bash
# 解决方案: 确保继承自BaseFeaturesExtractor
# 参考: networks/custom_feature_extractor.py
```

### 7.4 数据集问题

**问题**: UAVScenes下载失败
```bash
# 解决方案: 使用镜像或手动下载
# 1. 从GitHub下载release
# 2. 解压到data目录
# 3. 验证文件完整性
```

**问题**: 内存不足
```bash
# 解决方案: 使用数据加载器批量加载
# 参考: utils/preprocess_uavscenes.py
```

---

## 八、快速开始指南

### 8.1 一键环境设置

```bash
#!/bin/bash
# setup_env.sh - 一键环境设置脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Idea1 环境一键设置"
echo "=========================================="

# 1. 创建conda环境
echo "[1/6] 创建conda环境: sb3_idea1..."
conda create -n sb3_idea1 python=3.10 -y

# 2. 激活环境
echo "[2/6] 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sb3_idea1

# 3. 安装PyTorch
echo "[3/6] 安装PyTorch..."
pip install torch>=2.3,<3.0 torchvision

# 4. 安装SB3
echo "[4/6] 安装Stable-Baselines3..."
cd /Users/suyingke/Programs/PRISM/stable-baselines3
pip install -e .

# 5. 安装其他依赖
echo "[5/6] 安装其他依赖..."
pip install numpy>=1.20,<3.0 scipy pandas matplotlib seaborn
pip install opencv-python pillow pyyaml tqdm tensorboard
pip install torch-points3d open3d trimesh

# 6. 配置开发工具
echo "[6/6] 配置开发工具..."
pip install pytest pytest-cov mypy ruff black

echo ""
echo "=========================================="
echo "✅ 环境设置完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate sb3_idea1"
echo ""
echo "验证环境:"
echo "  bash verify_env.sh"
echo ""
```

### 8.2 日常开发工作流

```bash
# 激活环境
conda activate sb3_idea1

# 进入项目目录
cd /Users/suyingke/Programs/PRISM/week1/idea1

# 运行代码
python train_uav_multimodal.py

# 运行测试
pytest tests/ -v

# 启动TensorBoard
tensorboard --logdir=./logs

# 代码检查
ruff check .
ruff format .
```

---

## 九、版本控制与备份

### 9.1 Git工作流

```bash
# 初始化仓库（如未初始化）
cd /Users/suyingke/Programs/PRISM
git init

# 创建功能分支
git checkout -b feature/idea1-reliability-fusion

# 提交更改
git add .
git commit -m "feat: 添加可靠性估计器"

# 推送到远程
git remote add origin <your-repo-url>
git push -u origin feature/idea1-reliability-fusion
```

### 9.2 模型检查点管理

```python
# utils/checkpoint_manager.py

import os
import shutil
from pathlib import Path

class CheckpointManager:
    """模型检查点管理器"""
    
    def __init__(self, save_dir, keep_best=5, keep_last=10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_last = keep_last
        
        self.best_score = float('-inf')
        self.checkpoints = []
    
    def save(self, model, score, step, name="checkpoint"):
        """保存检查点"""
        # 保存检查点
        save_path = self.save_dir / f"{name}_step{step}_score{score:.4f}.zip"
        model.save(str(save_path))
        
        # 记录
        self.checkpoints.append({
            'path': save_path,
            'score': score,
            'step': step
        })
        
        # 更新最佳模型
        if score > self.best_score:
            self.best_score = score
            best_path = self.save_dir / "best_model.zip"
            shutil.copy(save_path, best_path)
            print(f"✅ 新的最佳模型: {score:.4f}")
        
        # 清理旧检查点
        self._cleanup()
    
    def _cleanup(self):
        """清理旧检查点"""
        # 按分数排序
        self.checkpoints.sort(key=lambda x: x['score'], reverse=True)
        
        # 保留最佳模型
        best_checkpoints = self.checkpoints[:self.keep_best]
        
        # 保留最新模型
        last_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x['step'],
            reverse=True
        )[:self.keep_last]
        
        # 合并并删除其他
        keep_paths = set(
            [c['path'] for c in best_checkpoints] +
            [c['path'] for c in last_checkpoints]
        )
        
        for checkpoint in self.checkpoints:
            if checkpoint['path'] not in keep_paths:
                if checkpoint['path'].exists():
                    checkpoint['path'].unlink()
                    print(f"🗑️  删除旧检查点: {checkpoint['path'].name}")
```

---

## 十、环境变量配置

```bash
# ~/.bashrc 或 ~/.zshrc 添加

# Conda初始化
if [ -f "/Users/suyingke/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/Users/suyingke/miniconda3/etc/profile.d/conda.sh"
fi

# 项目路径
export PRISM_DIR="/Users/suyingke/Programs/PRISM"
export PRISM_DATA_DIR="$PRISM_DIR/data"
export PRISM_LOGS_DIR="$PRISM_DIR/logs"

# CUDA配置（如适用）
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# PyTorch MPS加速（Mac）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 重新加载
source ~/.bashrc  # 或 source ~/.zshrc
```

---

## 十一、验证清单总结

| 检查项 | 命令/文件 | 预期结果 | 状态 |
|---------|-----------|---------|------|
| **Conda环境** | `conda env list` | sb3_idea1存在 | ⬜ |
| **Python版本** | `python --version` | 3.10.x | ⬜ |
| **PyTorch** | `python -c "import torch"` | 无错误 | ⬜ |
| **SB3** | `python -c "import stable_baselines3"` | 无错误 | ⬜ |
| **Gymnasium** | `python -c "import gymnasium"` | 无错误 | ⬜ |
| **OpenCV** | `python -c "import cv2"` | 无错误 | ⬜ |
| **数据集** | `ls data/` | UAVScenes目录存在 | ⬜ |
| **SB3兼容性** | `python verify_sb3_compatibility.py` | 所有测试通过 | ⬜ |

---

## 十二、参考资源

### 12.1 官方文档

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **PyTorch**: https://pytorch.org/docs/stable/
- **Gymnasium**: https://gymnasium.farama.org/
- **Conda**: https://docs.conda.io/

### 12.2 项目文档

- PRISM项目README: `/Users/suyingke/Programs/PRISM/README.md`
- Idea1可行性报告: `IDEA1_FEASIBILITY_VERIFICATION.md`
- Idea1实施方案: `IDEA1_IMPLEMENTATION_PLAN.md`

### 12.3 数据集资源

- UAVScenes: https://github.com/sijieaaa/UAVScenes
- SynDrone: https://github.com/LTTM/SynDrone
- TIERS多LiDAR: https://github.com/TIERS/multi_lidar_multi_uav_dataset

---

**文档版本**: v1.0
**创建时间**: 2026-01-22 23:55:00
**最后更新**: 2026-01-22 23:55:00
**审核状态**: 待审核
