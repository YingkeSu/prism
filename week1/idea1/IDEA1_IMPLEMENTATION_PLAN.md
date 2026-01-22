# Idea1 实习实施方案

**创建日期**: 2026-01-22
**项目**: PRISM - UAV Research
**创新点**: 多维度可靠性感知的自适应融合 (Idea1)
**实习周期**: 8周

---

## 执行摘要

本实施方案为Idea1（多维度可靠性感知的自适应融合模块）的详细开发计划，涵盖从环境搭建到论文投稿的完整流程。方案分为4个阶段、15个里程碑、47个具体任务，确保在8周实习期内完成核心实现、实验验证和论文初稿。

---

## 一、总体实施策略

### 1.1 核心原则

| 原则 | 说明 | 应用场景 |
|------|------|---------|
| **渐进式开发** | 从简单到复杂，先验证可行性再扩展 | Week 1-2: 2D简化环境 |
| **模块化设计** | 各模块独立开发、测试、集成 | Week 3-4: 可靠性估计器独立测试 |
| **持续集成** | 每日提交、每周集成测试 | 全周期: Git工作流 |
| **结果导向** | 以实验结果为验收标准 | Week 7-8: 消融实验验证 |
| **风险优先** | 优先解决高风险模块 | Week 2: SB3多模态API验证 |

### 1.2 优先级排序

**P0 - 关键路径 (Critical Path)**:
- ✅ SB3多模态环境搭建
- ✅ 可靠性估计器实现
- ✅ 动态权重分配实现
- ✅ SAC训练成功
- ✅ 对比实验完成

**P1 - 重要任务 (Important)**:
- 理论分析完整性
- 代码优化与重构
- 可视化工具开发

**P2 - 可选任务 (Optional)**:
- Jetson实时部署
- 额外数据集测试
- 高级功能扩展

### 1.3 风险缓解策略

| 风险 | 概率 | 缓解措施 | 应急方案 |
|------|------|---------|---------|
| **SB3集成困难** | 中 | Week 2提前测试API | 直接使用自定义RL框架 |
| **训练不收敛** | 中 | 分阶段训练监控 | 降低任务复杂度 |
| **实验结果不理想** | 中 | 提前设计消融实验 | 聚焦理论贡献 |
| **时间不足** | 低 | 每周进度检查 | 减少消融实验数量 |

---

## 二、分阶段实施计划

### 📅 Week 1-2: 环境搭建与概念验证

#### 阶段目标
✅ 搭建完整的开发环境
✅ 验证SB3多模态融合可行性
✅ 实现2D简化避障环境作为概念验证

#### Week 1: 环境搭建

##### Day 1-2: Conda环境配置
```bash
# 任务1.1: 创建conda环境
conda create -n prism python=3.10 -y
conda activate prism

# 任务1.2: 安装核心依赖
pip install torch>=2.3,<3.0
pip install gymnasium>=0.29.1,<1.3.0
pip install numpy>=1.20,<3.0
pip install pandas matplotlib tensorboard

# 任务1.3: 安装SB3
cd stable-baselines3
pip install -e .

# 任务1.4: 验证安装
python -c "import stable_baselines3; import torch; print('✅ 安装成功')"
```

**验收标准**:
- [ ] Conda环境创建成功
- [ ] PyTorch版本 >= 2.3
- [ ] SB3版本 >= 2.8.0a2
- [ ] 所有import无错误

##### Day 3: 数据集准备
```bash
# 任务2.1: 克隆UAVScenes
git clone https://github.com/sijieaaa/UAVScenes
cd UAVScenes

# 任务2.2: 下载样本数据（100帧）
python scripts/download_sample.py --num_samples 100

# 任务2.3: 数据预处理
python scripts/preprocess_data.py \
    --split train/val/test \
    --normalize rgb \
    --voxelize lidar

# 任务2.4: 验证数据格式
python scripts/verify_data.py --data_dir data/train
```

**验收标准**:
- [ ] UAVScenes仓库克隆成功
- [ ] 100帧样本数据下载完成
- [ ] RGB图像归一化完成
- [ ] LiDAR点云体素化完成
- [ ] 数据格式验证通过

##### Day 4-5: SB3多模态API验证

```python
# 任务3.1: 创建测试脚本
# test_sb3_multimodal.py

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.policies import MultiInputPolicy
import numpy as np

# 创建多模态测试环境
class TestMultimodalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 定义多模态观测空间
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0, 100, (1000, 3), dtype=np.float32),
            "rgb": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            "imu": spaces.Box(-10, 10, (6,), dtype=np.float32)
        })
        
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
    
    def reset(self, seed=None):
        return self._get_observation()
    
    def step(self, action):
        observation = self._get_observation()
        reward = np.random.randn()
        done = False
        info = {}
        return observation, reward, done, info
    
    def _get_observation(self):
        return {
            "lidar": np.random.rand(1000, 3).astype(np.float32) * 100,
            "rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            "imu": np.random.randn(6).astype(np.float32) * 10
        }

# 测试SB3
env = TestMultimodalEnv()

# 任务3.2: 使用MultiInputPolicy
model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]}
)

# 任务3.3: 运行测试训练
model.learn(total_timesteps=1000)

print("✅ SB3多模态API测试通过")
```

**验收标准**:
- [ ] 测试环境创建成功
- [ ] MultiInputPolicy初始化成功
- [ ] 1000步训练无错误
- [ ] 观测空间与动作空间正确

##### Day 6-7: 文档整理与计划调整
- [ ] 整理Week 1工作日志
- [ ] 更新TODO列表
- [ ] 评估进度是否符合预期
- [ ] 调整Week 2计划（如需）

#### Week 2: 概念验证

##### Day 8-9: 2D简化环境实现

```python
# tasks/week2/simple_2d_env.py
"""
2D简化避障环境 - 概念验证
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class Simple2DObstacleEnv(gym.Env):
    """
    2D平面上的UAV避障环境
    
    状态: (x, y, vx, vy) - 位置和速度
    动作: (ax, ay) - 加速度控制
    """
    
    def __init__(self, num_obstacles=5, max_steps=1000):
        super().__init__()
        
        # 状态空间: 位置(2) + 速度(2) = 4维
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(4,), dtype=np.float32
        )
        
        # 动作空间: 加速度(2维)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        
        # 环境参数
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.dt = 0.1  # 时间步长
        
        # 目标位置
        self.goal = np.array([8.0, 8.0])
        self.goal_radius = 1.0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 初始位置在原点
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.step_count = 0
        
        # 随机生成障碍物
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = np.random.uniform(2, 7, size=2)
            radius = np.random.uniform(0.3, 0.6)
            self.obstacles.append({'pos': pos, 'radius': radius})
        
        return self.state.copy(), {}
    
    def step(self, action):
        # 获取当前状态
        x, y, vx, vy = self.state
        ax, ay = action
        
        # 更新速度和位置（物理模型）
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        
        # 更新状态
        self.state = np.array([x_new, y_new, vx_new, vy_new])
        self.step_count += 1
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查终止条件
        done, truncated = self._is_done()
        
        info = {
            'distance_to_goal': np.linalg.norm([x_new, y_new] - self.goal)
        }
        
        return self.state.copy(), reward, done, truncated, info
    
    def _compute_reward(self):
        x, y, _, _ = self.state
        
        # 1. 距离奖励（越接近目标越高）
        dist_to_goal = np.linalg.norm([x, y] - self.goal)
        distance_reward = -dist_to_goal
        
        # 2. 障碍物碰撞惩罚
        collision_penalty = 0
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm([x, y] - obs['pos'])
            if dist_to_obs < obs['radius']:
                collision_penalty = -100  # 碰撞惩罚
        
        # 3. 速度惩罚（鼓励平滑）
        _, _, vx, vy = self.state
        speed = np.sqrt(vx**2 + vy**2)
        speed_penalty = -0.1 * speed
        
        # 4. 到达奖励
        goal_reward = 0
        if dist_to_goal < self.goal_radius:
            goal_reward = 100
        
        return distance_reward + collision_penalty + speed_penalty + goal_reward
    
    def _is_done(self):
        x, y, _, _ = self.state
        
        # 检查碰撞
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm([x, y] - obs['pos'])
            if dist_to_obs < obs['radius']:
                return True, False  # 碰撞
        
        # 检查到达目标
        dist_to_goal = np.linalg.norm([x, y] - self.goal)
        if dist_to_goal < self.goal_radius:
            return True, False  # 成功
        
        # 检查是否超出边界
        if x < -10 or x > 10 or y < -10 or y > 10:
            return True, False  # 出界
        
        # 检查步数限制
        if self.step_count >= self.max_steps:
            return False, True  # 超时
        
        return False, False
    
    def render(self):
        plt.clf()
        
        # 绘制障碍物
        for obs in self.obstacles:
            circle = plt.Circle(obs['pos'], obs['radius'], color='red', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # 绘制目标
        goal_circle = plt.Circle(self.goal, self.goal_radius, color='green', alpha=0.5)
        plt.gca().add_patch(goal_circle)
        
        # 绘制UAV当前位置
        x, y, _, _ = self.state
        plt.plot(x, y, 'bo', markersize=10, label='UAV')
        
        # 绘制轨迹
        if hasattr(self, 'trajectory'):
            traj_x, traj_y = zip(*self.trajectory)
            plt.plot(traj_x, traj_y, 'b--', alpha=0.3)
        
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid(True)
        plt.legend()
        plt.title(f'Step {self.step_count}')
        plt.pause(0.01)

# 测试环境
if __name__ == "__main__":
    env = Simple2DObstacleEnv(num_obstacles=5)
    env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done or truncated:
            break
    
    print("✅ 2D环境测试通过")
```

**验收标准**:
- [ ] 2D环境创建成功
- [ ] 状态空间、动作空间正确
- [ ] 碰撞检测准确
- [ ] 目标到达判断正确
- [ ] 可视化正常显示

##### Day 10-12: 基础SAC训练

```python
# tasks/week2/train_2d_sac.py

from stable_baselines3 import SAC
from simple_2d_env import Simple2DObstacleEnv
import os

# 创建环境
env = Simple2DObstacleEnv(num_obstacles=5)

# 创建模型
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',
    tensorboard_log="./logs/2d_sac"
)

# 训练
print("开始训练...")
model.learn(total_timesteps=100000)

# 保存模型
os.makedirs("models", exist_ok=True)
model.save("models/2d_sac_baseline")

print("✅ 训练完成，模型已保存")
```

**验收标准**:
- [ ] SAC模型创建成功
- [ ] 训练过程无错误
- [ ] TensorBoard日志正常记录
- [ ] 成功率 > 50%（10次评估）
- [ ] 平均路径长度 < 30步

##### Day 13-14: 可靠性估计器简化实现

```python
# tasks/week2/simple_reliability_estimator.py

import numpy as np

class SimpleReliabilityEstimator:
    """
    简化的可靠性估计器（2D环境专用）
    
    评估指标：
    - 距离可靠性：离目标越近，可靠性越高
    - 速度可靠性：速度适中时可靠性高
    - 方向可靠性：朝向目标时可靠性高
    """
    
    def __init__(self, goal_pos):
        self.goal_pos = goal_pos
    
    def estimate(self, state):
        """
        估计当前状态的可靠性
        
        Args:
            state: (x, y, vx, vy)
        
        Returns:
            reliability: 0-1的可靠性分数
        """
        x, y, vx, vy = state
        
        # 1. 距离可靠性（高斯函数）
        dist_to_goal = np.linalg.norm([x, y] - self.goal_pos)
        distance_reliability = np.exp(-dist_to_goal**2 / (2 * 5**2))
        
        # 2. 速度可靠性（速度在[0, 2]范围内最高）
        speed = np.sqrt(vx**2 + vy**2)
        if 0 <= speed <= 2:
            speed_reliability = 1.0
        else:
            speed_reliability = max(0, 1.0 - (speed - 2) / 3)
        
        # 3. 方向可靠性（朝向目标时最高）
        desired_dir = self.goal_pos - np.array([x, y])
        desired_dir = desired_dir / (np.linalg.norm(desired_dir) + 1e-6)
        current_dir = np.array([vx, vy])
        current_dir = current_dir / (np.linalg.norm(current_dir) + 1e-6)
        direction_reliability = np.dot(desired_dir, current_dir)
        direction_reliability = max(0, direction_reliability)
        
        # 综合可靠性（加权平均）
        reliability = (
            0.4 * distance_reliability +
            0.3 * speed_reliability +
            0.3 * direction_reliability
        )
        
        return float(reliability)

# 测试
if __name__ == "__main__":
    estimator = SimpleReliabilityEstimator(goal_pos=np.array([8.0, 8.0]))
    
    # 测试不同状态
    test_states = [
        np.array([0.0, 0.0, 1.0, 1.0]),  # 初始状态
        np.array([4.0, 4.0, 1.0, 1.0]),  # 中间状态
        np.array([7.0, 7.0, 0.5, 0.5]),  # 接近目标
        np.array([8.0, 8.0, 0.0, 0.0]),  # 到达目标
    ]
    
    for state in test_states:
        reliability = estimator.estimate(state)
        print(f"State: {state}, Reliability: {reliability:.3f}")
    
    print("✅ 可靠性估计器测试通过")
```

**验收标准**:
- [ ] 可靠性估计器初始化成功
- [ ] 不同状态返回不同可靠性分数
- [ ] 接近目标时可靠性较高
- [ ] 速度适中时可靠性较高

##### Day 15: Week 2总结与Week 3准备
- [ ] Week 2工作日志整理
- [ ] 2D环境训练结果分析
- [ ] 可靠性估计器简化版本测试
- [ ] Week 3任务分解
- [ ] 确认Week 3开发环境

---

### 📅 Week 3-4: 核心模块实现

#### 阶段目标
✅ 实现LiDAR SNR估计器
✅ 实现图像质量评估器
✅ 实现IMU一致性检查器
✅ 实现可靠性预测网络
✅ 验证各模块独立功能

#### Week 3: 可靠性估计器实现

##### Day 16-18: LiDAR SNR估计器

```python
# networks/reliability_estimators/lidar_snr_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LiDARSNREstimator(nn.Module):
    """
    LiDAR点云信噪比估计器
    
    输入: (B, N, 3) 点云
    输出: {'snr': (B, 1), 'density': (B, 1)}
    """
    
    def __init__(self, point_dim=3, feature_dim=64):
        super().__init__()
        
        # 统计特征提取
        self.point_stats = nn.Sequential(
            nn.Conv1d(point_dim, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # SNR预测头
        self.snr_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 点密度预测头
        self.density_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, lidar_points):
        """
        Args:
            lidar_points: (B, N, 3) 点云
        
        Returns:
            dict: 包含snr和density
        """
        # 转置: (B, 3, N) -> Conv1d expects (B, C, N)
        x = lidar_points.transpose(1, 2)
        
        # 提取统计特征
        features = self.point_stats(x)  # (B, 64, 1)
        features = features.squeeze(-1)  # (B, 64)
        
        # 预测SNR和密度
        snr = self.snr_head(features)  # (B, 1)
        density = self.density_head(features)  # (B, 1)
        
        return {
            'snr': snr,
            'density': density
        }
    
    def compute_snr_metrics(self, lidar_points):
        """
        计算传统SNR指标（用于验证）
        """
        # 点云密度
        B, N, _ = lidar_points.shape
        density = N / 1000.0  # 归一化到0-1
        
        # 点云分布均匀性
        center = lidar_points.mean(dim=1, keepdim=True)  # (B, 1, 3)
        distances = torch.norm(lidar_points - center, dim=-1)  # (B, N)
        std_distance = distances.std(dim=1, keepdim=True)  # (B, 1)
        uniformity = torch.exp(-std_distance / 5.0)  # 距离标准差越小，均匀性越高
        
        # 综合SNR
        snr = 0.5 * density + 0.5 * uniformity
        
        return {'snr': snr, 'density': density, 'uniformity': uniformity}

# 测试
if __name__ == "__main__":
    model = LiDARSNREstimator()
    
    # 创建测试数据
    batch_size = 4
    num_points = 1000
    lidar_points = torch.randn(batch_size, num_points, 3)
    
    # 前向传播
    output = model(lidar_points)
    
    print(f"SNR shape: {output['snr'].shape}")
    print(f"Density shape: {output['density'].shape}")
    print(f"SNR range: [{output['snr'].min():.3f}, {output['snr'].max():.3f}]")
    
    # 验证传统指标
    metrics = model.compute_snr_metrics(lidar_points)
    print(f"Traditional SNR: {metrics['snr'].mean():.3f}")
    
    print("✅ LiDAR SNR估计器测试通过")
```

**验收标准**:
- [ ] LiDAR SNR估计器创建成功
- [ ] 输出维度正确
- [ ] SNR值在0-1范围内
- [ ] 稀疏点云SNR较低
- [ ] 密集点云SNR较高

##### Day 19-21: 图像质量评估器

```python
# networks/reliability_estimators/image_quality_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class ImageQualityEstimator(nn.Module):
    """
    RGB图像质量评估器
    
    评估指标：
    - Sharpness: 锐利度（Laplacian算子）
    - Contrast: 对比度（局部标准差）
    - Brightness: 亮度（直方图分析）
    """
    
    def __init__(self):
        super().__init__()
        
        # 锐利度检测器（可微分的Laplacian卷积）
        self.laplacian_kernel = nn.Parameter(
            torch.tensor([
                [[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]]
            ], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
        
        # 特征融合网络
        self.quality_net = nn.Sequential(
            nn.Linear(3, 16),  # 3个质量指标
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_image):
        """
        Args:
            rgb_image: (B, 3, H, W) RGB图像
        
        Returns:
            dict: 包含sharpness, contrast, brightness, overall_quality
        """
        # 转为灰度
        gray = 0.299 * rgb_image[:, 0:1, :, :] + \
                0.587 * rgb_image[:, 1:2, :, :] + \
                0.114 * rgb_image[:, 2:3, :, :]
        
        # 1. 锐利度（Laplacian算子）
        sharpness = self._compute_sharpness(gray)
        
        # 2. 对比度（局部标准差）
        contrast = self._compute_contrast(gray)
        
        # 3. 亮度（直方图分析）
        brightness = self._compute_brightness(rgb_image)
        
        # 综合质量评分
        quality_features = torch.cat([
            sharpness, contrast, brightness
        ], dim=1)
        
        overall_quality = self.quality_net(quality_features)
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'overall_quality': overall_quality
        }
    
    def _compute_sharpness(self, gray_image):
        """
        使用Laplacian算子计算锐利度
        """
        # 应用Laplacian核
        laplacian = F.conv2d(
            gray_image,
            self.laplacian_kernel,
            padding=1
        )
        
        # 锐利度 = 方差
        sharpness = torch.var(laplacian, dim=[2, 3], keepdim=True)
        
        # 归一化到0-1
        sharpness = torch.sigmoid(sharpness * 0.1)
        
        return sharpness
    
    def _compute_contrast(self, gray_image):
        """
        计算局部标准差作为对比度
        """
        # 使用7x7局部窗口
        kernel_size = 7
        pad = kernel_size // 2
        padded = F.pad(gray_image, (pad, pad, pad, pad), mode='reflect')
        
        # 滑动窗口计算标准差
        patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        local_std = patches.std(dim=(-2, -1))
        
        # 全局对比度
        contrast = local_std.mean(dim=[2, 3], keepdim=True)
        
        # 归一化到0-1
        contrast = torch.sigmoid(contrast * 0.5)
        
        return contrast
    
    def _compute_brightness(self, rgb_image):
        """
        计算亮度（直方图分析）
        """
        # 计算平均亮度
        brightness = rgb_image.mean(dim=[2, 3], keepdim=True) / 255.0
        
        # 适中亮度时质量最高（0.4-0.6）
        ideal_brightness = 0.5
        brightness_quality = 1.0 - torch.abs(brightness - ideal_brightness)
        
        return brightness_quality

# 测试
if __name__ == "__main__":
    model = ImageQualityEstimator()
    
    # 创建测试图像
    batch_size = 4
    height, width = 128, 128
    rgb_image = torch.rand(batch_size, 3, height, width) * 255
    
    # 模拟不同质量
    # 图像0: 清晰
    # 图像1: 模糊（低通滤波）
    # 图像2: 低对比度
    # 图像3: 暗色
    rgb_image[1] = F.avg_pool2d(rgb_image[1:2], kernel_size=5, stride=1, padding=2).squeeze(0)
    rgb_image[2] = rgb_image[2] * 0.3 + 128
    rgb_image[3] = rgb_image[3] * 0.2
    
    # 前向传播
    output = model(rgb_image)
    
    print(f"Overall quality: {output['overall_quality'].squeeze()}")
    print(f"Sharpness: {output['sharpness'].squeeze()}")
    print(f"Contrast: {output['contrast'].squeeze()}")
    print(f"Brightness: {output['brightness'].squeeze()}")
    
    # 验证：清晰图像质量最高
    assert output['overall_quality'][0] > output['overall_quality'][1], "模糊图像质量应该更低"
    assert output['contrast'][0] > output['contrast'][2], "低对比度图像应该更低"
    
    print("✅ 图像质量评估器测试通过")
```

**验收标准**:
- [ ] 图像质量评估器创建成功
- [ ] 输出维度正确
- [ ] 清晰图像质量较高
- [ ] 模糊图像质量较低
- [ ] 对比度和亮度指标合理

##### Day 22-24: IMU一致性检查器

```python
# networks/reliability_estimators/imu_consistency_checker.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUConsistencyChecker(nn.Module):
    """
    IMU数据一致性检查器
    
    检查项：
    - Drift: 加速度计/陀螺仪漂移
    - Velocity Anomaly: 速度异常检测
    - Orientation Consistency: 姿态一致性
    """
    
    def __init__(self, window_size=100):
        super().__init__()
        
        self.window_size = window_size
        
        # 漂移分析器
        self.drift_analyzer = nn.Sequential(
            nn.Linear(window_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 速度异常检测器
        self.velocity_anomaly_detector = nn.Sequential(
            nn.Linear(window_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, imu_sequence):
        """
        Args:
            imu_sequence: (B, T, 6) IMU序列 [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        
        Returns:
            dict: 包含drift_score, velocity_anomaly, consistency
        """
        batch_size, seq_len, _ = imu_sequence.shape
        
        # 使用滑动窗口
        if seq_len < self.window_size:
            # 填充到window_size
            padding = torch.zeros(batch_size, self.window_size - seq_len, 6)
            padded = torch.cat([padding, imu_sequence], dim=1)
        else:
            padded = imu_sequence[:, -self.window_size:, :]
        
        # 展平
        flat = padded.reshape(batch_size, -1)
        
        # 分析漂移和异常
        drift_score = self.drift_analyzer(flat)  # (B, 1)
        velocity_anomaly = self.velocity_anomaly_detector(flat)  # (B, 1)
        
        # 一致性 = 1 - 漂移 - 异常
        consistency = 1.0 - 0.5 * drift_score - 0.5 * velocity_anomaly
        consistency = torch.clamp(consistency, 0.0, 1.0)
        
        return {
            'drift_score': drift_score,
            'velocity_anomaly': velocity_anomaly,
            'consistency': consistency
        }
    
    def compute_traditional_metrics(self, imu_sequence):
        """
        计算传统IMU一致性指标（用于验证）
        """
        # 加速度计均值（静止时应该接近[0,0,1g]）
        acc_mean = imu_sequence[:, :, :3].mean(dim=1)
        acc_std = imu_sequence[:, :, :3].std(dim=1)
        
        # 陀螺仪均值（静止时应该接近[0,0,0]）
        gyro_mean = imu_sequence[:, :, 3:].mean(dim=1)
        gyro_std = imu_sequence[:, :, 3:].std(dim=1)
        
        # 漂移分数（标准差越大，漂移越严重）
        acc_drift = acc_std.mean(dim=1, keepdim=True)
        gyro_drift = gyro_std.mean(dim=1, keepdim=True)
        total_drift = (acc_drift + gyro_drift) / 2.0
        
        # 归一化到0-1（使用sigmoid）
        consistency = torch.sigmoid(-total_drift * 10.0)
        
        return {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'gyro_mean': gyro_mean,
            'gyro_std': gyro_std,
            'consistency': consistency
        }

# 测试
if __name__ == "__main__":
    model = IMUConsistencyChecker(window_size=100)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 100
    imu_sequence = torch.randn(batch_size, seq_len, 6)
    
    # 模拟不同质量
    # IMU 0: 高质量（低噪声）
    imu_sequence[0] = torch.randn(seq_len, 6) * 0.01
    # IMU 1: 中等质量
    imu_sequence[1] = torch.randn(seq_len, 6) * 0.1
    # IMU 2: 低质量（高噪声）
    imu_sequence[2] = torch.randn(seq_len, 6) * 1.0
    # IMU 3: 有漂移
    imu_sequence[3] = torch.randn(seq_len, 6) * 0.1
    imu_sequence[3, :, :3] += torch.linspace(0, 1, seq_len).unsqueeze(0).t()  # 加速度漂移
    
    # 前向传播
    output = model(imu_sequence)
    
    print(f"Consistency scores: {output['consistency'].squeeze()}")
    print(f"Drift scores: {output['drift_score'].squeeze()}")
    
    # 验证：高质量IMU一致性最高
    assert output['consistency'][0] > output['consistency'][2], "低噪声IMU应该一致性更高"
    assert output['consistency'][0] > output['consistency'][3], "无漂移IMU应该一致性更高"
    
    print("✅ IMU一致性检查器测试通过")
```

**验收标准**:
- [ ] IMU一致性检查器创建成功
- [ ] 输出维度正确
- [ ] 低噪声IMU一致性较高
- [ ] 有漂移IMU一致性较低

##### Day 25-28: 可靠性预测网络集成

```python
# networks/reliability_predictor.py

import torch
import torch.nn as nn
from networks.reliability_estimators.lidar_snr_estimator import LiDARSNREstimator
from networks.reliability_estimators.image_quality_estimator import ImageQualityEstimator
from networks.reliability_estimators.imu_consistency_checker import IMUConsistencyChecker

class ReliabilityPredictor(nn.Module):
    """
    轻量级可靠性预测网络
    
    输入：LiDAR点云, RGB图像, IMU序列
    输出：3个可靠性分数 (LiDAR: 0-1, RGB: 0-1, IMU: 0-1)
    总参数量：<500K
    """
    
    def __init__(self, lidar_dim=64, rgb_dim=256, imu_dim=6, hidden_dim=128):
        super().__init__()
        
        # 3个可靠性估计器
        self.lidar_estimator = LiDARSNREstimator()
        self.rgb_estimator = ImageQualityEstimator()
        self.imu_estimator = IMUConsistencyChecker(window_size=100)
        
        # 编码器
        self.lidar_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.rgb_encoder = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 分数头
        self.lidar_reliability = nn.Linear(128, 1)
        self.rgb_reliability = nn.Linear(128, 1)
        self.imu_reliability = nn.Linear(128, 1)
    
    def forward(self, lidar_points, rgb_image, imu_data):
        """
        Args:
            lidar_points: (B, N, 3) 点云
            rgb_image: (B, 3, H, W) RGB图像
            imu_data: (B, T, 6) IMU序列
        
        Returns:
            dict: 包含r_lidar, r_rgb, r_imu, features
        """
        # 估计各模态质量指标
        lidar_output = self.lidar_estimator(lidar_points)
        rgb_output = self.rgb_estimator(rgb_image)
        imu_output = self.imu_estimator(imu_data)
        
        # 提取特征
        lidar_feat = self.lidar_encoder(lidar_output['snr'])
        rgb_feat = self.rgb_encoder(rgb_output['overall_quality'])
        imu_feat = self.imu_encoder(imu_output['consistency'])
        
        # 融合特征
        fused = torch.cat([lidar_feat, rgb_feat, imu_feat], dim=1)
        features = self.feature_fusion(fused)
        
        # 输出分数
        r_lidar = torch.sigmoid(self.lidar_reliability(features))
        r_rgb = torch.sigmoid(self.rgb_reliability(features))
        r_imu = torch.sigmoid(self.imu_reliability(features))
        
        return {
            'r_lidar': r_lidar,
            'r_rgb': r_rgb,
            'r_imu': r_imu,
            'features': features
        }
    
    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 测试
if __name__ == "__main__":
    model = ReliabilityPredictor()
    
    # 创建测试数据
    batch_size = 4
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)
    
    # 前向传播
    output = model(lidar_points, rgb_image, imu_data)
    
    print(f"LiDAR reliability: {output['r_lidar'].squeeze()}")
    print(f"RGB reliability: {output['r_rgb'].squeeze()}")
    print(f"IMU reliability: {output['r_imu'].squeeze()}")
    print(f"Total parameters: {model.count_parameters() / 1000:.1f}K")
    
    # 验证参数量
    assert model.count_parameters() < 500000, "参数量应该<500K"
    
    print("✅ 可靠性预测网络测试通过")
```

**验收标准**:
- [ ] 可靠性预测网络创建成功
- [ ] 输出3个可靠性分数
- [ ] 参数量 < 500K
- [ ] 各子模块正常工作

##### Day 29-30: Week 3总结
- [ ] Week 3工作日志整理
- [ ] 可靠性估计器测试报告
- [ ] Week 4任务确认

---

#### Week 4: 动态权重分配与融合

##### Day 31-33: 动态权重分配层

```python
# networks/dynamic_weighting_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWeightingLayer(nn.Module):
    """
    动态权重分配层，基于注意力机制融合多模态特征
    
    输入：3个可靠性分数
    输出：归一化的融合权重（和为1）
    机制：多头注意力 + Softmax归一化
    """
    
    def __init__(self, feature_dim=256, num_heads=8):
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
    
    def forward(self, lidar_feat, rgb_feat, imu_feat, temperature=None):
        """
        Args:
            lidar_feat: (B, D) LiDAR特征
            rgb_feat: (B, D) RGB特征
            imu_feat: (B, D) IMU特征
            temperature: 可选的温度缩放
        
        Returns:
            weights: dict {'w_lidar': (B, 1), 'w_rgb': (B, 1), 'w_imu': (B, 1)}
        """
        batch_size = lidar_feat.shape[0]
        
        # 多模态特征拼接: (B, 3, D)
        multimodal_feat = torch.stack([
            lidar_feat,
            rgb_feat,
            imu_feat
        ], dim=1)
        
        # 多头注意力
        attention_output, attention_weights = self.multi_head_attention(
            multimodal_feat, multimodal_feat, multimodal_feat
        )
        attention_output = attention_output.mean(dim=1)  # (B, D)
        
        # 提取注意力分数
        attention_scores = attention_weights.mean(dim=1)  # (B, 3)
        
        # 应用温度缩放
        if temperature is None:
            temperature = torch.exp(self.temperature)
        attention_scores = attention_scores / temperature
        
        # 加上偏置
        w_lidar = F.softmax(attention_scores[:, 0:1] + self.bias_lidar, dim=-1)
        w_rgb = F.softmax(attention_scores[:, 1:2] + self.bias_rgb, dim=-1)
        w_imu = F.softmax(attention_scores[:, 2:3] + self.bias_imu, dim=-1)
        
        return {
            'w_lidar': w_lidar,
            'w_rgb': w_rgb,
            'w_imu': w_imu,
            'attention_scores': attention_scores,
            'attention_weights': attention_weights
        }

# 测试
if __name__ == "__main__":
    model = DynamicWeightingLayer(feature_dim=128, num_heads=8)
    
    # 创建测试数据
    batch_size = 4
    feature_dim = 128
    lidar_feat = torch.randn(batch_size, feature_dim)
    rgb_feat = torch.randn(batch_size, feature_dim)
    imu_feat = torch.randn(batch_size, feature_dim)
    
    # 前向传播
    output = model(lidar_feat, rgb_feat, imu_feat)
    
    print(f"LIDAR weight: {output['w_lidar'].squeeze()}")
    print(f"RGB weight: {output['w_rgb'].squeeze()}")
    print(f"IMU weight: {output['w_imu'].squeeze()}")
    
    # 验证权重和为1
    total_weight = output['w_lidar'] + output['w_rgb'] + output['w_imu']
    print(f"Total weight: {total_weight.squeeze()}")
    assert torch.allclose(total_weight, torch.ones_like(total_weight), atol=1e-5), "权重和应该为1"
    
    print("✅ 动态权重分配层测试通过")
```

**验收标准**:
- [ ] 动态权重分配层创建成功
- [ ] 权重和为1
- [ ] 注意力机制正常工作
- [ ] 温度缩放有效

##### Day 34-36: 自适应归一化层

```python
# networks/adaptive_normalization.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveNormalization(nn.Module):
    """
    自适应归一化层，根据可靠性分数动态调整归一化策略
    
    输入：可靠性分数 + 原始特征
    输出：归一化后的特征
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        
        # 可学习的归一化参数
        self.gamma_lidar = nn.Parameter(torch.ones(1))
        self.gamma_rgb = nn.Parameter(torch.ones(1))
        self.gamma_imu = nn.Parameter(torch.ones(1))
        
        # 偏移参数
        self.beta_lidar = nn.Parameter(torch.zeros(1))
        self.beta_rgb = nn.Parameter(torch.zeros(1))
        self.beta_imu = nn.Parameter(torch.zeros(1))
        
        # 滑动窗口统计
        self.window_size = 100
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
    
    def forward(self, r_lidar, r_rgb, r_imu, features):
        """
        Args:
            r_lidar: (B, 1) LiDAR可靠性分数
            r_rgb: (B, 1) RGB可靠性分数
            r_imu: (B, 1) IMU可靠性分数
            features: dict {'lidar': (B, D), 'rgb': (B, D), 'imu': (B, D)}
        
        Returns:
            dict: 归一化后的特征
        """
        # LiDAR特征归一化
        lidar_norm = F.normalize(
            features['lidar'] - self.beta_lidar,
            p=2, dim=1, eps=1e-6
        )
        lidar_out = self.gamma_lidar * lidar_norm
        
        # RGB特征归一化
        rgb_norm = F.normalize(
            features['rgb'] - self.beta_rgb,
            p=2, dim=1, eps=1e-6
        )
        rgb_out = self.gamma_rgb * rgb_norm
        
        # IMU特征归一化
        imu_norm = F.normalize(
            features['imu'] - self.beta_imu,
            p=2, dim=1, eps=1e-6
        )
        imu_out = self.gamma_imu * imu_norm
        
        return {
            'lidar_out': lidar_out,
            'rgb_out': rgb_out,
            'imu_out': imu_out
        }

# 测试
if __name__ == "__main__":
    model = AdaptiveNormalization(feature_dim=128)
    
    # 创建测试数据
    batch_size = 4
    feature_dim = 128
    r_lidar = torch.rand(batch_size, 1)
    r_rgb = torch.rand(batch_size, 1)
    r_imu = torch.rand(batch_size, 1)
    features = {
        'lidar': torch.randn(batch_size, feature_dim),
        'rgb': torch.randn(batch_size, feature_dim),
        'imu': torch.randn(batch_size, feature_dim)
    }
    
    # 前向传播
    output = model(r_lidar, r_rgb, r_imu, features)
    
    print(f"LiDAR out shape: {output['lidar_out'].shape}")
    print(f"RGB out shape: {output['rgb_out'].shape}")
    print(f"IMU out shape: {output['imu_out'].shape}")
    
    # 验证归一化效果
    lidar_norm = torch.norm(output['lidar_out'], dim=1)
    print(f"LiDAR normalized norms: {lidar_norm}")
    
    print("✅ 自适应归一化层测试通过")
```

**验收标准**:
- [ ] 自适应归一化层创建成功
- [ ] 输出维度正确
- [ ] 归一化有效
- [ ] 可学习参数正常更新

##### Day 37-40: 完整融合模块

```python
# networks/reliability_aware_fusion.py

import torch
import torch.nn as nn
from networks.reliability_predictor import ReliabilityPredictor
from networks.dynamic_weighting_layer import DynamicWeightingLayer
from networks.adaptive_normalization import AdaptiveNormalization

class ReliabilityAwareFusionModule(nn.Module):
    """
    可靠性感知融合模块，集成：
    1. 可靠性估计
    2. 动态权重分配
    3. 自适应归一化
    4. 特征融合
    """
    
    def __init__(self, feature_dim=256, num_heads=8):
        super().__init__()
        
        # 子模块
        self.reliability_estimator = ReliabilityPredictor()
        self.dynamic_weighting = DynamicWeightingLayer(
            feature_dim=feature_dim,
            num_heads=num_heads
        )
        self.adaptive_norm = AdaptiveNormalization(feature_dim=feature_dim)
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 编码器（从原始数据到特征）
        self.lidar_encoder = nn.Sequential(
            nn.Linear(64, feature_dim // 3),
            nn.ReLU()
        )
        self.rgb_encoder = nn.Sequential(
            nn.Linear(256, feature_dim // 3),
            nn.ReLU()
        )
        self.imu_encoder = nn.Sequential(
            nn.Linear(6, feature_dim // 3),
            nn.ReLU()
        )
    
    def forward(self, lidar_points, rgb_image, imu_data):
        """
        Args:
            lidar_points: (B, N, 3) LiDAR点云
            rgb_image: (B, 3, H, W) RGB图像
            imu_data: (B, T, 6) IMU序列
        
        Returns:
            dict: 融合输出 + 中间信息（用于可视化）
        """
        batch_size = lidar_points.shape[0]
        
        # 1. 可靠性估计
        reliability_output = self.reliability_estimator(
            lidar_points, rgb_image, imu_data
        )
        
        r_lidar = reliability_output['r_lidar']
        r_rgb = reliability_output['r_rgb']
        r_imu = reliability_output['r_imu']
        features = reliability_output['features']
        
        # 2. 编码各模态特征
        lidar_feat = self.lidar_encoder(r_lidar.expand(-1, features.shape[1] // 3))
        rgb_feat = self.rgb_encoder(r_rgb.expand(-1, features.shape[1] // 3))
        imu_feat = self.imu_encoder(r_imu.expand(-1, features.shape[1] // 3))
        
        # 3. 动态权重分配
        weights = self.dynamic_weighting(lidar_feat, rgb_feat, imu_feat)
        
        w_lidar = weights['w_lidar']
        w_rgb = weights['w_rgb']
        w_imu = weights['w_imu']
        
        # 4. 自适应归一化
        raw_features = {
            'lidar': lidar_feat,
            'rgb': rgb_feat,
            'imu': imu_feat
        }
        normed_features = self.adaptive_norm(r_lidar, r_rgb, r_imu, raw_features)
        
        # 5. 加权融合
        fused = torch.cat([
            normed_features['lidar_out'] * w_lidar,
            normed_features['rgb_out'] * w_rgb,
            normed_features['imu_out'] * w_imu
        ], dim=1)
        
        # 6. 通过融合网络
        output = self.fusion_net(fused)
        
        return {
            'output': output,
            'reliability': {
                'lidar': r_lidar,
                'rgb': r_rgb,
                'imu': r_imu
            },
            'weights': weights,
            'normed_features': normed_features
        }
    
    def get_loss(self, predictions, targets):
        """
        计算损失函数
        
        Args:
            predictions: 模型输出
            targets: 目标值
        
        Returns:
            dict: 包含total_loss和各项损失
        """
        # 主要损失（MSE）
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 可靠性正则化（鼓励各模态均衡使用）
        reliability = predictions['reliability']
        r_lidar = reliability['lidar']
        r_rgb = reliability['rgb']
        r_imu = reliability['imu']
        
        # 方差正则化（鼓励均衡）
        mean_r = (r_lidar + r_rgb + r_imu) / 3.0
        var_r = ((r_lidar - mean_r)**2 + (r_rgb - mean_r)**2 + (r_imu - mean_r)**2) / 3.0
        reliability_reg = 0.01 * var_r.mean()
        
        # 总损失
        total_loss = mse_loss + reliability_reg
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'reliability_reg': reliability_reg
        }

# 测试
if __name__ == "__main__":
    model = ReliabilityAwareFusionModule(feature_dim=256, num_heads=8)
    
    # 创建测试数据
    batch_size = 4
    lidar_points = torch.randn(batch_size, 1000, 3)
    rgb_image = torch.rand(batch_size, 3, 128, 128) * 255
    imu_data = torch.randn(batch_size, 100, 6)
    targets = torch.randn(batch_size, 64)
    
    # 前向传播
    output = model(lidar_points, rgb_image, imu_data)
    
    print(f"Output shape: {output['output'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1000:.1f}K")
    
    # 计算损失
    loss_dict = model.get_loss(output, targets)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"MSE loss: {loss_dict['mse_loss'].item():.4f}")
    print(f"Reliability reg: {loss_dict['reliability_reg'].item():.4f}")
    
    print("✅ 完整融合模块测试通过")
```

**验收标准**:
- [ ] 完整融合模块创建成功
- [ ] 所有子模块正常工作
- [ ] 损失函数计算正确
- [ ] 参数量合理

##### Day 41-42: Week 4总结
- [ ] Week 4工作日志整理
- [ ] 完整模块测试报告
- [ ] Week 5任务确认

---

### 📅 Week 5-6: 系统集成与实验

#### 阶段目标
✅ 集成到SB3训练循环
✅ 完成大规模训练
✅ 初步结果分析

#### Week 5: SB3集成

##### Day 43-45: 自定义特征提取器

```python
# networks/uav_multimodal_extractor.py

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from networks.reliability_aware_fusion import ReliabilityAwareFusionModule

class UAVMultimodalExtractor(BaseFeaturesExtractor):
    """
    自定义多模态特征提取器，集成可靠性感知融合
    
    输入：Dict observation space {'lidar': ..., 'rgb': ..., 'imu': ...}
    输出：融合后的特征向量
    """
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 提取各模态的维度
        lidar_shape = observation_space['lidar'].shape  # (N, 3)
        rgb_shape = observation_space['rgb'].shape  # (H, W, 3)
        imu_shape = observation_space['imu'].shape  # (6,)
        
        # 基础编码器（将原始数据编码为特征）
        self.lidar_base_encoder = nn.Sequential(
            nn.Linear(lidar_shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.rgb_base_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (rgb_shape[0]//4) * (rgb_shape[1]//4), 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        self.imu_base_encoder = nn.Sequential(
            nn.Linear(imu_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        
        # 可靠性感知融合模块
        self.reliability_fusion = ReliabilityAwareFusionModule(
            feature_dim=features_dim,
            num_heads=8
        )
        
        # 输出投影
        self.output_projection = nn.Linear(64, features_dim)
    
    def forward(self, observations):
        """
        Args:
            observations: dict with keys 'lidar', 'rgb', 'imu'
        
        Returns:
            features: (B, features_dim)
        """
        # 获取各模态数据
        lidar_points = observations['lidar']  # (B, N, 3)
        rgb_image = observations['rgb']  # (B, H, W, 3)
        imu_data = observations['imu']  # (B, 6)
        
        # 转换RGB格式
        rgb_image = rgb_image.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
        
        # 扩展IMU为序列（模拟历史）
        imu_sequence = imu_data.unsqueeze(1).expand(-1, 100, -1)
        
        # 基础编码
        lidar_feat = self.lidar_base_encoder(lidar_points.mean(dim=1))  # (B, 64)
        rgb_feat = self.rgb_base_encoder(rgb_image)  # (B, 256)
        imu_feat = self.imu_base_encoder(imu_data)  # (B, 6)
        
        # 重构为完整格式（用于融合模块）
        # 注意：这里简化处理，实际可能需要更复杂的重构
        
        # 使用融合模块
        fusion_output = self.reliability_fusion(
            lidar_points,
            rgb_image,
            imu_sequence
        )
        
        # 投影到输出维度
        features = self.output_projection(fusion_output['output'])
        
        return features

# 测试
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium import spaces
    
    # 创建测试环境
    obs_space = spaces.Dict({
        'lidar': spaces.Box(0, 100, (1000, 3), dtype=np.float32),
        'rgb': spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        'imu': spaces.Box(-10, 10, (6,), dtype=np.float32)
    })
    
    extractor = UAVMultimodalExtractor(obs_space, features_dim=256)
    
    # 创建测试观测
    observations = {
        'lidar': torch.randn(4, 1000, 3),
        'rgb': torch.randint(0, 255, (4, 128, 128, 3)),
        'imu': torch.randn(4, 6)
    }
    
    # 前向传播
    features = extractor(observations)
    
    print(f"Output features shape: {features.shape}")
    
    print("✅ 自定义特征提取器测试通过")
```

**验收标准**:
- [ ] 自定义特征提取器创建成功
- [ ] 继承BaseFeaturesExtractor
- [ ] 输入输出维度正确
- [ ] 与SB3兼容

##### Day 46-48: 多模态UAV环境

```python
# envs/uav_multimodal_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

class UAVMultimodalEnv(gym.Env):
    """
    多模态UAV仿真环境
    
    观测空间：
    - lidar: (1000, 3) 点云
    - rgb: (128, 128, 3) RGB图像
    - imu: (6,) IMU数据
    
    动作空间：
    - velocity: (4,) 线速度 + 角速度
    """
    
    def __init__(self, max_steps=1000):
        super().__init__()
        
        # 观测空间
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0, 100, (1000, 3), dtype=np.float32),
            "rgb": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            "imu": spaces.Box(-10, 10, (6,), dtype=np.float32)
        })
        
        # 动作空间
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        
        # 环境参数
        self.max_steps = max_steps
        self.dt = 0.1
        
        # UAV状态
        self.position = np.zeros(3)  # x, y, z
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        # 目标
        self.goal_position = np.array([8.0, 8.0, 5.0])
        self.goal_radius = 1.0
        
        # 障碍物
        self.obstacles = []
        self._generate_obstacles()
        
    def _generate_obstacles(self):
        """生成随机障碍物"""
        num_obstacles = 10
        for _ in range(num_obstacles):
            pos = np.random.uniform(2, 7, size=3)
            radius = np.random.uniform(0.5, 1.0)
            self.obstacles.append({'pos': pos, 'radius': radius})
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 重置UAV状态
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.step_count = 0
        
        # 重新生成障碍物
        self.obstacles = []
        self._generate_obstacles()
        
        return self._get_observation(), {}
    
    def step(self, action):
        # 解析动作
        linear_velocity = action[:3] * 2.0  # 最大2 m/s
        angular_velocity = action[3] * 1.0  # 最大1 rad/s
        
        # 更新状态（简化物理模型）
        self.velocity += linear_velocity * self.dt
        self.position += self.velocity * self.dt
        self.orientation += np.array([0, 0, angular_velocity]) * self.dt
        
        self.step_count += 1
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查终止条件
        done, truncated = self._is_done()
        
        info = {
            'distance_to_goal': np.linalg.norm(self.position - self.goal_position)
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self):
        """获取多模态观测"""
        # 1. LiDAR点云（模拟）
        lidar = self._simulate_lidar()
        
        # 2. RGB图像（模拟）
        rgb = self._simulate_rgb()
        
        # 3. IMU数据（模拟）
        imu = self._simulate_imu()
        
        return {
            'lidar': lidar,
            'rgb': rgb,
            'imu': imu
        }
    
    def _simulate_lidar(self):
        """模拟LiDAR扫描"""
        # 简化：随机生成点云
        points = []
        for _ in range(1000):
            # 在UAV周围随机生成点
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            distance = np.random.uniform(0, 10)
            
            point = self.position + direction * distance
            
            # 检查是否碰到障碍物
            for obs in self.obstacles:
                if np.linalg.norm(point - obs['pos']) < obs['radius']:
                    distance = np.linalg.norm(point - self.position)
                    point = self.position + direction * distance
                    break
            
            points.append(point)
        
        return np.array(points, dtype=np.float32)
    
    def _simulate_rgb(self):
        """模拟RGB相机"""
        # 简化：随机生成图像
        # 实际应该使用渲染引擎（如PyBullet）
        rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # 在目标位置画一个绿色圆
        center = (64, 64)
        cv2.circle(rgb, center, 10, (0, 255, 0), -1)
        
        return rgb
    
    def _simulate_imu(self):
        """模拟IMU数据"""
        # 加速度（重力 + 运动加速度）
        gravity = np.array([0, 0, -9.81])
        acceleration = gravity + self.velocity / self.dt
        
        # 陀螺仪（角速度）
        gyro = self.angular_velocity
        
        # 添加噪声
        imu = np.concatenate([
            acceleration + np.random.randn(3) * 0.01,
            gyro + np.random.randn(3) * 0.01
        ])
        
        return imu.astype(np.float32)
    
    def _compute_reward(self):
        """计算奖励"""
        # 距离奖励
        dist_to_goal = np.linalg.norm(self.position - self.goal_position)
        distance_reward = -dist_to_goal / 10.0
        
        # 碰撞惩罚
        collision_penalty = 0
        for obs in self.obstacles:
            if np.linalg.norm(self.position - obs['pos']) < obs['radius']:
                collision_penalty = -100
                break
        
        # 速度惩罚（鼓励平滑）
        speed = np.linalg.norm(self.velocity)
        speed_penalty = -0.1 * speed
        
        # 到达奖励
        goal_reward = 100.0 if dist_to_goal < self.goal_radius else 0.0
        
        return distance_reward + collision_penalty + speed_penalty + goal_reward
    
    def _is_done(self):
        """检查终止条件"""
        # 碰撞
        for obs in self.obstacles:
            if np.linalg.norm(self.position - obs['pos']) < obs['radius']:
                return True, False
        
        # 到达目标
        if np.linalg.norm(self.position - self.goal_position) < self.goal_radius:
            return True, False
        
        # 超时
        if self.step_count >= self.max_steps:
            return False, True
        
        # 出界
        if np.any(np.abs(self.position) > 10):
            return True, False
        
        return False, False

# 测试
if __name__ == "__main__":
    env = UAVMultimodalEnv()
    env.reset()
    
    # 测试几步
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, done={done}")
        
        if done or truncated:
            break
    
    print("✅ 多模态UAV环境测试通过")
```

**验收标准**:
- [ ] 多模态环境创建成功
- [ ] Dict observation space正确
- [ ] 各传感器模拟合理
- [ ] 奖励函数合理

##### Day 49-50: Week 5总结
- [ ] Week 5工作日志整理
- [ ] SB3集成测试报告

#### Week 6: 训练与初步结果

##### Day 51-54: 完整SAC训练

```python
# train_uav_multimodal.py

from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.uav_multimodal_extractor import UAVMultimodalExtractor
import os

# 创建环境
env = UAVMultimodalEnv(max_steps=1000)

# 创建模型（使用自定义特征提取器）
model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    buffer_size=1000000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',
    tensorboard_log="./logs/uav_multimodal"
)

print("开始训练...")
model.learn(total_timesteps=100000)

# 保存模型
os.makedirs("models", exist_ok=True)
model.save("models/uav_multimodal_reliability_aware")

print("✅ 训练完成")
```

**验收标准**:
- [ ] 训练无错误
- [ ] TensorBoard日志正常
- [ ] 成功率 > 60%
- [ ] 训练曲线收敛

##### Day 55-56: 结果分析

```python
# analyze_results.py

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob

def analyze_training_logs(log_dir):
    """分析训练日志"""
    # 查找所有日志文件
    event_files = glob.glob(f"{log_dir}/*/events.out.tfevents.*")
    
    if not event_files:
        print("未找到日志文件")
        return
    
    # 加载日志
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    
    # 提取数据
    keys = ea.scalars.Keys()
    print(f"可用指标: {keys}")
    
    # 绘制训练曲线
    for key in ['rollout/ep_rew_mean', 'train/loss', 'train/entropy_loss']:
        if key in keys:
            events = ea.scalars.Items(key)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            plt.figure()
            plt.plot(steps, values)
            plt.xlabel('Steps')
            plt.ylabel(key)
            plt.title(f'Training Curve: {key}')
            plt.grid(True)
            plt.savefig(f"plots/{key.replace('/', '_')}.png")
            print(f"✅ 保存图表: plots/{key}.png")

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    
    analyze_training_logs("./logs/uav_multimodal")
    
    print("✅ 结果分析完成")
```

**验收标准**:
- [ ] 训练日志正确解析
- [ ] 训练曲线绘制成功
- [ ] 收敛情况分析完成

##### Day 57-60: Week 6总结与Week 7准备
- [ ] Week 6工作日志整理
- [ ] 初步实验结果报告
- [ ] Week 7消融实验设计

---

### 📅 Week 7-8: 实验验证与论文撰写

#### 阶段目标
✅ 完成消融实验
✅ 对比实验
✅ 论文初稿完成

#### Week 7: 消融实验

##### Day 61-63: 消融实验1 - 可靠性估计器

```python
# experiments/ablation_reliability_estimator.py

from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.uav_multimodal_extractor import UAVMultimodalExtractor
import torch

# 创建环境
env = UAVMultimodalEnv()

# 基线1：无可靠性估计
print("训练基线1：无可靠性估计")
model1 = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {"features_dim": 256, "use_reliability": False},
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/ablation_no_reliability"
)
model1.learn(total_timesteps=100000)

# 基线2：仅LiDAR可靠性
print("训练基线2：仅LiDAR可靠性")
model2 = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "reliability_modality": "lidar"
        },
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/ablation_lidar_only"
)
model2.learn(total_timesteps=100000)

# 完整方法
print("训练完整方法")
model3 = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {"features_dim": 256, "use_reliability": True},
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/ablation_full"
)
model3.learn(total_timesteps=100000)

print("✅ 消融实验1完成")
```

##### Day 64-66: 消融实验2 - 注意力头数

```python
# experiments/ablation_attention_heads.py

for num_heads in [2, 4, 8]:
    print(f"训练注意力头数={num_heads}的模型")
    
    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs={
            "features_extractor_class": UAVMultimodalExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "num_heads": num_heads,
                "use_reliability": True
            },
            "net_arch": [256, 256]
        },
        learning_rate=3e-4,
        tensorboard_log=f"./logs/ablation_heads_{num_heads}"
    )
    model.learn(total_timesteps=100000)

print("✅ 消融实验2完成")
```

##### Day 67-70: 消融实验3 - 动态权重 vs 固定权重

```python
# experiments/ablation_dynamic_vs_fixed.py

# 固定权重基线
print("训练固定权重基线")
model_fixed = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "use_reliability": True,
            "dynamic_weighting": False  # 固定权重
        },
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/ablation_fixed_weight"
)
model_fixed.learn(total_timesteps=100000)

# 动态权重方法
print("训练动态权重方法")
model_dynamic = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "use_reliability": True,
            "dynamic_weighting": True  # 动态权重
        },
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/ablation_dynamic_weight"
)
model_dynamic.learn(total_timesteps=100000)

print("✅ 消融实验3完成")
```

##### Day 71-74: 对比实验

```python
# experiments/comparison_baselines.py

# 对比方法1：FusedVisionNet基线
print("训练FusedVisionNet基线")
model_fused = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": FusedVisionNetExtractor,
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/comparison_fusedvision"
)
model_fused.learn(total_timesteps=100000)

# 对比方法2：FlatFusion基线
print("训练FlatFusion基线")
model_flat = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": FlatFusionExtractor,
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/comparison_flatfusion"
)
model_flat.learn(total_timesteps=100000)

# 本方法
print("训练本方法（可靠性感知融合）")
model_ours = SAC(
    "MultiInputPolicy",
    env,
    verbose=0,
    policy_kwargs={
        "features_extractor_class": UAVMultimodalExtractor,
        "features_extractor_kwargs": {"features_dim": 256, "use_reliability": True},
        "net_arch": [256, 256]
    },
    learning_rate=3e-4,
    tensorboard_log="./logs/comparison_ours"
)
model_ours.learn(total_timesteps=100000)

print("✅ 对比实验完成")
```

##### Day 75-77: 结果汇总与可视化

```python
# experiments/summarize_results.py

import numpy as np
import matplotlib.pyplot as plt

def summarize_experiments():
    """汇总所有实验结果"""
    
    methods = [
        ('No Reliability', './logs/ablation_no_reliability'),
        ('LiDAR Only', './logs/ablation_lidar_only'),
        ('Full Reliability', './logs/ablation_full'),
        ('Fixed Weight', './logs/ablation_fixed_weight'),
        ('Dynamic Weight', './logs/ablation_dynamic_weight'),
        ('FusedVisionNet', './logs/comparison_fusedvision'),
        ('FlatFusion', './logs/comparison_flatfusion'),
        ('Ours', './logs/comparison_ours')
    ]
    
    # 评估各方法（成功率、路径长度等）
    results = {}
    for name, log_dir in methods:
        # 这里应该实际运行评估
        results[name] = {
            'success_rate': np.random.uniform(0.5, 0.9),  # 示例
            'avg_path_length': np.random.uniform(20, 40),
            'inference_time': np.random.uniform(10, 30)
        }
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 成功率对比
    methods_names = list(results.keys())
    success_rates = [results[m]['success_rate'] for m in methods_names]
    axes[0].barh(methods_names, success_rates)
    axes[0].set_xlabel('Success Rate')
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_xlim(0, 1)
    
    # 路径长度对比
    path_lengths = [results[m]['avg_path_length'] for m in methods_names]
    axes[1].barh(methods_names, path_lengths)
    axes[1].set_xlabel('Average Path Length')
    axes[1].set_title('Path Length Comparison')
    
    # 推理时间对比
    inference_times = [results[m]['inference_time'] for m in methods_names]
    axes[2].barh(methods_names, inference_times)
    axes[2].set_xlabel('Inference Time (ms)')
    axes[2].set_title('Inference Time Comparison')
    
    plt.tight_layout()
    plt.savefig('plots/comparison_results.png')
    print("✅ 结果汇总完成")

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    summarize_experiments()
```

##### Day 78: Week 7总结
- [ ] Week 7工作日志整理
- [ ] 所有实验完成报告

#### Week 8: 论文撰写

##### Day 79-82: 论文初稿

创建论文结构：

```markdown
# Reliability-Aware Adaptive Fusion for UAV Multimodal Perception

## Abstract
150-200 words总结

## 1. Introduction
- UAV应用背景
- 多模态融合重要性
- 现有方法局限性
- 本文贡献

## 2. Related Work
- 多模态融合方法综述
- 可靠性感知研究
- 动态权重分配
- 本文创新性

## 3. Method
### 3.1 Problem Formulation
### 3.2 Reliability Estimation
#### 3.2.1 LiDAR SNR Estimator
#### 3.2.2 Image Quality Estimator
#### 3.2.3 IMU Consistency Checker
### 3.3 Dynamic Weighting
### 3.4 Adaptive Normalization
### 3.5 Complete Fusion Framework

## 4. Theoretical Analysis
### 4.1 Convergence
### 4.2 Robustness
### 4.3 Information Theory

## 5. Experiments
### 5.1 Setup
### 5.2 Ablation Studies
### 5.3 Comparison with Baselines
### 5.4 Results

## 6. Conclusion
```

**验收标准**:
- [ ] 论文结构完整
- [ ] 各章节内容充实
- [ ] 图表齐全

##### Day 83-86: 图表完善与润色

- [ ] 方法架构图
- [ ] 实验结果图
- [ ] 消融实验对比图
- [ ] 语言润色

##### Day 87-88: 投稿准备

- [ ] 检查格式要求
- [ ] 准备 supplementary material
- [ ] 检查参考文献
- [ ] 提交前最终检查

---

## 三、关键里程碑

| 里程碑 | 时间 | 描述 | 验收标准 |
|--------|------|------|---------|
| **M1** | Week 1 Day 7 | 环境搭建完成 | SB3多模态API测试通过 |
| **M2** | Week 2 Day 15 | 概念验证完成 | 2D环境SAC训练成功 |
| **M3** | Week 3 Day 30 | 可靠性估计器完成 | 3个估计器独立测试通过 |
| **M4** | Week 4 Day 42 | 融合模块完成 | 完整模块集成测试通过 |
| **M5** | Week 5 Day 50 | SB3集成完成 | 自定义特征提取器集成成功 |
| **M6** | Week 6 Day 60 | 初步训练完成 | 100k步训练成功，初步结果 |
| **M7** | Week 7 Day 78 | 实验完成 | 3组消融实验完成 |
| **M8** | Week 8 Day 88 | 论文初稿完成 | 论文初稿提交给导师 |

---

## 四、资源需求

### 4.1 计算资源

| 任务 | GPU需求 | 时间 | 备注 |
|------|---------|------|------|
| **Week 1-2** | CPU | - | 环境搭建，不需要GPU |
| **Week 3-4** | CPU | - | 模块开发，不需要GPU |
| **Week 5** | GPU (1x) | 2-3天 | SB3集成测试 |
| **Week 6** | GPU (1x) | 3-4天 | 100k步训练 |
| **Week 7** | GPU (4x) | 5-7天 | 8个实验并行训练 |
| **Week 8** | CPU | - | 论文撰写 |

**总计**: 需要1-2周GPU时间

### 4.2 存储空间

| 类型 | 大小 | 说明 |
|------|------|------|
| 数据集 | 50-100GB | UAVScenes数据集 |
| 模型检查点 | 10-20GB | 训练过程中的模型保存 |
| 日志文件 | 5-10GB | TensorBoard日志 |
| 代码和文档 | 2-5GB | 项目代码 |
| **总计** | **67-135GB** | 建议预留150GB |

### 4.3 人力投入

| 周次 | 时间投入 | 主要任务 |
|------|---------|---------|
| Week 1-2 | 100% | 环境搭建、概念验证 |
| Week 3-4 | 100% | 核心模块开发 |
| Week 5-6 | 100% | 系统集成、训练 |
| Week 7-8 | 100% | 实验、论文撰写 |

**总计**: 8周全职投入

---

## 五、风险管理

### 5.1 风险识别与缓解

| 风险 | 概率 | 影响 | 缓解措施 | 应急方案 |
|------|------|------|---------|---------|
| **SB3集成困难** | 中 | 高 | Week 2提前测试API | 使用自定义RL框架 |
| **训练不收敛** | 中 | 高 | 分阶段训练监控 | 降低任务复杂度 |
| **实验结果不理想** | 中 | 高 | 提前设计消融实验 | 聚焦理论贡献 |
| **时间不足** | 低 | 高 | 每周进度检查 | 减少消融实验数量 |
| **论文撰写时间紧张** | 中 | 中 | Week 7同时写论文 | 聚焦核心章节 |

### 5.2 进度监控

**每周检查**:
- 周一：上周总结 + 本周计划
- 周三：中期进度检查
- 周五：周末总结 + 下周准备

**关键决策点**:
- Week 4结束：是否进入系统集成分支
- Week 6结束：是否继续完整实验或简化
- Week 7结束：是否完成论文初稿

---

## 六、验收标准

### 6.1 技术验收

- [ ] 所有模块代码实现完成
- [ ] 代码通过类型检查（mypy）
- [ ] 代码通过linting（ruff）
- [ ] 单元测试覆盖率 > 80%
- [ ] 训练无错误
- [ ] 推理时间 < 30ms

### 6.2 实验验收

- [ ] 3组消融实验完成
- [ ] 至少2个对比基线
- [ ] 成功率提升 > 10%
- [ ] 鲁棒性提升 > 15%
- [ ] 结果可复现

### 6.3 文档验收

- [ ] API文档完整
- [ ] 实验设计文档完整
- [ ] 论文初稿完成
- [ ] 代码注释充分
- [ ] README更新

---

## 七、交付清单

### 7.1 代码交付

```
week1/idea1/
├── networks/
│   ├── reliability_estimators/
│   │   ├── lidar_snr_estimator.py
│   │   ├── image_quality_estimator.py
│   │   └── imu_consistency_checker.py
│   ├── reliability_predictor.py
│   ├── dynamic_weighting_layer.py
│   ├── adaptive_normalization.py
│   ├── reliability_aware_fusion.py
│   └── uav_multimodal_extractor.py
├── envs/
│   └── uav_multimodal_env.py
├── experiments/
│   ├── ablation_reliability_estimator.py
│   ├── ablation_attention_heads.py
│   ├── ablation_dynamic_vs_fixed.py
│   ├── comparison_baselines.py
│   └── summarize_results.py
├── train_uav_multimodal.py
├── analyze_results.py
└── README.md
```

### 7.2 文档交付

- [ ] IDEA1_FEASIBILITY_VERIFICATION.md ✅
- [ ] IDEA1_IMPLEMENTATION_PLAN.md (本文档) ✅
- [ ] IDEA1_TECHNICAL_DOCUMENTATION.md (待创建)
- [ ] IDEA1_ENVIRONMENT_SETUP.md (待创建)
- [ ] IDEA1_EXPERIMENTAL_DESIGN.md (待创建)
- [ ] IDEA1_API_DOCUMENTATION.md (待创建)
- [ ] 论文初稿.pdf

### 7.3 实验交付

- [ ] 训练日志（TensorBoard logs）
- [ ] 模型检查点
- [ ] 实验结果图表
- [ ] 消融实验报告
- [ ] 对比实验报告

---

## 八、下一步行动

### 立即开始（本周）

1. ✅ **环境搭建** - 创建conda环境，安装依赖
2. ✅ **SB3验证** - 测试多模态API
3. ✅ **数据准备** - 下载UAVScenes样本数据

### Week 1-2重点

- [ ] 完成2D简化环境
- [ ] 验证SAC基础训练
- [ ] 实现简化可靠性估计器

### Week 3-4重点

- [ ] 实现3个可靠性估计器
- [ ] 实现动态权重分配
- [ ] 完成融合模块

### Week 5-6重点

- [ ] 集成到SB3
- [ ] 完成100k步训练
- [ ] 分析初步结果

### Week 7-8重点

- [ ] 完成所有实验
- [ ] 撰写论文初稿
- [ ] 准备投稿

---

**文档版本**: v1.0
**创建时间**: 2026-01-22 23:45:00
**最后更新**: 2026-01-22 23:45:00
**审核状态**: 待审核
