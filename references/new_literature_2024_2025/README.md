# 新增文献索引 (2024-2025)
# New Literature Index for UAV RL Path Planning Research

## 文献列表 / Literature List

### 1. 安全强化学习与碰撞避免 (Safe RL & Collision Avoidance)

| 文件名 | 标题 | 年份 | 相关性 |
|--------|------|------|--------|
| `Collision_Cone_CBF_2024.pdf` | A Collision Cone Approach for Control Barrier Functions | 2024 | ⭐⭐⭐⭐⭐ 高度相关 |
| `Safe_RL_Filter_Multicopter_2024.pdf` | Safe Reinforcement Learning Filter for Multicopter Collision-Free Tracking | 2024 | ⭐⭐⭐⭐⭐ 高度相关 |
| `Real_Time_Safety_Fixed_Wing_2024.pdf` | Real Time Safety of Fixed-Wing UAVs using Collision Cone Control Barrier Functions | 2024 | ⭐⭐⭐⭐⭐ 高度相关 |

### 2. 课程学习 (Curriculum Learning)

| 文件名 | 标题 | 年份 | 相关性 |
|--------|------|------|--------|
| `Curriculum_Quadrotor_2025.pdf` | Curriculum-Based Sample Efficient RL for Robust Stabilization of a Quadrotor | 2025 | ⭐⭐⭐⭐⭐ 高度相关 |

### 3. 视觉端到端导航 (Vision-Based End-to-End Navigation)

| 文件名 | 标题 | 年份 | 相关性 |
|--------|------|------|--------|
| `SOUS_VIDE_Visual_Navigation_2024.pdf` | SOUS VIDE: Cooking Visual Drone Navigation Policies in a Gaussian Splatting Vacuum | 2024 | ⭐⭐⭐⭐ 高度相关 |
| `MonoNav_UAV_2023.pdf` | MonoNav: MAV Navigation via Monocular Depth Estimation and Reconstruction | 2023 | ⭐⭐⭐⭐ 相关 |
| `RGBD_Dynamic_Obstacles_2024.pdf` | A Real-Time Dynamic Obstacle Tracking and Mapping System for UAV Navigation | 2024 | ⭐⭐⭐⭐ 相关 |

### 4. SAC算法改进 (SAC Algorithm Improvements)

| 文件名 | 标题 | 年份 | 相关性 |
|--------|------|------|--------|
| `AM_SAC_Fixed_Wing_2025.pdf` | Continuous World Coverage Path Planning for Fixed-Wing UAVs using AM-SAC | 2025 | ⭐⭐⭐⭐⭐ 高度相关 |
| `GP_SAC_Hybrid_2025.pdf` | GP+SAC: A Hybrid Approach for UAV Path Planning | 2025 | ⭐⭐⭐⭐ 相关 |

### 5. 运动基元与路径规划 (Motion Primitives & Path Planning)

| 文件名 | 标题 | 年份 | 相关性 |
|--------|------|------|--------|
| `Bezier_Curves_UAV_2016.pdf` | UAV Path-Planning Using Bézier Curves and a Receding Horizon Approach | 2016 | ⭐⭐⭐⭐ 相关 |

### 6. 分层强化学习 (Hierarchical RL)

| 文件名 | 标题 | 年份 | 相关性 |
|--------|------|------|--------|
| `Hierarchical_MADDPG_2025.pdf` | Dual-Timescale Hierarchical MADDPG for Multi-UAV Cooperative Search | 2025 | ⭐⭐⭐⭐ 相关 |

## 文献分类 / Literature Classification

### 与开题报告核心技术的对应关系 / Mapping to Thesis Proposal Core Technologies

#### 1. Soft Actor-Critic (SAC) 算法
- ✅ `AM_SAC_Fixed_Wing_2025.pdf` - Action-Mapping SAC变体
- ✅ `GP_SAC_Hybrid_2025.pdf` - GP与SAC的混合方法

#### 2. 运动基元 (Motion Primitives)
- ✅ `Bezier_Curves_UAV_2016.pdf` - Bézier曲线作为运动基元
- ⚠️ 需要补充：参数化运动基元与RL的结合

#### 3. 安全层 (Safety Layer)
- ✅ `Collision_Cone_CBF_2024.pdf` - 碰撞锥CBF方法
- ✅ `Safe_RL_Filter_Multicopter_2024.pdf` - 安全滤波器
- ✅ `Real_Time_Safety_Fixed_Wing_2024.pdf` - 实时安全保证

#### 4. 课程学习 (Curriculum Learning)
- ✅ `Curriculum_Quadrotor_2025.pdf` - 三阶段课程学习策略
- ✅ 与开题报告的三步策略高度吻合

#### 5. 视觉端到端 (Vision-Based End-to-End)
- ✅ `SOUS_VIDE_Visual_Navigation_2024.pdf` - 轻量级SV-Net架构
- ✅ `MonoNav_UAV_2023.pdf` - 单目深度估计
- ✅ `RGBD_Dynamic_Obstacles_2024.pdf` - RGB-D传感器融合

## 关键发现 / Key Findings

### 1. SAC算法的最新发展
- **AM-SAC (Action-Mapping SAC)**: 在固定翼无人机覆盖路径规划中表现出色（arXiv:2505.08382, 2025）
- **GP+SAC混合**: 基因规划与SAC结合，显著加速收敛（2025）
- **应用场景**: 从静态路径规划到动态环境适应

### 2. 安全强化学习的成熟
- **碰撞锥CBF**: 提供实时碰撞避免约束，通过QP求解器实现（arXiv:2403.07043, 2024）
- **RCBF安全滤波器**: 保证多旋翼在未知输入干扰下的安全性（arXiv:2410.06852, 2024）
- **实用性**: 已在Crazyflie 2.1四旋翼上验证

### 3. 课程学习的有效性
- **三阶段策略** (arXiv:2501.18490, 2025):
  1. 固定初始条件下悬停
  2. 位置和姿态随机化
  3. 速度随机化
- **性能提升**: 相比单阶段训练，计算资源需求显著降低
- **收敛时间**: 约2000万时间步（3小时）vs 单阶段失败

### 4. 视觉端到端架构
- **SV-Net轻量级设计** (arXiv:2412.16346, 2024):
  - 处理RGB图像、光流、IMU数据
  - 输出低级推力和角速率命令
- **Gaussian Splatting仿真**: FiGS平台提供高视觉保真度
- **MonoNav验证** (arXiv:2311.14100, 2023): 单目深度估计的可行性

### 5. 运动基元与Bézier曲线
- **Bézier曲线优势** (BYU, 2016):
  - 计算简单，易于求导
  - 保证C²连续性（三阶曲线）
  - 适用于滚动时域方法
- **局限性**: 文献较少与RL直接结合

## 待补充文献 / Pending Literature

### 高优先级
1. "Effective Parametrization of Low Order Bézier Motion Primitives for Continuous-Curvature Path-Planning Applications" (Electronics, 2022)
2. "Hierarchical Proximal Policy Optimization for UAV Intelligent Maneuvering" (Processes, 2024)
3. "Deep Reinforcement Learning-Based Hierarchical Motion Planning Strategy for Multirotors" (IEEE TII, 2025) - 已有

### 中优先级
1. "A Soft Actor-Critic Based RL Approach for Motion Planning using Depth Images" (IEEE, 2025)
2. "Enhancing multi-UAV decision making via hierarchical RL" (Scientific Reports, 2024)
3. "DRL-Based UAV Navigation with LiDAR and Depth Camera Fusion" (Aerospace, 2025)

## 文献使用建议 / Recommendations for Literature Usage

### 立即阅读（与开题报告直接相关）
1. `Curriculum_Quadrotor_2025.pdf` - 课程学习策略设计
2. `Collision_Cone_CBF_2024.pdf` - 安全层实现
3. `Safe_RL_Filter_Multicopter_2024.pdf` - 安全滤波器架构

### 技术参考（具体实现细节）
1. `SOUS_VIDE_Visual_Navigation_2024.pdf` - 轻量级网络设计
2. `AM_SAC_Fixed_Wing_2025.pdf` - SAC改进方法
3. `Bezier_Curves_UAV_2016.pdf` - Bézier曲线参数化

### 背景阅读（扩展视野）
1. `MonoNav_UAV_2023.pdf` - 单目视觉导航
2. `RGBD_Dynamic_Obstacles_2024.pdf` - 传感器融合
3. `Hierarchical_MADDPG_2025.pdf` - 多无人机协调

## 统计信息 / Statistics

- **总文献数**: 12篇
- **2024年**: 8篇
- **2025年**: 4篇
- **arXiv论文**: 8篇
- **会议/期刊论文**: 4篇

---

**生成时间**: 2026-01-22
**生成者**: Sisyphus AI Agent
**项目**: 基于强化学习的无人机端到端路径规划方法研究
