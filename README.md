# PRISM - UAV Research Project

**项目**: 基于强化学习的无人机端到端路径规划方法研究
**创建日期**: 2026-01-22

---

## 📁 项目结构

```
PRISM/
├── stable-baselines3/              # 代码库 (强化学习算法实现)
├── references/                     # 参考文献库
│   ├── project_docs/               # 项目文档
│   ├── reviews/                    # 综述文献
│   ├── new_literature_2024_2025/   # 最新文献 (2024-2025)
│   └── multimodal_fusion_2024_2025/ # 多模态融合文献
└── .opencode/                     # OpenCode工作目录
```

---

## 📚 文献分类

### 1. 项目文档 (`references/project_docs/`)
- 开题报告 (23231037-苏英轲-开题.pdf)
- 实施计划 (IMPLEMENTATION_PLAN.md)
- 任务完成报告 (TASK_COMPLETION_REPORT.md)

### 2. 综述文献 (`references/reviews/`)
- UAV自主飞行机器学习综述 (3篇)
- 路径规划分类综述

### 3. 最新文献 (`references/new_literature_2024_2025/`)
**主题分类**:
- 安全强化学习与碰撞避免 (3篇)
- 课程学习 (1篇)
- 视觉端到端导航 (3篇)
- SAC算法改进 (2篇)
- 运动基元与路径规划 (1篇)
- 分层强化学习 (1篇)
- 论文摘要 (5篇)

**总计**: 16篇论文 + 5篇摘要

### 4. 多模态融合文献 (`references/multimodal_fusion_2024_2025/`)
**主题分类**:
- 数据集与基准 (1篇)
- Transformer融合框架 (3篇)
- 深度与LiDAR融合 (1篇)
- 动态权重分配 (2篇)
- 嵌入式实时系统 (4篇)
- 时序一致性 (1篇)
- 跨模态注意力 (3篇)
- 特定场景融合 (3篇)

**总计**: 18篇论文 + 18篇摘要

---

## 🔬 研究方向

### 核心技术
1. **Soft Actor-Critic (SAC) 算法**
   - AM-SAC (Action-Mapping SAC)
   - GP+SAC混合方法

2. **运动基元 (Motion Primitives)**
   - Bézier曲线参数化

3. **安全层 (Safety Layer)**
   - 碰撞锥CBF
   - 实时安全滤波器

4. **课程学习 (Curriculum Learning)**
   - 三阶段训练策略

5. **多模态融合**
   - LiDAR + RGB-D融合
   - 动态权重分配
   - Transformer架构

---

## 🚀 快速开始

### 文献阅读优先级

#### 高优先级 (必读)
- Curriculum-Based RL for Quadrotor (2025)
- Collision Cone CBF (2024)
- Safe RL Filter for Multicopter (2024)
- SaM²B - 可靠性感知融合 (2025)
- LSAF-LSTM - 动态权重分配 (2025)

#### 中优先级 (技术参考)
- FusedVisionNet - 跨注意力Transformer
- FlatFusion - Transformer设计选择
- GAFusion - 多重导引机制
- DMFusion - 深度与时序融合

#### 背景阅读
- UAV自主飞行综述 (reviews/)
- 路径规划分类综述 (reviews/)

---

## 📊 统计数据

- **总论文数**: 37篇
- **综述论文**: 3篇
- **最新文献 (2024-2025)**: 34篇
- **多模态融合研究**: 18篇
- **RL路径规划研究**: 16篇

---

## 🎯 研究目标

### 短期目标 (2个月)
- ✅ 文献综述完成
- ✅ 创新点分析
- ✅ 核心模块实现
- 📄 投稿IROS 2025或IEEE RA-L

### 中期目标 (4个月)
- ✅ 完整UAV多模态融合系统
- ✅ 大规模仿真实验
- 📄 投稿IEEE T-RO或ICRA

---

## 🔧 技术栈

### 仿真平台
- Flightmare
- AirSim
- PyBullet

### 深度学习
- PyTorch
- Stable-Baselines3

### 嵌入式平台
- NVIDIA Jetson Orin NX
- NVIDIA Jetson Xavier NX

### 可视化
- TensorBoard
- Weights & Biases

---

## 📞 相关资源

### 关键论文链接
- UAV-MM3D: https://arxiv.org/abs/2511.22404
- FlatFusion: https://arxiv.org/abs/2408.06832
- SaM²B: https://arxiv.org/abs/2512.24324
- FAST-LIVO2: https://arxiv.org/abs/2408.14035

### 代码库
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3

---

**最后更新**: 2026-01-22
**下次更新**: 完成核心模块实现后
