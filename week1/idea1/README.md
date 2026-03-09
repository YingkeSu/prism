# Idea1 - 多维度可靠性感知的自适应融合

## 项目简介

本项目实现了一个针对UAV导航的多模态传感器融合框架，通过可靠性感知机制自适应地融合LiDAR、RGB和IMU数据，提升强化学习在复杂环境下的鲁棒性和性能。

## 项目结构

```
week1/idea1/
├── docs/                    # 文档目录
│   ├── idea1/              # Idea1相关文档
│   │   ├── chinese_version.md              # 中文版本详细方案
│   │   ├── IDEA1_API_DOCUMENTATION.md      # API接口文档
│   │   ├── IDEA1_BASELINE_COMPARISON.md    # 基线对比文档
│   │   ├── IDEA1_DELIVERY_SUMMARY.md       # 交付物总结
│   │   ├── IDEA1_ENVIRONMENT_SETUP.md      # 环境配置文档
│   │   ├── IDEA1_EXPERIMENTAL_DESIGN.md    # 实验设计文档
│   │   ├── IDEA1_FEASIBILITY_VERIFICATION.md  # 可行性验证报告
│   │   ├── IDEA1_IMPLEMENTATION_PLAN.md    # 实施方案（8周计划）
│   │   ├── IDEA1_OPTIMIZATION_SUMMARY_CN.md # 优化总结（中文）
│   │   ├── IDEA1_OPTIMIZATION_VERIFICATION.md # 优化验证报告
│   │   ├── RGB_ENCODER_OPTIMIZATION.md     # RGB编码器优化报告
│   │   └── IDEA1_TECHNICAL_DOCUMENTATION.md  # 技术实现文档
│   ├── notebooklm/         # NotebookLM源文档（英文）
│   │   ├── 00_paper_onepager.md
│   │   ├── 01_contributions.md
│   │   ├── 02_system_overview.md
│   │   ├── 03_architecture_blocks.md
│   │   ├── 04_datasets.md
│   │   ├── 05_metrics.md
│   │   ├── 06_baselines.md
│   │   ├── 07_training_and_inference.md
│   │   ├── 08_results_summary.md
│   │   ├── 09_reproducibility.md
│   │   ├── 10_experiment_log.md
│   │   ├── 11_todo_and_risks.md
│   │   ├── 12_innovation_and_literature_review.md
│   │   ├── 13_mathematical_proofs.md
│   │   └── README.md
│   ├── process/            # 过程性文档
│   │   ├── TRAINING_HANG_FIX_REPORT.md     # 训练卡死问题修复报告
│   │   └── VISUALIZATION_IMPLEMENTATION_REPORT.md # 可视化实施报告
│   ├── project/            # 项目文档
│   │   ├── progress_report.md              # 项目进展报告
│   │   └── README.md
│   └── AGENTS.md           # AGENTS开发指南
│
├── tests/                   # 测试目录
│   ├── training/           # 训练测试
│   │   ├── train_minimal_test.py           # 最小化训练测试
│   │   ├── train_ultra_simple.py          # 极简训练脚本
│   │   └── train_epoch_10.py              # 10个epoch训练
│   └── integration/        # 集成测试
│       ├── demo.py                        # 完整demo脚本
│       └── test_basic_setup.py            # 基础设置测试
│
├── networks/               # 核心网络模块
│   ├── reliability_estimators/  # 可靠性估计器
│   │   ├── __init__.py
│   │   ├── lidar_snr_estimator.py       # LiDAR信噪比估计
│   │   ├── image_quality_estimator.py    # 图像质量评估
│   │   └── imu_consistency_checker.py  # IMU一致性检查
│   ├── __init__.py
│   ├── reliability_predictor.py         # 可靠性预测网络
│   ├── dynamic_weighting_layer.py       # 动态权重分配层
│   ├── adaptive_normalization.py        # 自适应归一化层
│   ├── reliability_aware_fusion.py      # 完整融合模块
│   └── uav_multimodal_extractor.py     # SB3自定义特征提取器
│
├── envs/                   # 环境模块
│   ├── __init__.py
│   ├── uav_multimodal_env.py            # 多模态UAV环境
│   └── simple_2d_env.py                # 2D简化环境
│
├── utils/                  # 工具函数（待填充）
└── experiments/            # 实验脚本（待填充）
```

## 快速开始

### 环境配置

```bash
# 创建conda环境
conda create -n sb3_idea1 python=3.10 -y

# 激活环境
conda activate sb3_idea1

# 安装依赖
pip install torch>=2.3,<3.0
pip install stable-baselines3>=2.8.0
pip install numpy>=1.20,<3.0
pip install gymnasium>=0.29.1,<1.3.0
```

详细配置说明请参考：[环境配置文档](docs/idea1/IDEA1_ENVIRONMENT_SETUP.md)

### 运行测试

```bash
# 测试UAV环境
python envs/uav_multimodal_env.py

# 测试融合模块
python -m networks.reliability_aware_fusion

# 运行基础设置测试
python tests/integration/test_basic_setup.py

# 运行完整demo
python tests/integration/demo.py
```

### 运行训练

```bash
# 推荐：统一训练入口（支持可靠性开关与SAC超参）
python train.py --timesteps 1000 --encoder-type gap --save

# 退化场景训练（用于鲁棒性验证）
python train.py --timesteps 1000 --encoder-type gap --degradation-level 0.5

# 提高任务难度（避免评估全满分）
python train.py --timesteps 1000 --encoder-type gap --difficulty hard --degradation-level 0.5

# 调整每回合时间预算（hard场景建议先尝试400）
python train.py --timesteps 1000 --encoder-type gap --difficulty hard --degradation-level 0.5 --max-steps 400

# 若需要可视化与详细调试日志
python train.py --timesteps 1000 --encoder-type gap --debug

# 最小化训练测试（10步）
python tests/training/train_minimal_test.py

# 极简训练脚本
python tests/training/train_ultra_simple.py

# 训练10个epoch
python tests/training/train_epoch_10.py

# 一键运行对比/消融实验套件（ours/no_reliability/fixed_equal）
python experiments/run_suite.py --timesteps 1000 --eval-episodes 50 --seeds 42,43,44

# 加入“关闭可靠性调制归一化”消融 + 退化场景
python experiments/run_suite.py --timesteps 1000 --eval-episodes 50 --seeds 42,43,44 --degradation-level 0.5 --include-norm-ablation

# 训练加速（共享特征提取器 + 更短IMU历史窗口）
python train.py --timesteps 1000 --encoder-type gap --imu-history-len 16

# 可靠性响应检查（退化强度↑时分数是否下降）
python experiments/check_reliability_response.py --levels 0.0,0.2,0.5,0.8 --samples-per-level 128

# 将套件结果按方法聚合为均值±方差
python experiments/summarize_suite.py --input-csv logs/experiments/suite/suite_summary.csv
```

## 核心模块说明

### 1. 可靠性估计器（Reliability Estimators）

- **LiDAR SNR估计器**: 评估点云信噪比和密度
- **图像质量评估器**: 评估图像的锐度、对比度和整体质量
- **IMU一致性检查器**: 检查加速度计和陀螺仪的一致性

### 2. 可靠性预测网络（Reliability Predictor）

融合三个估计器的质量指标，输出每个模态的可靠性分数（0-1），并生成融合特征。

### 3. 动态权重分配层（Dynamic Weighting Layer）

基于多头注意力机制，动态计算LiDAR、RGB、IMU的融合权重。

### 4. 自适应归一化层（Adaptive Normalization）

根据模态的可靠性分数，自适应地调整特征的归一化程度。

### 5. 可靠性感知融合模块（Reliability Aware Fusion）

完整的端到端融合模块，集成所有子模块，输出融合后的特征和中间结果。

### 6. UAV多模态特征提取器（UAV Multimodal Extractor）

Stable-Baselines3兼容的自定义特征提取器，用于RL训练。

## 技术指标

| 指标 | 设计目标 | 当前状态 |
|------|---------|---------|
| 参数量（核心模块） | <500K | **238.5K** ✅ |
| 参数量（融合模块） | <500K | 806.2K ⚠️ (超目标60%) |
| 推理时间 | <30ms | 预计5-10ms |
| 模态数量 | 3个 | ✅ 完成 |
| 注意力头数 | 8 | ✅ 完成 |

## 文档索引

- **项目进展报告**: [progress_report.md](docs/project/progress_report.md)
- **可行性验证**: [IDEA1_FEASIBILITY_VERIFICATION.md](docs/idea1/IDEA1_FEASIBILITY_VERIFICATION.md)
- **实施方案**: [IDEA1_IMPLEMENTATION_PLAN.md](docs/idea1/IDE1_IMPLEMENTATION_PLAN.md)
- **技术文档**: [IDEA1_TECHNICAL_DOCUMENTATION.md](docs/idea1/IDEA1_TECHNICAL_DOCUMENTATION.md)
- **API文档**: [IDEA1_API_DOCUMENTATION.md](docs/idea1/IDEA1_API_DOCUMENTATION.md)
- **基线对比**: [IDEA1_BASELINE_COMPARISON.md](docs/idea1/IDEA1_BASELINE_COMPARISON.md)
- **环境配置**: [IDEA1_ENVIRONMENT_SETUP.md](docs/idea1/IDEA1_ENVIRONMENT_SETUP.md)
- **实验设计**: [IDEA1_EXPERIMENTAL_DESIGN.md](docs/idea1/IDEA1_EXPERIMENTAL_DESIGN.md)
- **实验套件指南**: [EXPERIMENT_SUITE_GUIDE.md](docs/idea1/EXPERIMENT_SUITE_GUIDE.md)
- **性能优化验证**: [IDEA1_OPTIMIZATION_VERIFICATION.md](docs/idea1/IDEA1_OPTIMIZATION_VERIFICATION.md) ← **NEW**
- **AGENTS指南**: [AGENTS.md](docs/AGENTS.md)

## 当前进度

**总体进度**: ~75% (Week 1-5基本完成，质量良好) ✅ **+性能优化完成 (2026-02-04)**

### 已完成 ✅
- 所有核心模块实现（10个Python文件，~2752行代码）
- 完整文档体系（13个文档，含性能优化报告）
- 环境配置和SB3集成完成
- SB3集成测试100%通过（test_sb3_integration.py）
- 基础测试（环境、融合模块、自适应归一化）
- ReliabilityPredictor参数量满足<500K目标（238.5K）
- **🔥 性能优化完成**（200步: 1.65s, 7.8ms/step）← **NEW 2026-02-04**

### 待完成 ⏳
- 训练稳定性验证（10k-100k步）
- 消融实验（3组）
- 对比实验（4个基线）
- 论文撰写

详细进展请参考：[项目进展报告](docs/project/progress_report.md)

## 开发指南

请参考 [AGENTS.md](docs/AGENTS.md) 了解代码规范、测试方法和开发流程。

## 论文投稿计划

- **目标会议**: IROS 2025
- **截稿日期**: ~2025年9月15日
- **备选**: IEEE RA-L（滚动投稿）

## 联系方式

如有问题，请参考项目文档或联系项目负责人。

---

**项目版本**: v1.1
**最后更新**: 2026-02-04
**状态**: 开发中 (Week 1-5完成，性能优化完成，训练与实验阶段)
