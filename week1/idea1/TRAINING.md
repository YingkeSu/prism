# 训练工作流程指南

**最后更新**: 2026-02-04

## 快速开始

### 基础训练（100个时间步）
```bash
PYTHONPATH=/Users/suyingke/Programs/PRISM/week1/idea1 python train.py --timesteps 100
```

### 带模型保存的训练
```bash
PYTHONPATH=/Users/suyingke/Programs/PRISM/week1/idea1 python train.py --timesteps 100 --save
```

### 测试已训练模型
```bash
PYTHONPATH=/Users/suyingke/Programs/PRISM/week1/idea1 python train.py --test models/idea1_model_100ts.zip --episodes 5
```

## 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--timesteps N` | 训练的时间步数 | 100 |
| `--no-reliability` | 禁用可靠性感知融合（使用基线） | False |
| `--save` | 将训练好的模型保存到 `models/` 目录 | False |
| `--test PATH` | 测试指定路径的已保存模型 | None |
| `--episodes N` | 测试回合数（与 `--test` 配合使用） | 5 |
| `--encoder-type TYPE` | RGB编码器类型（baseline/gap/mobilenet） | gap |
| `--no-debug` | 禁用调试输出（减少日志信息） | False（默认启用调试）|

## 训练稳定性

### 已验证结果（100个时间步）

**执行了10次连续试验：**
- 成功率：10/10（100%）
- 平均时间：0.81秒
- 标准差：0.01秒
- 最短时间：0.80秒
- 最长时间：0.81秒

**结论**：在100个时间步的训练非常稳定，方差极小。

### 训练稳定性验证（2026-01-28更新）

**测试结果**：
- `tests/training/train_epoch_10.py` 在CPU上运行1000 timesteps
- 超过10分钟未完成（CPU训练性能瓶颈）
- 建议：使用GPU或云服务器进行大规模训练

**已知修复问题**：
- ✅ RGB通道顺序问题已修复（NHWC → NCHW）
- ✅ IMU维度扩展问题已修复（使用repeat而非expand）
- ✅ 固定权重链路已修复
- ✅ test_sb3_integration.py 100%通过

**下一步验证计划**：
```bash
# 1. 在GPU上运行训练稳定性测试
python tests/training/train_epoch_10.py

# 2. 监控TensorBoard日志
tensorboard --logdir ./logs

# 3. 验证Loss曲线和成功率趋势
```

## 训练模式

### 可靠性感知融合（默认模式）
使用完整的可靠性感知融合模块，包含动态权重分配。

```bash
# 使用默认GAP编码器（推荐，~102K参数）
python train.py --timesteps 100

# 使用MobileNetV3编码器（~1.08M参数）
python train.py --timesteps 100 --encoder-type mobilenet

# 使用Baseline编码器（16.9M参数）
python train.py --timesteps 100 --encoder-type baseline
```

### 基线模式（无可靠性）
使用简单的拼接，不包含可靠性感知融合。

```bash
python train.py --timesteps 100 --no-reliability
```

## 模型架构

- **基础算法**：SAC（Soft Actor-Critic）
- **策略**：MultiInputPolicy（用于多模态观测）
- **特征提取器**：UAVMultimodalExtractor（支持多种编码器）
- **输出维度**：256

### RGB编码器选项

| 编码器类型 | 参数量 | 说明 | 速度估计 |
|-----------|--------|------|----------|
| **baseline** | 16.9M | 原始完整编码器 | ~10分钟/1000步 |
| **gap** | ~102K | 全局平均池化编码器（推荐） | ~20-40秒/1000步 |
| **mobilenet** | ~1.08M | 轻量级MobileNetV3 | ~15-25秒/1000步 |

### 调试模式

默认启用调试模式（`debug=True`），会输出详细的训练信息。使用 `--no-debug` 可以减少日志输出。

## 环境详情

### 观测空间
- **LiDAR**：(1000, 3) float32 - 点云数据
- **RGB**：(128, 128, 3) uint8 - 摄像机图像
- **IMU**：(6,) float32 - 惯性测量单元数据

### 动作空间
- **形状**：(4,) float32
- **范围**：[-1, 1]
- **分量**：[vx, vy, vz, omega] - 速度和角速度

## 故障排除

### 训练看似卡住或非常缓慢

当前环境设置如下（以代码为准）：
- 智能体从原点(0,0,0)出发
- 目标位于(3,3,2)，目标半径为1.5
- 回合在达成目标/碰撞/越界或达到最大步数(200)后结束

**常见原因与解决方案**：
1. 日志过多导致感知“卡住”：使用 `--no-debug` 减少日志输出
2. 确保 `verbose=0`（train.py 默认值）以避免SB3详细日志拖慢训练
3. 本地仅做小规模训练（100-500步），长程训练转到GPU

### CPU训练超时

**症状**：CPU上运行1000 timesteps超过10分钟未完成

**原因**：当前MacBook M2无独立GPU，训练速度受限

**解决方案**：
1. 使用云服务器（AWS/Azure/Colab Pro）进行GPU训练
2. 本地调试使用小规模训练（100-500步）
3. 大规模训练（10k-100k步）使用GPU

**推荐GPU资源**：
- AWS: p3.2xlarge (V100) 或 p4d.24xlarge (A100)
- Azure: Standard_NC6s_v3 (V100)
- Google Colab Pro: T4或V100 GPU

### 导入错误

如果遇到 `ModuleNotFoundError`：
```bash
export PYTHONPATH=/Users/suyingke/Programs/PRISM/week1/idea1:$PYTHONPATH
```

### TensorBoard

训练日志保存在 `logs/sb3/` 目录。使用以下命令查看：
```bash
tensorboard --logdir logs/sb3
```

## 已知问题

1. **verbose > 0 导致超时**：SB3 2.7.1在MultiInputPolicy和详细日志方面存在问题。使用 `verbose=0`（train.py中的默认值）。

2. **长时间训练会话**：单回合上限为200步，训练耗时基本随`timesteps`线性增长；长时训练建议使用GPU。

3. **CPU训练性能瓶颈**（新增）：当前硬件（MacBook M2）无独立GPU，大规模训练（>1000步）需要数小时甚至更久。

4. **已修复问题**：
   - ✅ RGB通道顺序不匹配（已修复，NHWC → NCHW）
   - ✅ IMU维度扩展失败（已修复，使用repeat而非expand）
   - ✅ SB3特征提取器集成问题（已修复，test_sb3_integration.py 100%通过）

## 性能指标

| 时间步 | 平均时间 | 每时间步时间 | 备注 |
|--------|---------|-------------|------|
| 10 | 2.29秒 | 229毫秒 | 最快（最小学习） |
| 100 | 0.81秒 | 8毫秒 | 推荐用于快速测试 |
| 1000 | ~8秒（估计） | 8毫秒 | 更长时间训练 |

## 文件结构

```
idea1/
├── train.py                          # 主训练入口
├── envs/
│   └── uav_multimodal_env.py     # 训练环境
├── networks/
│   └── uav_multimodal_extractor.py  # 特征提取器
├── models/                            # 已保存模型（使用--save时创建）
└── logs/sb3/                          # TensorBoard日志
```

## GPU资源配置

### 推荐配置

| 平台 | GPU类型 | vCPU | 内存 | 预估成本 | 适用场景 |
|------|--------|------|------|---------|---------|
| **AWS** | p3.2xlarge (V100) | 8 | 61 GB | ~$3.06/小时 | 中等规模训练 |
| **AWS** | p4d.24xlarge (A100) | 96 | 1.1 TB | ~$32.77/小时 | 大规模训练 |
| **Azure** | Standard_NC6s_v3 (V100) | 6 | 112 GB | ~$3.00/小时 | 中等规模训练 |
| **Colab Pro** | T4/V100 | - | - | $10/月 | 开发和调试 |

### 快速启动GPU训练

```bash
# 1. 在GPU服务器上安装环境
conda create -n sb3_idea1 python=3.10 -y
conda activate sb3_idea1
pip install torch>=2.3,<3.0 --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3>=2.8.0
pip install numpy>=1.20,<3.0
pip install gymnasium>=0.29.1,<1.3.0

# 2. 上传代码到GPU服务器
rsync -avz /Users/suyingke/Programs/PRISM/week1/idea1 user@server:/path/to/project

# 3. 运行大规模训练
python train.py --timesteps 10000 --save

# 4. 下载训练好的模型和日志
rsync -avz user@server:/path/to/project/models/ ./models/
rsync -avz user@server:/path/to/project/logs/ ./logs/
```

## 训练技巧

1. **从小开始**：首先用10-100个时间步测试
2. **监控进度**：使用TensorBoard跟踪学习过程
3. **保存模型**：使用 `--save` 标志设置检查点
4. **比较模式**：运行带有和不带 `--no-reliability` 的训练
5. **充分测试**：使用 `--test` 评估训练好的模型

## 示例工作流程

```bash
# 1. 训练基线模型
python train.py --timesteps 100 --no-reliability --save

# 2. 训练可靠性感知模型
python train.py --timesteps 100 --save

# 3. 测试两个模型
python train.py --test models/idea1_model_100ts.zip --episodes 10
python train.py --test models/idea1_model_100ts.zip --episodes 10 --no-reliability

# 4. 比较结果
# 检查两次运行的TensorBoard日志
```
