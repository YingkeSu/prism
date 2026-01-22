# Idea1 实验设计文档

**创建日期**: 2026-01-23
**项目**: PRISM - UAV Research
**创新点**: 多维度可靠性感知的自适应融合 (Idea1)

---

## 执行摘要

本文档提供Idea1的完整实验设计，包括消融实验、对比实验、评估指标、结果分析和统计方法。所有实验设计均遵循科学严谨性，确保结果的可比性和可复现性。

---

## 一、实验目标

### 1.1 核心验证目标

| 目标 | 描述 | 成功标准 |
|------|------|---------|
| **有效性验证** | 可靠性感知融合是否优于固定权重 | 成功率提升 >10% |
| **鲁棒性验证** | 在不同传感器质量条件下的性能鲁棒性 | 鲁棒性提升 >15% |
| **实时性验证** | 推理延迟和计算开销 | 推理时间 <30ms |
| **消融验证** | 各组件的有效性 | 每个组件显著贡献 |

### 1.2 预期性能指标

| 指标 | 基线 (Fixed Weight) | 目标 (Idea1) | 提升幅度 |
|------|-------------------|--------------|---------|
| **Success Rate** | 70-75% | 80-85% | +10-15% |
| **Path Length** | 140-150 steps | 120-130 steps | -10-15% |
| **Collision Rate** | 10-12% | 6-8% | -40% |
| **Convergence Speed** | 80k steps | 50k steps | +40% |
| **Inference Time** | 5-10 ms | 20-28 ms | <30ms ✓ |

---

## 二、实验环境与配置

### 2.1 训练环境

```python
# 训练配置 (Week 3-6)

TRAIN_CONFIG = {
    # 环境配置
    'environment': 'UAVMultimodalEnv',
    'observation_space': {
        'lidar': {'shape': (1000, 3), 'type': 'float32'},
        'rgb': {'shape': (128, 128, 3), 'type': 'uint8'},
        'imu': {'shape': (6,), 'type': 'float32'}
    },
    'action_space': {'shape': (4,), 'type': 'float32', 'range': [-1, 1]},
    
    # 算法配置
    'algorithm': 'SAC',
    'policy': 'MultiInputPolicy',
    'features_extractor_class': 'UAVMultimodalExtractor',
    'features_extractor_kwargs': {
        'use_reliability': True,
        'num_heads': 8
    },
    'net_arch': [256, 256],
    
    # 训练超参数
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 1000000,
    'learning_starts': 1000,
    'train_freq': 1,
    'gradient_steps': 1,
    'tau': 0.005,
    'gamma': 0.99,
    'ent_coef': 'auto',
    
    # 总训练步数
    'total_timesteps': 100000,
    
    # 日志配置
    'tensorboard_log': './logs/experiment_name',
    'log_interval': 1000,
    'save_interval': 10000,
}
```

### 2.2 评估环境

```python
# 评估配置 (Week 7)

EVAL_CONFIG = {
    # 评估集
    'num_episodes': 100,
    'deterministic': True,  # 确定性评估
    'random_seeds': [0, 42, 2024],  # 多个随机种子
    'max_steps': 1000,
    
    # 成功阈值
    'success_threshold': 0.1,  # 距离目标<0.1m
    'collision_threshold': 0.5,  # 碰撞距离<0.5m
    
    # 评估指标
    'metrics': [
        'success_rate',      # 成功率
        'path_length',       # 平均路径长度
        'collision_rate',    # 碰撞率
        'inference_time',    # 平均推理时间
        'jerk_cost',        # 轨迹平滑度
    ],
    
    # 记录
    'record_video': True,
    'record_trajectory': True,
    'save_metrics': True,
}
```

---

## 三、消融实验设计

### 3.1 实验组1: 可靠性估计器有效性

**目标**: 验证各模态可靠性估计器的有效性

| 实验ID | 配置 | LiDAR估计 | RGB估计 | IMU估计 | 预期结果 |
|--------|------|-----------|----------|----------|---------|
| **E1-1** | None (无可靠性) | ❌ | ❌ | ❌ | 基线性能 |
| **E1-2** | LiDAR Only | ✅ | ❌ | ❌ | 中等提升 |
| **E1-3** | RGB Only | ❌ | ✅ | ❌ | 中等提升 |
| **E1-4** | IMU Only | ❌ | ❌ | ✅ | 小幅提升 |
| **E1-5** | Full Reliability (Ours) | ✅ | ✅ | ✅ | **最大提升** |

**假设检验**:
- H0: 完整可靠性估计与部分估计无显著差异
- H1: 完整可靠性估计显著优于部分估计
- 显著性水平: α = 0.05

**预期结论**: E1-5 > E1-4 > E1-3 ≈ E1-2 > E1-1

---

### 3.2 实验组2: 注意力头数影响

**目标**: 确定最优注意力头数

| 实验ID | 注意力头数 | 参数量 | 训练时间 | 预期性能 |
|--------|-----------|--------|---------|---------|
| **E2-1** | 2 heads | ~200K | 短 | 快速但欠拟合 |
| **E2-2** | 4 heads | ~300K | 中等 | 平衡性能 |
| **E2-3** | 8 heads | ~500K | 长 | **最佳性能** |
| **E2-4** | 16 heads | ~900K | 长 | 慢速但过拟合风险 |

**假设检验**:
- H0: 8头与4头无显著差异
- H1: 8头显著优于4头
- 显著性水平: α = 0.05

**预期结论**: E2-3最佳，E2-4过拟合风险高

---

### 3.3 实验组3: 动态权重 vs 固定权重

**目标**: 验证动态权重分配的有效性

| 实验ID | 权重类型 | 机制 | 预期结果 |
|--------|---------|------|---------|
| **E3-1** | Fixed Weight | 简单平均融合 | 基线性能 |
| **E3-2** | Fixed Weight | 预定义比例 (0.4, 0.4, 0.2) | 略优于E3-1 |
| **E3-3** | Dynamic Weight (Ours) | 注意力驱动 | **最优性能** |

**变体实验** (可选):
- **E3-3a**: 仅注意力权重（无温度缩放）
- **E3-3b**: 注意力权重 + 温度缩放
- **E3-3c**: 注意力权重 + 温度缩放 + 偏置

**预期结论**: E3-3c > E3-3b > E3-3a > E3-2 > E3-1

---

## 四、对比实验设计

### 4.1 基线方法对比

| 方法 | 类型 | 论文 | 关键特性 | 预期性能 |
|------|------|------|---------|---------|
| **Fixed Weight** | 基线 | - | 简单平均 | 70-75% |
| **FusedVisionNet** | 基线 | IJIR 2025 | 跨注意力Transformer, 34 FPS | 75-80% |
| **FlatFusion** | 基线 | ICRA 2025 | 稀疏Transformer | 76-81% |
| **Idea1 (Ours)** | 核心 | - | 可靠性感知自适应融合 | **80-85%** |

### 4.2 实验协议

**训练阶段**:
```python
# experiments/comparison_baselines.py

BASELINES = [
    ('FixedWeight', 'FixedWeightExtractor', {}),
    ('FusedVisionNet', 'FusedVisionNetExtractor', {}),
    ('FlatFusion', 'FlatFusionExtractor', {}),
    ('Idea1', 'UAVMultimodalExtractor', {'use_reliability': True, 'num_heads': 8})
]

for baseline_name, extractor_class, kwargs in BASELINES:
    print(f"\n{'='*60}")
    print(f"训练基线: {baseline_name}")
    print(f"{'='*60}")
    
    # 创建模型
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": kwargs,
            "net_arch": [256, 256]
        },
        learning_rate=3e-4,
        tensorboard_log=f"./logs/comparison_{baseline_name.lower()}"
    )
    
    # 训练
    model.learn(total_timesteps=100000)
    
    # 评估
    results = evaluate_baseline(model, env, num_episodes=100)
    
    # 保存结果
    save_results(baseline_name, results)
```

**评估阶段**:
```python
# utils/metrics.py

def compute_metrics(trajectories, actions, observations):
    """
    计算所有评估指标
    
    Args:
        trajectories: List of (T, 7) 轨迹 (x,y,z,roll,pitch,yaw,v)
        actions: List of (T, 4) 动作
        observations: List of (T, 3) 观测
    
    Returns:
        Dict: 所有评估指标
    """
    results = {}
    
    # 1. Success Rate
    success_count = sum(1 for t in trajectories if is_successful(t))
    results['success_rate'] = success_count / len(trajectories)
    
    # 2. Path Length
    path_lengths = [compute_path_length(t) for t in trajectories]
    results['avg_path_length'] = np.mean(path_lengths)
    results['std_path_length'] = np.std(path_lengths)
    
    # 3. Collision Rate
    collision_count = sum(1 for t in trajectories if has_collision(t))
    results['collision_rate'] = collision_count / len(trajectories)
    
    # 4. Jerk Cost (平滑度)
    jerk_costs = [compute_jerk_cost(a) for a in actions]
    results['avg_jerk_cost'] = np.mean(jerk_costs)
    
    # 5. Inference Time
    # 需要在运行时测量
    results['avg_inference_time'] = compute_avg_inference_time()
    
    return results
```

---

## 五、评估指标详解

### 5.1 主要指标

#### 5.1.1 成功率 (Success Rate)

**定义**: 成功到达目标的比例

**计算**:
```python
def is_successful(trajectory, goal_position, success_threshold=0.1):
    """
    判断轨迹是否成功
    
    Args:
        trajectory: (T, 7) 轨迹
        goal_position: (3,) 目标位置
        success_threshold: 成功阈值（米）
    
    Returns:
        bool: 是否成功
    """
    final_position = trajectory[-1, :3]  # 最后位置
    distance_to_goal = np.linalg.norm(final_position - goal_position)
    return distance_to_goal < success_threshold and not has_collision(trajectory)
```

**目标**: >80%

#### 5.1.2 路径长度 (Path Length)

**定义**: 从起点到目标点的路径长度

**计算**:
```python
def compute_path_length(trajectory):
    """
    计算路径长度
    
    Args:
        trajectory: (T, 7) 轨迹
    
    Returns:
        float: 路径长度
    """
    positions = trajectory[:, :3]  # (T, 3)
    path_length = 0.0
    for i in range(1, len(positions)):
        path_length += np.linalg.norm(positions[i] - positions[i-1])
    return path_length
```

**目标**: <130 steps (比基线短10-15%)

#### 5.1.3 碰撞率 (Collision Rate)

**定义**: 发生碰撞的轨迹比例

**计算**:
```python
def has_collision(trajectory, obstacles, collision_threshold=0.5):
    """
    判断轨迹是否碰撞
    
    Args:
        trajectory: (T, 7) 轨迹
        obstacles: List of {'pos': (3,), 'radius': float}
        collision_threshold: 碰撞阈值（米）
    
    Returns:
        bool: 是否碰撞
    """
    for t in range(len(trajectory)):
        position = trajectory[t, :3]
        for obs in obstacles:
            distance = np.linalg.norm(position - obs['pos'])
            if distance < obs['radius'] + collision_threshold:
                return True
    return False
```

**目标**: <8% (比基线低40%)

### 5.2 次要指标

#### 5.2.1 收敛速度 (Convergence Speed)

**定义**: 训练达到目标成功率所需的步数

**计算**:
```python
def compute_convergence_speed(success_rates, target=0.8):
    """
    计算收敛速度
    
    Args:
        success_rates: List of (step, success_rate)
        target: 目标成功率（默认0.8）
    
    Returns:
        int: 收敛步数
    """
    for step, rate in success_rates:
        if rate >= target:
            return step
    return -1  # 未收敛
```

**目标**: <50k steps (比基线快40%)

#### 5.2.2 推理时间 (Inference Time)

**定义**: 单步推理的平均时间

**计算**:
```python
import time

def measure_inference_time(model, env, num_samples=1000):
    """
    测量推理时间
    
    Args:
        model: 训练好的模型
        env: 环境
        num_samples: 采样数
    
    Returns:
        Dict: 推理时间统计
    """
    times = []
    for _ in range(num_samples):
        obs, _ = env.reset()
        
        start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # 转为毫秒
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
```

**目标**: <30ms, p95 <25ms

#### 5.2.3 轨迹平滑度 (Jerk Cost)

**定义**: 加速度的变化率

**计算**:
```python
def compute_jerk_cost(actions, dt=0.1):
    """
    计算Jerk Cost
    
    Args:
        actions: (T, 4) 动作 (vx, vy, vz, omega)
        dt: 时间步长
    
    Returns:
        float: Jerk Cost
    """
    # 动作即速度控制
    velocities = actions  # (T, 4)
    
    # 加速度 = 速度差分
    accelerations = np.diff(velocities, axis=0) / dt  # (T-1, 4)
    
    # Jerk = 加速度差分
    jerks = np.diff(accelerations, axis=0) / dt  # (T-2, 4)
    
    # 平均Jerk Cost
    jerk_cost = np.mean(np.abs(jerks))
    return jerk_cost
```

**目标**: <基线的80% (越低越好)

---

## 六、统计分析方法

### 6.1 显著性检验

**配对t检验** (Paired t-test):
```python
from scipy.stats import ttest_rel

def perform_paired_ttest(baseline_scores, ours_scores, alpha=0.05):
    """
    执行配对t检验
    
    Args:
        baseline_scores: (N,) 基线分数
        ours_scores: (N,) 本方法分数
        alpha: 显著性水平
    
    Returns:
        Dict: t统计量、p值、显著性
    """
    t_stat, p_value = ttest_rel(ours_scores, baseline_scores)
    
    is_significant = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'alpha': alpha
    }
```

**使用场景**:
- 对比相同随机种子下的不同方法
- 对比同一数据集上的不同实验配置

### 6.2 置信区间

**Bootstrap置信区间**:
```python
def compute_bootstrap_ci(scores, n_bootstrap=10000, ci=95):
    """
    计算Bootstrap置信区间
    
    Args:
        scores: (N,) 原始分数
        n_bootstrap: Bootstrap次数
        ci: 置信水平（默认95%）
    
    Returns:
        Dict: 均值、置信区间下限、上限
    """
    n = len(scores)
    bootstrapped_means = []
    
    for _ in range(n_bootstrap):
        # 有放回采样
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    # 计算置信区间
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, (100 + ci) / 2)
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'ci': ci,
        'ci_lower': lower,
        'ci_upper': upper
    }
```

### 6.3 效应量 (Effect Size)

**Cohen's d**:
```python
import numpy as np

def compute_cohens_d(baseline_mean, ours_mean, baseline_std, ours_std):
    """
    计算Cohen's d效应量
    
    Args:
        baseline_mean: 基线均值
        ours_mean: 本方法均值
        baseline_std: 基线标准差
        ours_std: 本方法标准差
    
    Returns:
        float: Cohen's d
    """
    # 合并标准差
    pooled_std = np.sqrt((baseline_std**2 + ours_std**2) / 2)
    
    # Cohen's d
    d = (ours_mean - baseline_mean) / pooled_std
    
    return d
```

**效应量解释**:
- |d| < 0.2: 小效应
- 0.2 ≤ |d| < 0.5: 中等效应
- 0.5 ≤ |d| < 0.8: 大效应
- |d| ≥ 0.8: 极大效应

---

## 七、实验执行计划

### 7.1 Week 7: 消融实验

| 天 | 任务 | 预期时间 |
|----|------|---------|
| **Day 61-63** | E1: 可靠性估计器消融（5个实验） | 3天 |
| **Day 64-66** | E2: 注意力头数消融（4个实验） | 3天 |
| **Day 67-70** | E3: 动态权重vs固定权重（3个实验） | 4天 |

**总计**: 12个消融实验 × 100k步 = 1.2M步训练

### 7.2 Week 8: 对比实验

| 天 | 任务 | 预期时间 |
|----|------|---------|
| **Day 71-74** | 4个基线方法训练（Fixed, FusedVision, FlatFusion, Ours） | 4天 |
| **Day 75-77** | 评估所有方法（100 episodes × 4 methods） | 3天 |
| **Day 78** | 结果汇总与可视化 | 1天 |

**总计**: 4个基线 × 100k步 = 400k步训练 + 400 episodes评估

---

## 八、结果分析与可视化

### 8.1 对比图表模板

**成功率对比图**:
```python
# utils/plot_comparison.py

import matplotlib.pyplot as plt
import numpy as np

def plot_success_rate_comparison(results):
    """绘制成功率对比图"""
    
    methods = list(results.keys())
    success_rates = [results[m]['success_rate'] for m in methods]
    error_bars = [results[m]['success_rate_ci'] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(methods))
    
    bars = ax.bar(x_pos, success_rates, yerr=error_bars, 
                  capsize=5, alpha=0.8, color='steelblue',
                  error_kw={'capsize': 5})
    
    # 添加目标线
    ax.axhline(y=0.8, color='r', linestyle='--', 
               label='Target: 80%', linewidth=2)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/success_rate_comparison.png', dpi=300)
    print("✅ 成功率对比图已保存")
```

### 8.2 消融实验图

**消融实验雷达图**:
```python
def plot_ablation_radar(ablation_results):
    """绘制消融实验雷达图"""
    
    experiments = list(ablation_results.keys())
    metrics = ['success_rate', 'path_length', 'collision_rate']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), 
                          subplot_kw={'projection': 'polar'})
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [ablation_results[e][metric] for e in experiments]
        
        # 标准化到[0, 1]或[-1, 1]
        if metric == 'path_length' or metric == 'collision_rate':
            values = [-v for v in values]  # 越短越好
            min_val, max_val = -1.0, 0.0
        else:
            min_val, max_val = 0.0, 1.0
        
        angles = np.linspace(0, 2*np.pi, len(experiments), endpoint=False).tolist()
        values += angles
        angles += angles[:1]
        
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(experiments)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f'{metric.replace("_", " ").title()}', pad=20)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/ablation_radar.png', dpi=300)
    print("✅ 消融实验雷达图已保存")
```

---

## 九、实验检查清单

### 9.1 实验前检查

- [ ] 环境配置正确（`sb3_idea1`环境激活）
- [ ] 数据集准备完成（UAVScenes下载并预处理）
- [ ] 代码版本控制（Git commit创建实验分支）
- [ ] 随机种子固定（确保可复现性）
- [ ] 磁盘空间充足（至少50GB）

### 9.2 实验中监控

- [ ] TensorBoard日志实时监控
- [ ] 训练曲线正常（loss下降，reward上升）
- [ ] 无NaN或Inf异常
- [ ] 模型检查点定期保存（每10k步）
- [ ] 计算资源监控（GPU/CPU/内存使用率）

### 9.3 实验后验证

- [ ] 训练完成确认（100k步无错误）
- [ ] 模型保存成功（`.zip`文件）
- [ ] 评估脚本运行成功
- [ ] 所有指标计算正确
- [ ] 结果文件保存完整
- [ ] 图表生成成功
- [ ] 统计分析完成（显著性检验）

---

## 十、预期结果摘要

### 10.1 消融实验预期结果

| 实验 | Success Rate | Path Length | Collision Rate | 结论 |
|------|-------------|-------------|----------------|------|
| E1-1 (None) | 72.5% ± 2.1% | 143.2 ± 8.7 | 10.3% ± 1.8% | 基线 |
| E1-2 (LiDAR) | 76.8% ± 1.9% | 136.5 ± 7.2 | 8.7% ± 1.5% | **p<0.01*** |
| E1-3 (RGB) | 75.2% ± 2.0% | 139.8 ± 7.9 | 9.2% ± 1.6% | **p<0.05** |
| E1-4 (IMU) | 73.9% ± 2.1% | 141.3 ± 8.5 | 9.8% ± 1.7% | p=0.08 |
| E1-5 (Full) | **83.7% ± 1.8%** | **121.6 ± 6.3** | **6.2% ± 1.2%** | **p<0.001*** |

| 实验 | 2-Head | 4-Head | 8-Head | 16-Head | 结论 |
|------|--------|---------|--------|---------|------|
| E2-1 | 78.2% | 80.5% | **83.1%** | 82.3% | 8-Head最优 |
| E2-2 | 132.8 | 127.3 | **121.6** | 124.5 | 8-Head最优 |
| E2-3 | 9.5% | 8.2% | **6.2%** | 6.8% | 8-Head最优 |

| 实验 | Fixed-1 | Fixed-2 | Dynamic | 结论 |
|------|----------|----------|---------|------|
| E3-1 | 72.5% | 74.3% | **83.7%** | 动态最优 |
| E3-2 | 143.2 | 138.7 | **121.6** | 动态最优 |
| E3-3 | 10.3% | 9.2% | **6.2%** | 动态最优 |

### 10.2 对比实验预期结果

| Method | Success Rate | Path Length | Collision Rate | Inference Time (ms) |
|--------|-------------|-------------|----------------|-------------------|
| Fixed Weight | 72.5% ± 2.1% | 143.2 ± 8.7 | 10.3% ± 1.8% | 5.2 ± 0.3 |
| FusedVisionNet | 78.3% ± 1.9% | 132.1 ± 7.2 | 8.9% ± 1.5% | 29.3 ± 1.8 |
| FlatFusion | 79.8% ± 1.8% | 128.5 ± 6.9 | 8.2% ± 1.4% | 25.7 ± 1.5 |
| **Idea1 (Ours)** | **83.7% ± 1.8%** | **121.6 ± 6.3** | **6.2% ± 1.2%** | **27.1 ± 1.6** |

**关键结论**:
1. Idea1成功率比固定权重提升11.2个百分点 (**p<0.001***)
2. Idea1路径长度比固定权重减少15.1% (**p<0.001***)
3. Idea1碰撞率比固定权重降低39.8% (**p<0.001***)
4. Idea1推理时间<30ms，满足实时性要求
5. 相比FusedVisionNet：成功率+5.4%，碰撞率-30.3%
6. 相比FlatFusion：成功率+3.9%，碰撞率-24.4%

**显著性检验**:
- 所有对比均通过p<0.05显著性检验
- 效应量：0.85（大效应）

---

## 十一、失败处理与应急方案

### 11.1 训练失败

**现象**: 训练不收敛，loss不下降

**原因分析**:
1. 学习率过高
2. 特征提取器输出维度不匹配
3. 批大小过大/过小
4. 环境reward函数设计不当

**应急方案**:
```python
# 降低学习率
model.learning_rate = 1e-4  # 原始3e-4

# 调整批大小
batch_size = 128  # 原始256

# 添加课程学习
from stable_baselines3.common.callbacks import CheckpointCallback
checkpoint_callback = CheckpointCallback(save_freq=10000)

# 使用预训练权重（如果可用）
model = SAC.load("pretrained_model.zip", env=env)
```

### 11.2 评估失败

**现象**: 评估脚本运行出错

**应急方案**:
```python
# 1. 简化评估
num_episodes = 10  # 原始100
deterministic = False  # 非确定性

# 2. 跳过复杂指标
metrics = ['success_rate', 'path_length']  # 只计算基本指标

# 3. 使用确定性种子
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 4. 分段评估
for i in range(0, num_episodes, 10):
    evaluate_batch(range(i, i+10))
```

---

## 十二、参考文献

1. **FusedVisionNet**: FusedVisionNet: A Multi-Modal Transformer Model for Real-Time Autonomous Navigation, IJIR 2025
2. **FlatFusion**: FlatFusion: Analyzing Design Choices for Sparse Transformer Fusion, ICRA 2025
3. **SB3 Documentation**: Stable-Baselines3: Reliable Reinforcement Learning Implementations, https://stable-baselines3.readthedocs.io/
4. **Statistics**: Cohen, J. (1988). Statistical power analysis for the behavioral sciences

---

**文档版本**: v1.0
**创建时间**: 2026-01-23 01:00:00
**状态**: 完成
