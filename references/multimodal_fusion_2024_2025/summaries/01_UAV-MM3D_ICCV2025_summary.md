# UAV-MM3D: A Large-Scale Synthetic Benchmark for 3D Perception of Unmanned Aerial Vehicles with Multi-Modal Data

**基本信息**
- **论文标题**: UAV-MM3D: A Large-Scale Synthetic Benchmark for 3D Perception of Unmanned Aerial Vehicles with Multi-Modal Data
- **作者**: Wang et al.
- **发表会议**: ICCV 2025
- **年份**: 2025年
- **arXiv ID**: 2511.22404
- **原文链接**: https://arxiv.org/html/2511.22404v1

**研究背景**
当前UAV感知任务缺乏大规模、标准化的多模态基准数据集，这阻碍了公平对比算法性能和推进多模态融合技术的发展。

**核心贡献**
1. **大规模多模态数据集**: 发布了首个专为UAV 3D感知设计的大规模合成数据集
   - 包含LiDAR点云数据
   - 包含RGB、红外（IR）、雷达（Radar）和DVS（事件视觉）等多模态数据
   - 数据在不同场景和天气条件下同步采集

2. **LGFusionNet基线模型**: 提出了LiDAR引导的多模态融合网络
   - 利用LiDAR深度信息对齐RGB和IR分支特征
   - 主要用于6-DoF位姿估计
   - ResNet-50骨干网络

**技术细节**
- **数据规模**: 大规模数据集（具体规模待查）
- **模态**: LiDAR + RGB + IR + Radar + DVS
- **任务支持**: 语义分割、深度估计、6-DoF定位、位置识别、新视角合成（NVS）
- **基准**: 提供标准化评估协议

**实验结果**
- LGFusionNet在多模态融合任务上建立了性能基线
- 数据集支持多种感知任务的训练和评估

**与UAV导航的相关性**
- 直接面向UAV感知任务
- 多模态数据同步设计对UAV实时融合有参考价值
- 6-DoF定位是UAV位姿估计的关键任务

**局限与改进空间**
- 数据集为合成数据，与真实场景可能存在域差异
- LGFusionNet主要关注融合，未考虑动态传感器可靠性

**可借鉴点**
1. 多模态数据同步与标注方法
2. LiDAR引导的融合对齐策略
3. 标准化的UAV感知基准设计
