# High Temporal Consistency through Semantic Similarity Propagation in Semi-Supervised Video Semantic Segmentation for Autonomous Flight

**基本信息**
- **论文标题**: High Temporal Consistency through Semantic Similarity Propagation in Semi-Supervised Video Semantic Segmentation for Autonomous Flight
- **简称**: SSP (Semantic Similarity Propagation)
- **会议**: CVPR 2025
- **年份**: 2025年
- **原文链接**: https://arxiv.org/html/2503.15676v2

**研究背景**
半监督视频语义分割中，时序一致性是关键挑战。现有方法在相邻帧间存在语义不一致，导致视频闪烁和目标身份丢失。

**核心贡献**
1. **语义相似性传播（SSP）**:
   - **核心创新**: 通过语义相似性传播保证时序一致性
   - 在视频序列中传播语义标签
   - 减少帧间语义不一致

2. **半监督学习**:
   - 利用少量标注数据
   - 伪标签生成策略
   - 提升标注效率

3. **UAV自主飞行场景**:
   - 针对UAV自主飞行任务设计
   - 适合真实世界的复杂场景
   - 实时性要求

**技术细节**
- **输入**: 视频序列
- **输出**: 语义分割序列
- **方法**: 语义相似性传播
- **学习**: 半监督训练
- **一致性**: 时序语义一致性

**实验结果**
- 在UAVid和RuralScapes数据集上验证
- 时序一致性显著提升
- 半监督学习接近全监督性能
- 实时性满足在线推理需求

**与UAV导航的相关性**
直接面向UAV自主飞行：
1. **视频语义分割**: UAV场景理解的核心能力
2. **时序一致性**: UAV快速运动中的关键问题
3. **半监督学习**: 减少标注成本的实用方法

**可借鉴点**
1. **语义相似性传播机制**: 保证时序一致性
2. **半监督学习策略**: 伪标签生成
3. **UAV特定数据集**: UAVid, RuralScapes
4. **实时性优化**: 在线推理的高效实现

**局限与改进空间**
1. 仅考虑语义一致性，未考虑多模态融合
2. 伪标签生成可能引入错误
3. 对动态场景处理不足
4. 缺乏可靠性感知

**创新机会（针对UAV）**
1. **多模态时序一致性**: 融合LiDAR和视觉的时序信息
2. **动态伪标签生成**: 根据场景复杂度调整策略
3. **可靠性感知的一致性**: 在低质量传感器数据时降低传播强度
4. **预测性时序融合**: 预测未来语义状态，提前优化
