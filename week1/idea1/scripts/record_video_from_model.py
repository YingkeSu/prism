#!/usr/bin/env python3
"""
使用已保存的模型录制视频示例

用法:
    python scripts/record_video_from_model.py --model models/idea1_model_5000ts.zip --n-videos 5
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from utils.video_recorder import record_evaluation_video


def main():
    parser = argparse.ArgumentParser(
        description="使用已保存的模型录制智能体行为视频"
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='已保存模型的路径 (例如: models/idea1_model_5000ts.zip)'
    )

    parser.add_argument(
        '--n-videos',
        type=int,
        default=5,
        help='要录制的episode数量 (默认: 5)'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='每个episode的最大步数 (默认: 200)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='videos/agent_evaluation.mp4',
        help='输出视频文件路径 (默认: videos/agent_evaluation.mp4)'
    )

    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='使用确定性动作 (默认: True)'
    )

    args = parser.parse_args()

    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 错误: 找不到模型文件: {args.model}")
        print(f"   请先训练模型:")
        print(f"   python train.py --timesteps 1000 --save")
        sys.exit(1)

    print("="*60)
    print("使用已保存模型录制视频")
    print("="*60)
    print(f"模型路径: {args.model}")
    print(f"录制数量: {args.n_videos} episodes")
    print(f"最大步数: {args.max_steps}")
    print(f"输出文件: {args.output}")
    print()

    # 1. 加载模型
    print("⏳ 正在加载模型...")
    try:
        model = SAC.load(str(model_path))
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    # 2. 创建环境（需要rgb_array渲染模式）
    print("⏳ 正在创建环境...")
    env = UAVMultimodalEnv(max_steps=args.max_steps, render_mode='rgb_array')
    print("✅ 环境创建成功")

    # 3. 录制视频
    print()
    print("🎬 开始录制视频...")
    print()

    try:
        record_evaluation_video(
            model=model,
            env=env,
            video_path=args.output,
            n_episodes=args.n_videos,
            deterministic=args.deterministic
        )

        print()
        print("="*60)
        print("✅ 视频录制完成！")
        print("="*60)
        print(f"视频已保存到: {args.output}")
        print()
        print("查看视频:")
        print(f"  macOS: open {args.output}")
        print(f"  Linux: xdg-open {args.output}")
        print(f"  Windows: start {args.output}")

    except Exception as e:
        print()
        print(f"❌ 视频录制失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
