"""
Demonstration script for agent-environment interaction visualization.

This script shows how to use the visualization utilities to monitor
agent behavior during training and evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import SAC

from envs.uav_multimodal_env import UAVMultimodalEnv
from utils.visualization import VisualizationCallback
from utils.video_recorder import record_evaluation_video, create_recording_env


def train_with_visualization(total_timesteps: int = 200, record_videos: bool = True):
    """
    Train a model with comprehensive visualization enabled.

    This demonstrates:
    - Trajectory tracking (3D positions)
    - Action logging (velocity commands)
    - TensorBoard metrics
    - Video recording during training
    """
    print("="*60)
    print("Training with Visualization")
    print("="*60)

    # Create environment
    env = UAVMultimodalEnv(max_steps=200)

    # Create model
    model = SAC(
        'MultiInputPolicy',
        env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log='logs/visualization_demo'
    )

    # Create visualization callback with trajectory tracking
    viz_callback = VisualizationCallback(
        debug=True,
        log_every_steps=50,
        log_dir='logs/visualization_demo/train',
        save_trajectories=True,
        trajectory_log_interval=5
    )

    # Train
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=viz_callback,
        log_interval=1
    )

    # Save model
    model_path = Path("models/visualization_demo_model.zip")
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    return str(model_path), env


def evaluate_with_video(model_path: str, n_videos: int = 3):
    """
    Evaluate a trained model and generate videos.

    This demonstrates video recording for qualitative assessment.
    """
    print("\n" + "="*60)
    print("Evaluation with Video Recording")
    print("="*60)

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = SAC.load(model_path)

    # Create environment for recording with rgb_array render mode
    env = UAVMultimodalEnv(max_steps=200, render_mode='rgb_array')

    # Record videos
    video_path = f"videos/evaluation_demo.mp4"
    print(f"\nRecording {n_videos} episode(s)...")

    record_evaluation_video(
        model=model,
        env=env,
        video_path=video_path,
        n_episodes=n_videos,
        deterministic=True
    )

    print(f"\n✅ Videos saved to: videos/")
    print(f"\nTo view the video:")
    print(f"  - macOS: open {video_path}")
    print(f"  - Linux: xdg-open {video_path}")
    print(f"  - Windows: start {video_path}")


def view_tensorboard_logs(log_dir: str = "logs/visualization_demo"):
    """
    Print instructions for viewing TensorBoard logs.
    """
    print("\n" + "="*60)
    print("TensorBoard Visualization")
    print("="*60)
    print(f"\nTo view training progress in TensorBoard:")
    print(f"  1. Install tensorboard: pip install tensorboard")
    print(f"  2. Run: tensorboard --logdir {log_dir}")
    print(f"  3. Open browser to: http://localhost:6006")
    print(f"\nAvailable visualizations:")
    print(f"  - Scalars: episode/reward, episode/length")
    print(f"  - Trajectories: trajectory/x, trajectory/y, trajectory/z")
    print(f"  - Actions: action/vx, action/vy, action/vz, action/omega")
    print(f"  - Reliability: reliability/lidar, reliability/rgb, reliability/imu")
    print(f"  - Fusion weights: fusion_weights/lidar, fusion_weights/rgb, fusion_weights/imu")
    print(f"  - Sensors: sensors/rgb (image), sensors/lidar_*, sensors/imu_*")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate agent-environment interaction visualization"
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'both'],
        default='both',
        help='Operation mode: train, evaluate, or both (default: both)'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=200,
        help='Training timesteps (default: 200)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/visualization_demo_model.zip',
        help='Path to saved model for evaluation (default: models/visualization_demo_model.zip)'
    )

    parser.add_argument(
        '--n-videos',
        type=int,
        default=3,
        help='Number of videos to generate during evaluation (default: 3)'
    )

    args = parser.parse_args()

    # Run based on mode
    if args.mode in ['train', 'both']:
        model, env = train_with_visualization(args.timesteps)

        if args.mode == 'both':
            evaluate_with_video(str(model), args.n_videos)
    elif args.mode == 'evaluate':
        evaluate_with_video(args.model, args.n_videos)

    # Print TensorBoard instructions
    view_tensorboard_logs()

    print("\n" + "="*60)
    print("✅ Visualization demonstration complete!")
    print("="*60)


if __name__ == "__main__":
    main()
