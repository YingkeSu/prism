"""train.py

Main training entry point for UAV multi-modal RL with quality-aware fusion.

Supports three fusion modes:
- fixed_weight: Equal or custom fixed weights
- dynamic_weight: Dynamic attention-based weighting
- quality_aware: Quality-gated dynamic weighting with optional temporal prediction
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.uav_multimodal_extractor import UAVMultimodalExtractor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train UAV multi-modal RL with quality-aware fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Fusion mode
    parser.add_argument(
        "--fusion-mode",
        type=str,
        choices=["fixed_weight", "dynamic_weight", "quality_aware"],
        default="quality_aware",
        help="Fusion mode to use",
    )

    # Environment
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode",
    )

    # Training
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Model architecture
    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Feature dimension",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (for dynamic/quality_aware modes)",
    )
    parser.add_argument(
        "--use-reliability",
        action="store_true",
        help="Use reliability estimation in fusion",
    )

    # Quality-aware specific
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Use temporal quality prediction (only for quality_aware mode)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Historical quality sequence length (for temporal prediction)",
    )

    # Fixed weight specific
    parser.add_argument(
        "--fixed-weights",
        type=float,
        nargs=3,
        default=[1/3, 1/3, 1/3],
        metavar=("W_LIDAR", "W_RGB", "W_IMU"),
        help="Fixed weights for lidar, rgb, imu (must sum to 1.0)",
    )

    # Logging
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/sb3",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (default: auto-generated)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0: silent, 1: progress bar, 2: detailed)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration dictionary from arguments."""
    config = {
        "fusion_mode": args.fusion_mode,
        "max_steps": args.max_steps,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "features_dim": args.features_dim,
        "num_heads": args.num_heads,
        "use_reliability": args.use_reliability,
        "logdir": args.logdir,
        "verbose": args.verbose,
    }

    # Mode-specific config
    if args.fusion_mode == "fixed_weight":
        # Normalize fixed weights to sum to 1
        weights = np.array(args.fixed_weights, dtype=np.float32)
        weights = weights / weights.sum()
        config["fixed_weights"] = tuple(weights.tolist())
    elif args.fusion_mode == "quality_aware":
        config["use_temporal"] = args.use_temporal
        config["seq_len"] = args.seq_len

    return config


def build_extractor_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build feature extractor kwargs based on fusion mode."""
    mode = config["fusion_mode"]
    kwargs = {
        "features_dim": config["features_dim"],
        "use_reliability": config["use_reliability"],
    }

    if mode == "fixed_weight":
        kwargs["fixed_weights"] = config.get("fixed_weights", (1/3, 1/3, 1/3))
    elif mode in ("dynamic_weight", "quality_aware"):
        kwargs["num_heads"] = config["num_heads"]

    return kwargs


def generate_run_name(config: Dict[str, Any]) -> str:
    """Generate run name from configuration."""
    mode = config["fusion_mode"]
    seed = config["seed"]

    if mode == "fixed_weight":
        weights = config["fixed_weights"]
        return f"fixed_lidar{weights[0]:.2f}_rgb{weights[1]:.2f}_imu{weights[2]:.2f}_seed{seed}"
    elif mode == "dynamic_weight":
        heads = config["num_heads"]
        return f"dynamic_heads{heads}_seed{seed}"
    elif mode == "quality_aware":
        heads = config["num_heads"]
        temporal = "_temporal" if config.get("use_temporal", False) else ""
        return f"quality_heads{heads}{temporal}_seed{seed}"
    else:
        return f"unknown_mode_{mode}_seed{seed}"


def main() -> None:
    """Main training function."""
    args = parse_args()
    config = build_config(args)

    # Set random seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Print configuration
    print("=" * 60)
    print(f"Fusion Mode: {config['fusion_mode'].upper()}")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Create environment
    print("Creating environment...")
    env = UAVMultimodalEnv(max_steps=config["max_steps"])

    # Create run name
    run_name = args.run_name or generate_run_name(config)
    print(f"Run name: {run_name}")

    # Build extractor kwargs
    extractor_kwargs = build_extractor_kwargs(config)
    print(f"Extractor kwargs: {extractor_kwargs}")

    # Create model
    print("Creating model...")
    model = SAC(
        "MultiInputPolicy",
        env,
        device="auto",
        learning_rate=config["learning_rate"],
        tensorboard_log=config["logdir"],
        verbose=config["verbose"],
        policy_kwargs={
            "features_extractor_class": UAVMultimodalExtractor,
            "features_extractor_kwargs": extractor_kwargs,
        },
    )

    num_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model created with {num_params / 1e6:.2f}M parameters")

    # Setup callbacks
    os.makedirs(config["logdir"], exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=config["logdir"],
        name_prefix=f"{run_name}_checkpoint",
    )

    # Start training
    print(f"Training for {config['total_timesteps']} timesteps...")
    print(f"Logs: {config['logdir']}/{run_name}")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            log_interval=max(1, config["total_timesteps"] // 10),
            tb_log_name=run_name,
            callback=checkpoint_callback,
        )

        print("=" * 60)
        print("Training completed successfully!")
        print(f"Run name: {run_name}")
        print(f"Logs saved to: {config['logdir']}/{run_name}")
        print("=" * 60)

        # Save final model
        model_path = os.path.join(config["logdir"], f"{run_name}_final.zip")
        model.save(model_path)
        print(f"Final model saved to: {model_path}")

    except Exception as e:
        print("=" * 60)
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
