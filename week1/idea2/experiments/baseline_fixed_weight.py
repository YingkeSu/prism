import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks.uav_multimodal_extractor import UAVMultimodalExtractor


def baseline_fixed_weight():
    """Train model with fixed modality weights"""

    print("="*60)
    print("Baseline Study: Fixed Weight Fusion")
    print("="*60)

    # Configuration
    config = {
        'use_reliability': True,
        'num_heads': 8,
        'fixed_weights': (1/3, 1/3, 1/3),  # Equal weights: sum=1
        'total_timesteps': 1000,
        'seed': 42,
        'logdir': 'logs/sb3'
    }

    # Print configuration
    print("Configuration:")
    print(f"  use_reliability: {config['use_reliability']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  fixed_weights: {config['fixed_weights']}")
    print(f"  total_timesteps: {config['total_timesteps']}")
    print(f"  seed: {config['seed']}")
    print(f"  logdir: {config['logdir']}")

    # Create environment
    env = UAVMultimodalEnv(max_steps=100)

    # Create run name
    run_name = f"baseline_fixed_weight_seed{config['seed']}"

    # Create model with MultiInputPolicy and custom feature extractor
    model = SAC(
        "MultiInputPolicy",
        env,
        device="auto",
        learning_rate=3e-4,
        tensorboard_log=config['logdir'],
        verbose=2,
        policy_kwargs={
            'features_extractor_class': UAVMultimodalExtractor,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': config['use_reliability'],
                'num_heads': config['num_heads'],
                'fixed_weights': config['fixed_weights']  # To be added in Task 3c
            }
        }
    )

    print(f"Model created with {sum(p.numel() for p in model.policy.parameters()) / 1e6:.1f}M parameters")

    # Start training
    print(f"Training for {config['total_timesteps']} timesteps...")

    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            log_interval=100,
            tb_log_name=run_name
        )

        print(f"✅ Training completed successfully!")
        print(f"  Run name: {run_name}")
        print(f"  TensorBoard logs saved to {config['logdir']}/{run_name}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    baseline_fixed_weight()
