"""
Performance Benchmark Script for Optimized Training

This script measures the performance improvement from the applied optimizations:
1. Memory allocation optimization (repeat -> expand)
2. torch.compile() optimization
3. Reduced TensorBoard logging frequency
4. GPU cache warmup
5. cuDNN benchmark mode

Usage:
    python benchmark_performance.py --encoder-type gap --timesteps 100
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder
from utils.visualization import VisualizationCallback


class PerformanceBenchmark:
    """
    Benchmark training performance with detailed timing breakdown.
    """

    def __init__(self, encoder_type: str = 'gap', use_reliability: bool = True):
        self.encoder_type = encoder_type
        self.use_reliability = use_reliability
        self.env = None
        self.model = None

    def setup(self):
        """Initialize environment and model."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SETUP")
        print("="*60)

        self.env = UAVMultimodalEnv(max_steps=200)

        encoder_class_map = {
            'baseline': 'BaselineExtractor',
            'gap': 'GAPEncoder',
            'mobilenet': 'MobileNetV3Encoder'
        }

        policy_kwargs = {
            'features_extractor_class': GAPEncoder if self.encoder_type == 'gap' else None,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': self.use_reliability,
                'debug_mode': False
            }
        }

        if self.encoder_type != 'gap':
            from networks import BaselineExtractor, MobileNetV3Encoder
            encoder_class_map = {
                'baseline': BaselineExtractor,
                'mobilenet': MobileNetV3Encoder
            }
            policy_kwargs['features_extractor_class'] = encoder_class_map[self.encoder_type]

        self.model = SAC(
            'MultiInputPolicy',
            self.env,
            learning_rate=3e-4,
            verbose=0,
            policy_kwargs=policy_kwargs
        )

        print(f"✓ Environment created: UAVMultimodalEnv")
        print(f"✓ Model created: SAC with {self.encoder_type} encoder")
        print(f"✓ Reliability fusion: {self.use_reliability}")

    def benchmark_forward_pass(self, num_steps: int = 100) -> Dict[str, float]:
        """
        Benchmark forward pass timing.

        Args:
            num_steps: Number of steps to measure

        Returns:
            Dict with timing statistics (in milliseconds)
        """
        print(f"\n{'='*60}")
        print(f"FORWARD PASS BENCHMARK ({num_steps} steps)")
        print(f"{'='*60}")

        obs, _ = self.env.reset()
        times_ms = []

        for i in range(num_steps):
            start = time.perf_counter()

            with torch.no_grad():
                action = self.model.predict(obs, deterministic=False)[0]

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            times_ms.append(elapsed_ms)

            if (i + 1) % 20 == 0:
                print(f"  Step {i+1:3d}/{num_steps}: {elapsed_ms:6.2f} ms")

        stats = {
            'mean_ms': np.mean(times_ms),
            'median_ms': np.median(times_ms),
            'std_ms': np.std(times_ms),
            'min_ms': np.min(times_ms),
            'max_ms': np.max(times_ms),
            'total_ms': np.sum(times_ms)
        }

        print(f"\n  Forward Pass Statistics:")
        print(f"    Mean:   {stats['mean_ms']:6.2f} ms/step")
        print(f"    Median: {stats['median_ms']:6.2f} ms/step")
        print(f"    Std:    {stats['std_ms']:6.2f} ms")
        print(f"    Min:    {stats['min_ms']:6.2f} ms")
        print(f"    Max:    {stats['max_ms']:6.2f} ms")
        print(f"    Total:  {stats['total_ms']:6.2f} ms ({num_steps} steps)")

        return stats

    def benchmark_training(self, total_timesteps: int) -> Dict[str, float]:
        """
        Benchmark full training loop.

        Args:
            total_timesteps: Number of timesteps to train

        Returns:
            Dict with training statistics
        """
        print(f"\n{'='*60}")
        print(f"TRAINING BENCHMARK ({total_timesteps} timesteps)")
        print(f"{'='*60}")

        viz_callback = VisualizationCallback(
            debug=False,
            log_every_steps=1000,
            log_dir=None
        )

        start_time = time.time()

        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            callback=viz_callback
        )

        elapsed_time = time.time() - start_time

        stats = {
            'total_time_s': elapsed_time,
            'time_per_step_ms': (elapsed_time / total_timesteps) * 1000,
            'steps_per_second': total_timesteps / elapsed_time
        }

        print(f"\n  Training Statistics:")
        print(f"    Total Time:     {stats['total_time_s']:6.2f} s")
        print(f"    Time per Step:  {stats['time_per_step_ms']:6.2f} ms")
        print(f"    Steps per Sec:  {stats['steps_per_second']:6.2f}")

        return stats

    def print_summary(self, forward_stats: Dict, training_stats: Dict | None):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        print(f"\nConfiguration:")
        print(f"  Encoder Type:       {self.encoder_type}")
        print(f"  Reliability Fusion: {self.use_reliability}")

        print(f"\nForward Pass Performance:")
        print(f"  Mean Latency:       {forward_stats['mean_ms']:6.2f} ms/step")
        print(f"  Median Latency:     {forward_stats['median_ms']:6.2f} ms/step")
        print(f"  Std Deviation:      {forward_stats['std_ms']:6.2f} ms")

        print(f"\nTraining Loop Performance:")
        if training_stats:
            print(f"  Total Time:         {training_stats['total_time_s']:6.2f} s")
            print(f"  Time per Step:      {training_stats['time_per_step_ms']:6.2f} ms")
            print(f"  Throughput:         {training_stats['steps_per_second']:6.2f} steps/s")
        else:
            print("  Skipped (forward-only)")

        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimized training performance")

    parser.add_argument(
        '--encoder-type',
        type=str,
        default='gap',
        choices=['baseline', 'gap', 'mobilenet'],
        help="RGB encoder type (default: gap)"
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=100,
        help="Number of timesteps to benchmark (default: 100)"
    )

    parser.add_argument(
        '--forward-only',
        action='store_true',
        help="Only benchmark forward pass, skip full training"
    )

    parser.add_argument(
        '--no-reliability',
        action='store_true',
        help="Disable reliability-aware fusion"
    )

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(
        encoder_type=args.encoder_type,
        use_reliability=not args.no_reliability
    )

    benchmark.setup()

    forward_stats = benchmark.benchmark_forward_pass(num_steps=100)

    if not args.forward_only:
        training_stats = benchmark.benchmark_training(total_timesteps=args.timesteps)
    else:
        training_stats = None

    benchmark.print_summary(forward_stats, training_stats)


if __name__ == "__main__":
    main()
