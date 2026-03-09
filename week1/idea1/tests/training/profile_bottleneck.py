"""
Detailed Performance Profiler for Training Bottleneck Diagnosis

This script profiles each component of the training loop to identify
the actual bottleneck causing 1000-step training to exceed 300 seconds.
"""

import time
from typing import Dict, List
import torch
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


class TrainingProfiler:
    """Profile each training component separately."""

    def __init__(self):
        self.env = UAVMultimodalEnv(max_steps=200)
        self.model = None
        self.timings = {
            'rollout': [],
            'forward': [],
            'backward': [],
            'optimizer_step': [],
            'total_step': []
        }

    def setup_model(self, use_compile: bool = False):
        """Setup model with or without torch.compile."""
        policy_kwargs = {
            'features_extractor_class': GAPEncoder,
            'features_extractor_kwargs': {
                'features_dim': 256,
                'use_reliability': True,
                'debug_mode': False
            }
        }

        self.model = SAC(
            'MultiInputPolicy',
            self.env,
            learning_rate=3e-4,
            verbose=0,
            policy_kwargs=policy_kwargs
        )

        if use_compile:
            print("Applying torch.compile()...")
            start = time.time()
            self.model.policy.features_extractor = torch.compile(
                self.model.policy.features_extractor,
                mode="reduce-overhead",
                fullgraph=False
            )
            compile_time = time.time() - start
            print(f"  Compilation took: {compile_time:.2f}s")

    def profile_single_step(self, step_num: int) -> Dict[str, float]:
        """Profile a single training step with detailed timing."""
        step_timings = {}

        obs, _ = self.env.reset()

        # Phase 1: Rollout (collect experience)
        start_rollout = time.perf_counter()
        for _ in range(1):  # Single step rollout
            action = self.model.predict(obs, deterministic=False)[0]
            obs, reward, done, truncated, info = self.env.step(action)
        step_timings['rollout_ms'] = (time.perf_counter() - start_rollout) * 1000

        # Phase 2: Forward pass through policy
        start_forward = time.perf_counter()
        with torch.no_grad():
            _ = self.model.policy._predict(obs, deterministic=False)
        step_timings['forward_ms'] = (time.perf_counter() - start_forward) * 1000

        # Phase 3: Full training step (includes backward + optimizer)
        start_train = time.perf_counter()
        self.model.learn(total_timesteps=1, log_interval=None)
        step_timings['learn_ms'] = (time.perf_counter() - start_train) * 1000

        return step_timings

    def profile_training_ramp(self, max_steps: int = 200, step_interval: int = 50):
        """Profile training at different step counts to identify degradation."""
        print(f"\n{'='*70}")
        print(f"TRAINING RAMP PROFILING (0 to {max_steps} steps)")
        print(f"{'='*70}\n")

        results = []

        for num_steps in range(step_interval, max_steps + 1, step_interval):
            print(f"\n--- Testing {num_steps} steps ---")

            # Reset model for each test
            self.setup_model(use_compile=False)
            start_total = time.time()

            # Profile in chunks
            chunk_times = []
            for i in range(0, num_steps, 10):
                chunk_start = time.time()
                steps_to_do = min(10, num_steps - i)
                self.model.learn(total_timesteps=steps_to_do, log_interval=None)
                chunk_time = time.time() - chunk_start
                chunk_times.append(chunk_time)

            total_time = time.time() - start_total
            time_per_step = (total_time / num_steps) * 1000

            print(f"  Total time:   {total_time:.2f}s")
            print(f"  Time per step: {time_per_step:.2f}ms")
            print(f"  Chunk times:   {[f'{t:.2f}s' for t in chunk_times[-3:]]}")

            results.append({
                'num_steps': num_steps,
                'total_time_s': total_time,
                'time_per_step_ms': time_per_step,
                'chunk_times': chunk_times
            })

        return results

    def profile_compile_impact(self):
        """Compare performance with and without torch.compile()."""
        print(f"\n{'='*70}")
        print("TORCH.COMPILE IMPACT ANALYSIS")
        print(f"{'='*70}\n")

        num_test_steps = 50

        # Test WITHOUT compile
        print("Testing WITHOUT torch.compile()...")
        self.setup_model(use_compile=False)
        start = time.time()
        self.model.learn(total_timesteps=num_test_steps, log_interval=None)
        time_no_compile = time.time() - start
        print(f"  {num_test_steps} steps: {time_no_compile:.2f}s ({time_no_compile/num_test_steps*1000:.2f}ms/step)")

        # Test WITH compile
        print("\nTesting WITH torch.compile()...")
        self.setup_model(use_compile=True)
        start = time.time()
        self.model.learn(total_timesteps=num_test_steps, log_interval=None)
        time_with_compile = time.time() - start
        print(f"  {num_test_steps} steps: {time_with_compile:.2f}s ({time_with_compile/num_test_steps*1000:.2f}ms/step)")

        speedup = time_no_compile / time_with_compile if time_with_compile > 0 else 0
        print(f"\n  Speedup: {speedup:.2f}x")

        return {
            'no_compile_s': time_no_compile,
            'with_compile_s': time_with_compile,
            'speedup': speedup
        }

    def profile_replay_buffer_growth(self):
        """Test if replay buffer growth affects performance."""
        print(f"\n{'='*70}")
        print("REPLAY BUFFER GROWTH IMPACT")
        print(f"{'='*70}\n")

        self.setup_model(use_compile=False)

        # Test at different buffer sizes
        buffer_sizes = [100, 500, 1000, 2000]

        for target_size in buffer_sizes:
            # Fill buffer to target size
            print(f"Testing with buffer size ~{target_size}...")
            start = time.time()
            self.model.learn(total_timesteps=target_size, log_interval=None)
            fill_time = time.time() - start

            # Now test 10 steps at this buffer size
            start = time.time()
            self.model.learn(total_timesteps=10, log_interval=None)
            test_time = time.time() - start

            print(f"  Fill time: {fill_time:.2f}s")
            print(f"  10 steps at full buffer: {test_time:.2f}s ({test_time/10*1000:.2f}ms/step)\n")

    def print_diagnostics(self):
        """Print diagnostic information."""
        print(f"\n{'='*70}")
        print("SYSTEM DIAGNOSTICS")
        print(f"{'='*70}\n")

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

        # Initialize a temporary model for diagnostics
        if self.model is None:
            policy_kwargs = {
                'features_extractor_class': GAPEncoder,
                'features_extractor_kwargs': {
                    'features_dim': 256,
                    'use_reliability': True,
                    'debug_mode': False
                }
            }
            self.model = SAC('MultiInputPolicy', self.env, policy_kwargs=policy_kwargs, verbose=0)

        print(f"\nSB3 SAC policy type: {type(self.model.policy).__name__}")

        # Check if features_extractor is compiled
        if hasattr(self.model.policy, 'features_extractor') and self.model.policy.features_extractor is not None:
            fe = self.model.policy.features_extractor
            print(f"Features extractor type: {type(fe)}")
            print(f"Is compiled: {type(fe).__name__ == 'CompiledFn'}")
        else:
            print("Features extractor: None (MultiInputPolicy uses extractors_dict)")


def main():
    print("="*70)
    print("TRAINING PERFORMANCE PROFILER")
    print("="*70)

    profiler = TrainingProfiler()
    profiler.print_diagnostics()

    # Test 1: Training ramp to find degradation point
    print("\n" + "="*70)
    print("TEST 1: Training Ramp Analysis")
    print("="*70)
    ramp_results = profiler.profile_training_ramp(max_steps=200, step_interval=50)

    # Test 2: Compile impact
    print("\n" + "="*70)
    print("TEST 2: torch.compile() Impact")
    print("="*70)
    compile_results = profiler.profile_compile_impact()

    # Test 3: Replay buffer impact
    print("\n" + "="*70)
    print("TEST 3: Replay Buffer Growth")
    print("="*70)
    profiler.profile_replay_buffer_growth()

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print("\nKey Findings:")
    print(f"  1. Ramp test: Check if time_per_step increases with steps")
    print(f"  2. Compile speedup: {compile_results['speedup']:.2f}x")
    print(f"     - If < 1.0: torch.compile() SLOWS down training")
    print(f"     - If > 1.0: torch.compile() HELPS")

    print("\nRecommendations:")
    if compile_results['speedup'] < 1.0:
        print("  ⚠️  DISABLE torch.compile() - it's causing slowdown")
    else:
        print("  ✓  KEEP torch.compile() - it's helping")

    # Check ramp results for degradation
    if len(ramp_results) >= 2:
        first_time = ramp_results[0]['time_per_step_ms']
        last_time = ramp_results[-1]['time_per_step_ms']
        degradation_ratio = last_time / first_time if first_time > 0 else 0

        print(f"\n  Performance degradation: {degradation_ratio:.2f}x")
        if degradation_ratio > 2.0:
            print("  ⚠️  MAJOR performance degradation detected!")
            print("     Likely causes: Memory leak, replay buffer issue, or buffer size")
        elif degradation_ratio > 1.2:
            print("  ⚠️  Minor performance degradation")
        else:
            print("  ✓  Performance stable")


if __name__ == "__main__":
    main()
