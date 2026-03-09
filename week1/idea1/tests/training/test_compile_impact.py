"""
Test torch.compile() impact on training performance.
"""

import time
import torch
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


def test_training(with_compile: bool, num_steps: int = 100):
    """Test training with/without torch.compile."""
    print(f"\n{'='*60}")
    print(f"Testing {num_steps} steps - COMPILE: {with_compile}")
    print(f"{'='*60}")

    env = UAVMultimodalEnv(max_steps=200)

    policy_kwargs = {
        'features_extractor_class': GAPEncoder,
        'features_extractor_kwargs': {
            'features_dim': 256,
            'use_reliability': True,
            'debug_mode': False
        }
    }

    model = SAC(
        'MultiInputPolicy',
        env,
        learning_rate=3e-4,
        verbose=0,
        policy_kwargs=policy_kwargs,
        buffer_size=10000
    )

    if with_compile:
        print("Applying torch.compile()...")
        start_compile = time.time()
        model.policy.features_extractor = torch.compile(
            model.policy.features_extractor,
            mode="reduce-overhead",
            fullgraph=False
        )
        compile_time = time.time() - start_compile
        print(f"  Compilation took: {compile_time:.2f}s")

    print(f"\nTraining {num_steps} steps...")
    start_train = time.time()
    model.learn(total_timesteps=num_steps, log_interval=None)
    train_time = time.time() - start_train

    time_per_step = (train_time / num_steps) * 1000

    print(f"\nResults:")
    print(f"  Total time:     {train_time:.2f}s")
    print(f"  Time per step:  {time_per_step:.2f}ms")
    print(f"  Throughput:     {num_steps/train_time:.2f} steps/s")

    return {
        'with_compile': with_compile,
        'num_steps': num_steps,
        'total_time_s': train_time,
        'time_per_step_ms': time_per_step,
        'throughput_sps': num_steps/train_time
    }


# Test WITHOUT compile
result_no_compile = test_training(with_compile=False, num_steps=200)

# Test WITH compile
result_with_compile = test_training(with_compile=True, num_steps=200)

# Comparison
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}\n")

speedup = result_no_compile['total_time_s'] / result_with_compile['total_time_s']

print(f"WITHOUT torch.compile():")
print(f"  200 steps: {result_no_compile['total_time_s']:.2f}s")
print(f"  {result_no_compile['time_per_step_ms']:.2f}ms/step")

print(f"\nWITH torch.compile():")
print(f"  200 steps: {result_with_compile['total_time_s']:.2f}s")
print(f"  {result_with_compile['time_per_step_ms']:.2f}ms/step")

print(f"\nSpeedup: {speedup:.2f}x")

if speedup < 1.0:
    print(f"\n⚠️  CONCLUSION: torch.compile() SLOWS training by {1/speedup:.2f}x")
    print("   RECOMMENDATION: DISABLE torch.compile()")
elif speedup > 1.2:
    print(f"\n✓ CONCLUSION: torch.compile() SPEEDS UP training by {speedup:.2f}x")
    print("   RECOMMENDATION: KEEP torch.compile()")
else:
    print(f"\n→ CONCLUSION: torch.compile() has minimal impact")
    print("   RECOMMENDATION: Optional - can disable for simplicity")

# Verify 200-step target
print(f"\n{'='*60}")
print("TARGET VERIFICATION (200 steps < 20s)")
print(f"{'='*60}\n")

if result_no_compile['total_time_s'] < 20:
    print(f"✓ WITHOUT compile: {result_no_compile['total_time_s']:.2f}s < 20s ✓")
else:
    print(f"✗ WITHOUT compile: {result_no_compile['total_time_s']:.2f}s > 20s ✗")

if result_with_compile['total_time_s'] < 20:
    print(f"✓ WITH compile:    {result_with_compile['total_time_s']:.2f}s < 20s ✓")
else:
    print(f"✗ WITH compile:    {result_with_compile['total_time_s']:.2f}s > 20s ✗")
