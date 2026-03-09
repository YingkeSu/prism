"""
Minimal test: 200 steps with NO callbacks, minimal config.
"""

import time
import torch
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


print("="*70)
print("MINIMAL 200-STEP TEST (NO CALLBACKS, MINIMAL CONFIG)")
print("="*70)

# Create environment
env = UAVMultimodalEnv(max_steps=200)

# MINIMAL policy_kwargs
policy_kwargs = {
    'features_extractor_class': GAPEncoder,
    'features_extractor_kwargs': {
        'features_dim': 256,
        'use_reliability': True,
        'debug_mode': False  # CRITICAL: disable debug mode
    }
}

# Create model with SMALL buffer
model = SAC(
    'MultiInputPolicy',
    env,
    learning_rate=3e-4,
    verbose=0,
    policy_kwargs=policy_kwargs,
    buffer_size=50000,  # Smaller buffer
    batch_size=8,
    learning_starts=100,
    train_freq=(4, "step"),
    gradient_steps=1,
    tensorboard_log=None  # NO tensorboard
)

print("\nTraining 200 steps (with progress updates)...")
start = time.time()

# Train in chunks to show progress
chunk_size = 50
for i in range(0, 200, chunk_size):
    chunk_start = time.time()
    model.learn(total_timesteps=chunk_size, log_interval=None, tb_log_name=None)
    chunk_time = time.time() - chunk_start
    elapsed = time.time() - start
    
    print(f"  Steps {i:3d}-{i+chunk_size:3d}: {chunk_time:5.2f}s  |  Total: {elapsed:5.2f}s")

total_time = time.time() - start

print(f"\n{'='*70}")
print(f"RESULTS: 200 steps in {total_time:.2f}s")
print(f"{'='*70}")

ms_per_step = (total_time / 200) * 1000
print(f"\n  Time per step: {ms_per_step:.2f}ms")
print(f"  Throughput: {200/total_time:.2f} steps/s")

print("\n" + "="*70)
print("TARGET: 200 steps < 20s")
print("="*70)

if total_time < 20:
    print(f"\n✅ SUCCESS: {total_time:.2f}s < 20s")
    print(f"   Performance: {ms_per_step:.2f}ms/step - EXCELLENT")
elif total_time < 40:
    print(f"\n⚠️  CLOSE: {total_time:.2f}s (target: <20s)")
    print(f"   Performance: {ms_per_step:.2f}ms/step - NEEDS OPTIMIZATION")
else:
    print(f"\n❌ FAIL: {total_time:.2f}s >> 20s")
    print(f"   Performance: {ms_per_step:.2f}ms/step - TOO SLOW")
    print("\n   Bottleneck analysis needed...")
