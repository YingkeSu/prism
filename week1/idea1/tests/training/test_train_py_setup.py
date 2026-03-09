"""
Direct test of train.py setup but WITHOUT VisualizationCallback.
"""

import time
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


print("="*70)
print("TESTING TRAIN.PY SETUP WITHOUT CALLBACK")
print("="*70)

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
    verbose=0,  # NO verbose output
    tensorboard_log=None,  # NO tensorboard
    policy_kwargs=policy_kwargs,
    buffer_size=50000,
    batch_size=8,
    learning_starts=100,
    train_freq=(4, "step"),
    gradient_steps=1
)

print("\nTraining 200 steps (verbose=0, no callbacks)...")
start = time.time()

model.learn(
    total_timesteps=200,
    log_interval=1,
    callback=None  # NO callback
)

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
else:
    print(f"\n❌ FAIL: {total_time:.2f}s > 20s")
    print("\n   Bottleneck must be in:")
    print("   - VisualizationCallback")
    print("   - TensorBoard logging")
    print("   - Verbose output")
