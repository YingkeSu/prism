"""
Quick diagnostic to find where training gets stuck.
"""

import time
import torch
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


print("="*60)
print("QUICK TRAINING BOTTLENECK DIAGNOSTIC")
print("="*60)

# Test 1: Environment creation
print("\n[1/6] Creating environment...")
start = time.time()
env = UAVMultimodalEnv(max_steps=200)
print(f"✓ Environment created: {time.time() - start:.2f}s")

# Test 2: Reset environment
print("\n[2/6] Testing environment reset...")
start = time.time()
obs, info = env.reset()
print(f"✓ Reset complete: {time.time() - start:.2f}s")
print(f"  Observation keys: {list(obs.keys())}")

# Test 3: Single environment step
print("\n[3/6] Testing single environment step...")
start = time.time()
action = env.action_space.sample()  # Returns array directly
obs, reward, done, truncated, info = env.step(action)
print(f"✓ Single step: {time.time() - start:.4f}s")

# Test 4: Model creation WITHOUT compile
print("\n[4/6] Creating model (NO compile)...")
start = time.time()
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
    buffer_size=10000  # Smaller buffer for faster testing
)
print(f"✓ Model created: {time.time() - start:.2f}s")

# Test 5: Single model.predict call
print("\n[5/6] Testing model.predict()...")
obs, _ = env.reset()
start = time.time()
action = model.predict(obs, deterministic=True)
print(f"✓ predict() call: {time.time() - start:.4f}s")

# Test 6: Training - just 1 step
print("\n[6/6] Testing model.learn() for 1 timestep...")
start = time.time()
model.learn(total_timesteps=1, log_interval=None)
elapsed = time.time() - start
print(f"✓ 1 timestep learned: {elapsed:.2f}s ({elapsed*1000:.2f}ms)")

if elapsed > 5:
    print(f"\n⚠️  WARNING: 1 timestep took {elapsed:.2f}s - this is TOO SLOW!")
    print("    Expected: < 1 second")
else:
    print(f"\n✓ Performance looks good!")

# Test 7: Training ramp - 10, 50, 100 steps
print("\n[BONUS] Testing training ramp...")
for num_steps in [10, 50]:
    print(f"\n  Testing {num_steps} steps...")
    start = time.time()
    model.learn(total_timesteps=num_steps, log_interval=None)
    elapsed = time.time() - start
    time_per_step = (elapsed / num_steps) * 1000
    print(f"  ✓ {num_steps} steps: {elapsed:.2f}s ({time_per_step:.2f}ms/step)")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
