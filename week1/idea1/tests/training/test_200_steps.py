"""
Test performance at 200+ steps where SAC actually starts updating.
"""

import time
import torch
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


print("="*70)
print("PROFILING TRAINING BEYOND 100 STEPS (SAC UPDATE THRESHOLD)")
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

# Test WITHOUT torch.compile first
print("\n" + "="*70)
print("TEST 1: WITHOUT torch.compile()")
print("="*70)

model = SAC(
    'MultiInputPolicy',
    env,
    learning_rate=3e-4,
    verbose=0,
    policy_kwargs=policy_kwargs,
    buffer_size=10000
)

print(f"\nTraining 200 steps (crossing SAC update threshold)...")
start = time.time()
model.learn(total_timesteps=200, log_interval=1)
time_200_no_compile = time.time() - start
ms_per_step = (time_200_no_compile / 200) * 1000

print(f"\n✓ 200 steps complete: {time_200_no_compile:.2f}s")
print(f"  Time per step: {ms_per_step:.2f}ms")
print(f"  Throughput: {200/time_200_no_compile:.2f} steps/s")

# Test 500 steps to check degradation
print(f"\nTraining additional 300 steps (total 500)...")
start = time.time()
model.learn(total_timesteps=300, log_interval=1)
time_300_more = time.time() - start
ms_per_step_300 = (time_300_more / 300) * 1000

print(f"\n✓ Steps 200-500 (300 steps): {time_300_more:.2f}s")
print(f"  Time per step: {ms_per_step_300:.2f}ms")
print(f"  Degradation: {ms_per_step_300/ms_per_step:.2f}x")

# Test WITH torch.compile
print("\n" + "="*70)
print("TEST 2: WITH torch.compile()")
print("="*70)

model2 = SAC(
    'MultiInputPolicy',
    env,
    learning_rate=3e-4,
    verbose=0,
    policy_kwargs=policy_kwargs,
    buffer_size=10000
)

print("\nApplying torch.compile()...")
start_compile = time.time()
model2.policy.features_extractor = torch.compile(
    model2.policy.features_extractor,
    mode="reduce-overhead",
    fullgraph=False
)
compile_time = time.time() - start_compile
print(f"  Compilation: {compile_time:.2f}s")

print(f"\nTraining 200 steps...")
start = time.time()
model2.learn(total_timesteps=200, log_interval=1)
time_200_with_compile = time.time() - start
ms_per_step_c = (time_200_with_compile / 200) * 1000

print(f"\n✓ 200 steps complete: {time_200_with_compile:.2f}s")
print(f"  Time per step: {ms_per_step_c:.2f}ms")
print(f"  Throughput: {200/time_200_with_compile:.2f} steps/s")

# Summary
print("\n" + "="*70)
print("SUMMARY: 200-STEP PERFORMANCE")
print("="*70)

print(f"\nWITHOUT torch.compile():")
print(f"  200 steps: {time_200_no_compile:.2f}s")
print(f"  {ms_per_step:.2f}ms/step")

print(f"\nWITH torch.compile():")
print(f"  200 steps: {time_200_with_compile:.2f}s")
print(f"  {ms_per_step_c:.2f}ms/step")
print(f"  + compilation: {compile_time:.2f}s")

speedup = time_200_no_compile / time_200_with_compile
print(f"\nSpeedup: {speedup:.2f}x")

print("\n" + "="*70)
print("TARGET VERIFICATION: 200 steps < 20s")
print("="*70)

if time_200_no_compile < 20:
    print(f"\n✅ SUCCESS (without compile): {time_200_no_compile:.2f}s < 20s")
else:
    print(f"\n❌ FAIL (without compile): {time_200_no_compile:.2f}s > 20s")

if time_200_with_compile < 20:
    print(f"✅ SUCCESS (with compile): {time_200_with_compile:.2f}s < 20s")
else:
    print(f"❌ FAIL (with compile): {time_200_with_compile:.2f}s > 20s")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if time_200_no_compile < 20:
    print("\n✓ Training WITHOUT torch.compile() meets target")
    print("  → Remove torch.compile() from train.py")
    print(f"  → 200-step time: {time_200_no_compile:.2f}s")
elif time_200_with_compile < 20:
    print("\n✓ Training WITH torch.compile() meets target")
    print("  → Keep torch.compile() in train.py")
    print(f"  → 200-step time: {time_200_with_compile:.2f}s")
else:
    print(f"\n❌ NEITHER configuration meets 20s target")
    print(f"   Best time: {min(time_200_no_compile, time_200_with_compile):.2f}s")
    print("   → Need further optimization")
