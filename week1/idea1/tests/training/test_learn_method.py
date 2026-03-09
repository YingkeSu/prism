"""
Test: Single learn(200) vs chunked learn(50)*4
"""

import time
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


print("="*70)
print("TESTING: SINGLE vs CHUNKED learn()")
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

# Test 1: Single learn(200)
print("\n" + "="*70)
print("Test 1: Single learn(200) call")
print("="*70)

model = SAC(
    'MultiInputPolicy',
    env,
    learning_rate=3e-4,
    verbose=0,
    policy_kwargs=policy_kwargs,
    buffer_size=50000
)

start = time.time()
model.learn(total_timesteps=200, log_interval=None)
time_single = time.time() - start

print(f"\n✓ Single learn(200): {time_single:.2f}s")

# Test 2: Chunked learn(50)*4
print("\n" + "="*70)
print("Test 2: Chunked learn(50)*4")
print("="*70)

model = SAC(
    'MultiInputPolicy',
    env,
    learning_rate=3e-4,
    verbose=0,
    policy_kwargs=policy_kwargs,
    buffer_size=50000
)

start = time.time()
for _ in range(4):
    model.learn(total_timesteps=50, log_interval=None)
time_chunked = time.time() - start

print(f"\n✓ Chunked learn(50)*4: {time_chunked:.2f}s")

# Summary
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"\nSingle call:   {time_single:.2f}s")
print(f"Chunked calls: {time_chunked:.2f}s")
print(f"Difference:    {abs(time_single - time_chunked):.2f}s")

if time_single > 30:
    print(f"\n⚠️  Single learn(200) is VERY SLOW")
    print(f"   → Possible issue: SB3 logging overhead")
elif time_chunked > 30:
    print(f"\n⚠️  Chunked learn is also SLOW")
    print(f"   → Different issue than learn() method")
else:
    print(f"\n✓ Both methods are fast")
