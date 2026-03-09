"""
Test log_interval parameter impact on performance.
"""

import time
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


print("="*70)
print("TESTING log_interval IMPACT")
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

for log_interval in [None, 1, 10]:
    print(f"\n{'='*70}")
    print(f"log_interval = {log_interval}")
    print(f"{'='*70}")

    model = SAC(
        'MultiInputPolicy',
        env,
        learning_rate=3e-4,
        verbose=0,
        policy_kwargs=policy_kwargs,
        buffer_size=50000,
        batch_size=8,
        learning_starts=100,
        train_freq=(4, "step"),
        gradient_steps=1
    )

    start = time.time()
    model.learn(
        total_timesteps=200,
        log_interval=log_interval,
        callback=None
    )
    elapsed = time.time() - start

    print(f"\n  200 steps: {elapsed:.2f}s ({elapsed/200*1000:.2f}ms/step)")

    if elapsed > 30:
        print(f"  ⚠️  TOO SLOW - stopping test")
        break

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
