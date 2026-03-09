"""
Ultimate test: Fresh environment, minimal config, 100-step target first.
"""

import time
import sys
from stable_baselines3 import SAC
from envs.uav_multimodal_env import UAVMultimodalEnv
from networks import GAPEncoder


def test_training(num_steps: int, timeout_s: int = 30):
    """Test training with timeout."""
    print(f"\n{'='*70}")
    print(f"Testing {num_steps} steps (timeout: {timeout_s}s)")
    print(f"{'='*70}")

    # FRESH environment each test
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
        buffer_size=50000
    )

    start = time.time()

    try:
        model.learn(
            total_timesteps=num_steps,
            log_interval=None,
            tb_log_name=None
        )
    except KeyboardInterrupt:
        print("\n  ⏹  Interrupted by user")
        return None

    elapsed = time.time() - start

    if elapsed > timeout_s:
        print(f"\n  ❌ TIMEOUT: {elapsed:.2f}s > {timeout_s}s")
        return False
    else:
        print(f"\n  ✅ SUCCESS: {elapsed:.2f}s")
        print(f"     Performance: {elapsed/num_steps*1000:.2f}ms/step")
        return True


# Test progressively
print("="*70)
print("PROGRESSIVE TRAINING TEST")
print("="*70)

results = {}

for num_steps in [50, 100, 150, 200]:
    success = test_training(num_steps, timeout_s=15)
    results[num_steps] = success

    if success is False:
        print(f"\n⚠️  {num_steps} steps failed - stopping test")
        break
    elif success is None:
        print(f"\n⏹  Test interrupted")
        break

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

for num_steps, success in results.items():
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {num_steps:3d} steps: {status}")

# Calculate expected 200-step time based on successful tests
successful_times = [k for k, v in results.items() if v]
if successful_times:
    max_successful = max(successful_times)
    if max_successful >= 50:
        print(f"\n✅ Max successful test: {max_successful} steps")
        print(f"   → 200 steps estimated: <{200/max_successful*15:.1f}s")
