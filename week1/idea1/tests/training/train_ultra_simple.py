"""
Ultra-simple Training Test - Just to verify training works
"""

import time
from stable_baselines3 import SAC
from envs.simple_2d_env import Simple2DObstacleEnv

print("="*60)
print("ULTRA-SIMPLE TRAINING TEST")
print("="*60)

try:
    env = Simple2DObstacleEnv()

    print(f"Environment created")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create model
    model = SAC('MlpPolicy', env, learning_rate=3e-4, verbose=2)

    print(f"Model created")

    print("Training for 10 steps...")
    start_time = time.time()

    model.learn(total_timesteps=10, log_interval=5)

    elapsed_time = time.time() - start_time

    print(f"\nTraining completed in {elapsed_time:.2f} seconds!")
    print(f"SUCCESS!")

except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
