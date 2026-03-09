"""
Simplified 2D Navigation Environment for Training
Uses only position (no multi-modal sensors) for fast training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any


class SimpleNavigationEnv(gym.Env):
    """
    Simplified 2D navigation environment
    
    - State: 3D position only
    - Action: 3D velocity
    - Goal: Reach target position
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, max_steps: int = 200):
        super().__init__()
        
        self.max_steps = max_steps
        self.dt = 0.1
        
        # Simple observation: just position
        self.observation_space = spaces.Box(
            low=-20, high=20, shape=(3,), dtype=np.float32
        )
        
        # Action: velocity in 3D
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        self.position = np.zeros(3)
        self.step_count = 0
        self.last_distance = None
        
        # Closer goal
        self.goal_position = np.array([2.0, 2.0, 1.0])
        self.goal_radius = 0.5
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.position = np.zeros(3)
        self.step_count = 0
        self.last_distance = np.linalg.norm(self.position - self.goal_position)
        
        return self.position.copy(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Apply action (velocity)
        velocity = action * 0.5  # Scale down
        self.position = self.position + velocity * self.dt
        self.step_count += 1
        
        # Compute reward
        dist_to_goal = np.linalg.norm(self.position - self.goal_position)
        
        # Strong progress reward
        if self.last_distance is not None:
            progress_reward = (self.last_distance - dist_to_goal) * 100.0
        else:
            progress_reward = 0.0
        
        self.last_distance = dist_to_goal
        
        # Small alive bonus
        alive_bonus = 0.1
        
        # Goal reward
        goal_reward = 0
        if dist_to_goal < self.goal_radius:
            goal_reward = 1000
        
        # Boundary penalty
        boundary_penalty = 0
        if np.any(np.abs(self.position) > 10):
            boundary_penalty = -10
        
        reward = progress_reward + alive_bonus + goal_reward + boundary_penalty
        
        # Check termination
        done = dist_to_goal < self.goal_radius
        truncated = self.step_count >= self.max_steps or np.any(np.abs(self.position) > 10)
        
        info = {
            'distance_to_goal': dist_to_goal,
            'is_success': done
        }
        
        return self.position.copy(), reward, done, truncated, info


def test_simple_env():
    """Test the simplified environment"""
    env = SimpleNavigationEnv(max_steps=200)
    obs, info = env.reset()
    
    print('=== Simple Navigation Environment Test ===')
    print(f'Goal: {env.goal_position}')
    print(f'Initial obs: {obs}')
    
    # Test random agent
    episode_rewards = []
    for ep in range(10):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 200:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f'Episode {ep+1}: reward={episode_reward:.2f}, steps={steps}, success={info["is_success"]}')
    
    print(f'\\nAverage reward: {np.mean(episode_rewards):.2f}')


if __name__ == '__main__':
    test_simple_env()
