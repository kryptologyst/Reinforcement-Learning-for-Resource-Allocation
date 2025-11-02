"""
Resource Allocation Environment for Reinforcement Learning.

This module implements a gymnasium-compatible environment for multi-job resource allocation,
where an agent must dynamically allocate limited resources to competing tasks to maximize reward.
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns


class ResourceAllocationEnv(gym.Env):
    """
    A gymnasium-compatible environment for resource allocation problems.
    
    The agent must allocate limited resources to multiple jobs with varying demands
    to maximize cumulative reward while balancing efficiency and fairness.
    
    Args:
        num_jobs: Number of jobs competing for resources
        total_resources: Total amount of resources available
        max_steps: Maximum number of steps per episode
        reward_type: Type of reward function ('matching', 'efficiency', 'fairness')
        noise_level: Amount of noise in job demands (0.0 to 1.0)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        num_jobs: int = 3,
        total_resources: int = 100,
        max_steps: int = 10,
        reward_type: str = "matching",
        noise_level: float = 0.1,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_jobs = num_jobs
        self.total_resources = total_resources
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.noise_level = noise_level
        
        # Action space: continuous allocation for each job (normalized)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_jobs,), dtype=np.float32
        )
        
        # Observation space: job demands + current allocations + step count
        obs_dim = num_jobs * 2 + 1  # demands + allocations + step
        self.observation_space = spaces.Box(
            low=0.0, high=total_resources, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset(seed=seed)
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Generate random job demands
        self.job_demands = self.np_random.uniform(
            10, 50, size=self.num_jobs
        ).astype(np.float32)
        
        # Initialize allocations
        self.current_allocations = np.zeros(self.num_jobs, dtype=np.float32)
        
        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.allocation_history = []
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Normalize action to ensure it sums to 1
        action_normalized = action / (action.sum() + 1e-8)
        
        # Convert to actual resource allocation
        allocation = action_normalized * self.total_resources
        
        # Update current allocations
        self.current_allocations = allocation
        
        # Calculate reward based on reward type
        reward = self._calculate_reward(allocation)
        
        # Update episode tracking
        self.step_count += 1
        self.episode_reward += reward
        self.allocation_history.append(allocation.copy())
        
        # Check termination conditions
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Add noise to job demands for next step (simulate changing demands)
        if not terminated:
            noise = self.np_random.normal(0, self.noise_level * self.job_demands)
            self.job_demands = np.clip(
                self.job_demands + noise, 10, 50
            ).astype(np.float32)
        
        info = {
            "episode_reward": self.episode_reward,
            "allocation": allocation,
            "demands": self.job_demands,
            "efficiency": self._calculate_efficiency(allocation),
            "fairness": self._calculate_fairness(allocation)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.job_demands,
            self.current_allocations,
            [self.step_count / self.max_steps]  # Normalized step count
        ])
        return obs.astype(np.float32)
    
    def _calculate_reward(self, allocation: np.ndarray) -> float:
        """Calculate reward based on allocation and job demands."""
        if self.reward_type == "matching":
            # Reward for matching allocation to demand
            mismatch = np.abs(self.job_demands - allocation)
            reward = -mismatch.sum() / self.total_resources
        elif self.reward_type == "efficiency":
            # Reward for efficient resource utilization
            utilization = np.minimum(allocation, self.job_demands).sum()
            reward = utilization / self.total_resources
        elif self.reward_type == "fairness":
            # Reward for fair allocation (minimize variance)
            fair_allocation = self.total_resources / self.num_jobs
            variance = np.var(allocation)
            reward = -variance / (fair_allocation ** 2)
        else:
            # Combined reward
            efficiency = np.minimum(allocation, self.job_demands).sum() / self.total_resources
            fairness = -np.var(allocation) / (self.total_resources / self.num_jobs) ** 2
            reward = 0.7 * efficiency + 0.3 * fairness
        
        return float(reward)
    
    def _calculate_efficiency(self, allocation: np.ndarray) -> float:
        """Calculate resource utilization efficiency."""
        utilized = np.minimum(allocation, self.job_demands).sum()
        return float(utilized / self.total_resources)
    
    def _calculate_fairness(self, allocation: np.ndarray) -> float:
        """Calculate allocation fairness (inverse of variance)."""
        variance = np.var(allocation)
        return float(1.0 / (1.0 + variance))
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        return None
    
    def _render_human(self) -> None:
        """Render environment in human-readable format."""
        print(f"\nStep {self.step_count}/{self.max_steps}")
        print(f"Job Demands: {self.job_demands}")
        print(f"Allocations: {self.current_allocations}")
        print(f"Efficiency: {self._calculate_efficiency(self.current_allocations):.3f}")
        print(f"Fairness: {self._calculate_fairness(self.current_allocations):.3f}")
        print(f"Episode Reward: {self.episode_reward:.3f}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot demands vs allocations
        x = np.arange(self.num_jobs)
        width = 0.35
        
        ax1.bar(x - width/2, self.job_demands, width, label='Demands', alpha=0.8)
        ax1.bar(x + width/2, self.current_allocations, width, label='Allocations', alpha=0.8)
        ax1.set_xlabel('Job Index')
        ax1.set_ylabel('Resources')
        ax1.set_title('Resource Allocation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot allocation history
        if len(self.allocation_history) > 1:
            history = np.array(self.allocation_history)
            for i in range(self.num_jobs):
                ax2.plot(history[:, i], label=f'Job {i}', marker='o')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Allocation')
            ax2.set_title('Allocation History')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to RGB array
        fig.canvas.draw()
        rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return rgb_array
    
    def close(self) -> None:
        """Clean up resources."""
        pass


# Additional utility environments for testing
class SimpleResourceEnv(ResourceAllocationEnv):
    """Simplified version for quick testing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            num_jobs=2,
            total_resources=50,
            max_steps=5,
            **kwargs
        )


class ComplexResourceEnv(ResourceAllocationEnv):
    """More complex version for advanced testing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            num_jobs=5,
            total_resources=200,
            max_steps=20,
            noise_level=0.2,
            **kwargs
        )
