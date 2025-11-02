"""
Unit tests for RL resource allocation project.
"""

import pytest
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.envs import ResourceAllocationEnv, SimpleResourceEnv, ComplexResourceEnv
from src.agents import PolicyGradientAgent, create_agent
from src.utils import Config


class TestResourceAllocationEnv:
    """Test cases for ResourceAllocationEnv."""
    
    def test_env_creation(self):
        """Test environment creation."""
        env = ResourceAllocationEnv()
        assert env.num_jobs == 3
        assert env.total_resources == 100
        assert env.max_steps == 10
    
    def test_reset(self):
        """Test environment reset."""
        env = ResourceAllocationEnv(seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.num_jobs * 2 + 1,)  # demands + allocations + step
        assert isinstance(info, dict)
    
    def test_step(self):
        """Test environment step."""
        env = ResourceAllocationEnv(seed=42)
        obs, _ = env.reset()
        
        # Test valid action
        action = np.array([0.3, 0.4, 0.3])  # Normalized allocation
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert 'episode_reward' in info
        assert 'allocation' in info
        assert 'demands' in info
    
    def test_action_space(self):
        """Test action space properties."""
        env = ResourceAllocationEnv()
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (env.num_jobs,)
        assert env.action_space.low.all() == 0.0
        assert env.action_space.high.all() == 1.0
    
    def test_observation_space(self):
        """Test observation space properties."""
        env = ResourceAllocationEnv()
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (env.num_jobs * 2 + 1,)
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        env = ResourceAllocationEnv(seed=42)
        obs, _ = env.reset()
        
        # Test matching reward
        env.reward_type = "matching"
        action = np.array([0.5, 0.3, 0.2])
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)
    
    def test_episode_termination(self):
        """Test episode termination."""
        env = ResourceAllocationEnv(max_steps=3)
        obs, _ = env.reset()
        
        for step in range(5):  # More than max_steps
            action = np.array([0.33, 0.33, 0.34])
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step < 2:  # Before max_steps
                assert not terminated
            else:  # At max_steps
                assert terminated
                break


class TestSimpleResourceEnv:
    """Test cases for SimpleResourceEnv."""
    
    def test_simple_env_creation(self):
        """Test simple environment creation."""
        env = SimpleResourceEnv()
        assert env.num_jobs == 2
        assert env.total_resources == 50
        assert env.max_steps == 5


class TestComplexResourceEnv:
    """Test cases for ComplexResourceEnv."""
    
    def test_complex_env_creation(self):
        """Test complex environment creation."""
        env = ComplexResourceEnv()
        assert env.num_jobs == 5
        assert env.total_resources == 200
        assert env.max_steps == 20
        assert env.noise_level == 0.2


class TestPolicyGradientAgent:
    """Test cases for PolicyGradientAgent."""
    
    def test_agent_creation(self):
        """Test agent creation."""
        env = ResourceAllocationEnv()
        config = {"learning_rate": 0.001, "gamma": 0.99}
        agent = PolicyGradientAgent(env, config)
        
        assert agent.lr == 0.001
        assert agent.gamma == 0.99
        assert agent.env == env
    
    def test_predict(self):
        """Test agent prediction."""
        env = ResourceAllocationEnv(seed=42)
        config = {"learning_rate": 0.001, "gamma": 0.99}
        agent = PolicyGradientAgent(env, config)
        
        obs, _ = env.reset()
        action = agent.predict(obs, deterministic=True)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (env.num_jobs,)
        assert np.isclose(action.sum(), 1.0, atol=1e-6)  # Should be normalized
    
    def test_predict_with_log_prob(self):
        """Test agent prediction with log probability."""
        env = ResourceAllocationEnv(seed=42)
        config = {"learning_rate": 0.001, "gamma": 0.99}
        agent = PolicyGradientAgent(env, config)
        
        obs, _ = env.reset()
        action, log_prob = agent.predict(obs, return_log_prob=True)
        
        assert isinstance(action, np.ndarray)
        assert isinstance(log_prob, float)


class TestAgentFactory:
    """Test cases for agent factory function."""
    
    def test_create_policy_gradient_agent(self):
        """Test creating policy gradient agent."""
        env = ResourceAllocationEnv()
        config = {"learning_rate": 0.001}
        agent = create_agent("policy_gradient", env, config)
        
        assert isinstance(agent, PolicyGradientAgent)
    
    def test_create_invalid_agent(self):
        """Test creating invalid agent type."""
        env = ResourceAllocationEnv()
        config = {}
        
        with pytest.raises(ValueError):
            create_agent("invalid_agent", env, config)


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.get("environment.num_jobs") == 3
        assert config.get("agent.type") == "ppo"
    
    def test_set_and_get(self):
        """Test setting and getting configuration values."""
        config = Config()
        config.set("test.value", 42)
        assert config.get("test.value") == 42
    
    def test_get_with_default(self):
        """Test getting configuration value with default."""
        config = Config()
        value = config.get("nonexistent.key", "default")
        assert value == "default"
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "environment" in config_dict
        assert "agent" in config_dict


class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self):
        """Test basic training loop."""
        env = ResourceAllocationEnv(max_steps=3, seed=42)
        config = {"learning_rate": 0.01, "gamma": 0.99}
        agent = PolicyGradientAgent(env, config)
        
        # Run a few episodes
        for episode in range(3):
            obs, _ = env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                action = agent.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            assert episode_reward is not None
    
    def test_env_agent_compatibility(self):
        """Test environment and agent compatibility."""
        env = ResourceAllocationEnv(seed=42)
        config = {"learning_rate": 0.001}
        agent = PolicyGradientAgent(env, config)
        
        obs, _ = env.reset()
        action = agent.predict(obs)
        
        # Action should be compatible with environment
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


if __name__ == "__main__":
    pytest.main([__file__])
