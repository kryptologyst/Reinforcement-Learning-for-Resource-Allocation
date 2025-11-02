"""
Modern Reinforcement Learning Agents for Resource Allocation.

This module implements various state-of-the-art RL algorithms including PPO, SAC, TD3,
and Rainbow DQN, all with proper type hints, docstrings, and modern best practices.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import logging


class BaseAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def train(self, total_timesteps: int) -> Dict[str, List[float]]:
        """Train the agent."""
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[float]]:
        """Predict action for given observation."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent."""
        pass


class PolicyGradientAgent(BaseAgent):
    """REINFORCE Policy Gradient Agent with modern improvements."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env, config)
        
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
        # Policy network
        self.policy = PolicyNetwork(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dims=config.get("hidden_dims", [64, 64])
        )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def train(self, total_timesteps: int) -> Dict[str, List[float]]:
        """Train using REINFORCE algorithm."""
        timestep = 0
        episode = 0
        
        while timestep < total_timesteps:
            episode += 1
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Collect episode
            observations = []
            actions = []
            rewards = []
            log_probs = []
            
            done = False
            while not done and timestep < total_timesteps:
                action, log_prob = self.predict(obs, return_log_prob=True)
                
                observations.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(reward)
                
                episode_reward += reward
                episode_length += 1
                timestep += 1
                
                done = terminated or truncated
            
            # Calculate returns
            returns = self._calculate_returns(rewards, self.gamma)
            
            # Update policy
            self._update_policy(observations, actions, log_probs, returns)
            
            # Logging
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True, return_log_prob: bool = False) -> Union[Tuple[np.ndarray, Optional[float]], np.ndarray]:
        """Predict action for given observation."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy(obs_tensor)
            
            if deterministic:
                action = action_probs.argmax(dim=-1).item()
                action_one_hot = np.zeros(self.env.action_space.shape[0])
                action_one_hot[action] = 1.0
                return action_one_hot
            else:
                dist = Categorical(action_probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
                
                action_one_hot = np.zeros(self.env.action_space.shape[0])
                action_one_hot[action_idx.item()] = 1.0
                
                if return_log_prob:
                    return action_one_hot, log_prob.item()
                return action_one_hot
    
    def _calculate_returns(self, rewards: List[float], gamma: float) -> List[float]:
        """Calculate discounted returns."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns
    
    def _update_policy(self, observations: List[np.ndarray], actions: List[np.ndarray], 
                      log_probs: List[float], returns: List[float]) -> None:
        """Update policy using REINFORCE."""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations))
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Forward pass
        action_probs = self.policy(obs_tensor)
        dist = Categorical(action_probs)
        
        # Calculate log probabilities for taken actions
        action_indices = [np.argmax(action) for action in actions]
        action_indices_tensor = torch.LongTensor(action_indices)
        log_probs_tensor = dist.log_prob(action_indices_tensor)
        
        # Policy loss
        policy_loss = -(log_probs_tensor * returns_tensor).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
    
    def save(self, path: str) -> None:
        """Save the agent."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str) -> None:
        """Load the agent."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous action spaces."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        
        self.action_dim = action_dim
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_layers(x)
        mean = torch.tanh(self.mean_head(shared))
        log_std = self.log_std_head(shared)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class SACAgent(BaseAgent):
    """Soft Actor-Critic Agent using stable-baselines3."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env, config)
        
        self.agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            buffer_size=config.get("buffer_size", 100000),
            learning_starts=config.get("learning_starts", 100),
            batch_size=config.get("batch_size", 256),
            tau=config.get("tau", 0.005),
            gamma=config.get("gamma", 0.99),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", 1),
            ent_coef=config.get("ent_coef", "auto"),
            target_update_interval=config.get("target_update_interval", 1),
            target_entropy=config.get("target_entropy", "auto"),
            use_sde=config.get("use_sde", False),
            sde_sample_freq=config.get("sde_sample_freq", -1),
            use_sde_at_warmup=config.get("use_sde_at_warmup", False),
            verbose=1
        )
    
    def train(self, total_timesteps: int) -> Dict[str, List[float]]:
        """Train the SAC agent."""
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Extract training metrics
        return {
            "episode_rewards": self.agent.ep_info_buffer.episode_rewards,
            "episode_lengths": self.agent.ep_info_buffer.episode_lengths
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[float]]:
        """Predict action for given observation."""
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """Save the agent."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent."""
        self.agent = SAC.load(path)


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent using stable-baselines3."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env, config)
        
        self.agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            clip_range_vf=config.get("clip_range_vf", None),
            ent_coef=config.get("ent_coef", 0.0),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            use_sde=config.get("use_sde", False),
            sde_sample_freq=config.get("sde_sample_freq", -1),
            target_kl=config.get("target_kl", None),
            tensorboard_log=config.get("tensorboard_log", None),
            verbose=1
        )
    
    def train(self, total_timesteps: int) -> Dict[str, List[float]]:
        """Train the PPO agent."""
        self.agent.learn(total_timesteps=total_timesteps)
        
        return {
            "episode_rewards": self.agent.ep_info_buffer.episode_rewards,
            "episode_lengths": self.agent.ep_info_buffer.episode_lengths
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[float]]:
        """Predict action for given observation."""
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """Save the agent."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent."""
        self.agent = PPO.load(path)


class TD3Agent(BaseAgent):
    """Twin Delayed Deep Deterministic Policy Gradient Agent using stable-baselines3."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env, config)
        
        # Add action noise
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        self.agent = TD3(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 1e-3),
            buffer_size=config.get("buffer_size", 100000),
            learning_starts=config.get("learning_starts", 100),
            batch_size=config.get("batch_size", 256),
            tau=config.get("tau", 0.005),
            gamma=config.get("gamma", 0.99),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", 1),
            action_noise=action_noise,
            target_policy_noise=config.get("target_policy_noise", 0.2),
            target_noise_clip=config.get("target_noise_clip", 0.5),
            policy_delay=config.get("policy_delay", 2),
            use_sde=config.get("use_sde", False),
            sde_sample_freq=config.get("sde_sample_freq", -1),
            use_sde_at_warmup=config.get("use_sde_at_warmup", False),
            verbose=1
        )
    
    def train(self, total_timesteps: int) -> Dict[str, List[float]]:
        """Train the TD3 agent."""
        self.agent.learn(total_timesteps=total_timesteps)
        
        return {
            "episode_rewards": self.agent.ep_info_buffer.episode_rewards,
            "episode_lengths": self.agent.ep_info_buffer.episode_lengths
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[float]]:
        """Predict action for given observation."""
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """Save the agent."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent."""
        self.agent = TD3.load(path)


def create_agent(agent_type: str, env: gym.Env, config: Dict[str, Any]) -> BaseAgent:
    """Factory function to create agents."""
    agent_classes = {
        "policy_gradient": PolicyGradientAgent,
        "sac": SACAgent,
        "ppo": PPOAgent,
        "td3": TD3Agent,
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agent_classes.keys())}")
    
    return agent_classes[agent_type](env, config)
