"""
Main training script for RL resource allocation project.

This script provides a comprehensive training pipeline with support for multiple
RL algorithms, configuration management, logging, and visualization.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import gymnasium as gym

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.envs import ResourceAllocationEnv, SimpleResourceEnv, ComplexResourceEnv
from src.agents import create_agent
from src.utils import Config, setup_logging, TrainingLogger, plot_training_curves


def create_environment(config: Config) -> gym.Env:
    """Create environment based on configuration."""
    env_config = config.get("environment", {})
    env_type = env_config.get("env_type", "standard")
    
    if env_type == "simple":
        return SimpleResourceEnv(**env_config)
    elif env_type == "complex":
        return ComplexResourceEnv(**env_config)
    else:
        return ResourceAllocationEnv(**env_config)


def train_agent(
    agent_type: str,
    config: Config,
    loggers: Dict[str, Any],
    save_path: str
) -> Dict[str, List[float]]:
    """Train a single agent."""
    # Create environment
    env = create_environment(config)
    
    # Create agent
    agent_config = config.get("agent", {})
    agent = create_agent(agent_type, env, agent_config)
    
    # Setup training logger
    training_logger = TrainingLogger(loggers)
    
    # Training parameters
    total_timesteps = config.get("training.total_timesteps", 10000)
    eval_freq = config.get("training.eval_freq", 1000)
    save_freq = config.get("training.save_freq", 5000)
    
    loggers['main'].info(f"Starting training with {agent_type} agent")
    loggers['main'].info(f"Total timesteps: {total_timesteps}")
    
    # Train agent
    metrics = agent.train(total_timesteps)
    
    # Save agent
    agent.save(save_path)
    loggers['main'].info(f"Agent saved to {save_path}")
    
    return metrics


def evaluate_agent(
    agent_type: str,
    config: Config,
    model_path: str,
    num_episodes: int = 10
) -> Dict[str, float]:
    """Evaluate a trained agent."""
    # Create environment
    env = create_environment(config)
    
    # Create and load agent
    agent_config = config.get("agent", {})
    agent = create_agent(agent_type, env, agent_config)
    agent.load(model_path)
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    efficiency_scores = []
    fairness_scores = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if 'efficiency' in info:
            efficiency_scores.append(info['efficiency'])
        if 'fairness' in info:
            fairness_scores.append(info['fairness'])
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_efficiency': np.mean(efficiency_scores) if efficiency_scores else 0.0,
        'mean_fairness': np.mean(fairness_scores) if fairness_scores else 0.0
    }


def compare_agents(
    agent_types: List[str],
    config: Config,
    models_dir: str
) -> None:
    """Compare multiple trained agents."""
    results = {}
    
    for agent_type in agent_types:
        model_path = os.path.join(models_dir, f"{agent_type}_model")
        
        if os.path.exists(model_path):
            eval_results = evaluate_agent(agent_type, config, model_path)
            results[agent_type] = eval_results
            
            print(f"\n{agent_type.upper()} Results:")
            print(f"  Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
            print(f"  Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            print(f"  Efficiency: {eval_results['mean_efficiency']:.3f}")
            print(f"  Fairness: {eval_results['mean_fairness']:.3f}")
        else:
            print(f"Model not found for {agent_type}: {model_path}")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agents for resource allocation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--agent", type=str, default="ppo", 
                       choices=["policy_gradient", "ppo", "sac", "td3"],
                       help="Agent type to train")
    parser.add_argument("--timesteps", type=int, default=10000,
                       help="Total training timesteps")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--compare", action="store_true", help="Compare all trained agents")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory containing trained models")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.timesteps:
        config.set("training.total_timesteps", args.timesteps)
    
    # Setup logging
    log_dir = config.get("paths.logs", "logs")
    models_dir = config.get("paths.models", "models")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    loggers = setup_logging(
        log_level=config.get("logging.level", "INFO"),
        log_file=os.path.join(log_dir, f"{args.agent}_training.log"),
        tensorboard_log=os.path.join(log_dir, "tensorboard") if config.get("logging.tensorboard") else None,
        wandb_project=config.get("logging.wandb_project") if config.get("logging.wandb") else None,
        wandb_config=config.to_dict()
    )
    
    # Train agent
    model_path = os.path.join(models_dir, f"{args.agent}_model")
    metrics = train_agent(args.agent, config, loggers, model_path)
    
    # Plot training curves
    plot_training_curves(
        metrics,
        title=f"{args.agent.upper()} Training Progress",
        save_path=os.path.join(log_dir, f"{args.agent}_training_curves.png")
    )
    
    # Evaluation
    if args.eval:
        loggers['main'].info("Running evaluation...")
        eval_results = evaluate_agent(args.agent, config, model_path)
        
        loggers['main'].info(f"Evaluation Results:")
        loggers['main'].info(f"  Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
        loggers['main'].info(f"  Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
        loggers['main'].info(f"  Efficiency: {eval_results['mean_efficiency']:.3f}")
        loggers['main'].info(f"  Fairness: {eval_results['mean_fairness']:.3f}")
    
    # Compare agents
    if args.compare:
        loggers['main'].info("Comparing all trained agents...")
        agent_types = ["policy_gradient", "ppo", "sac", "td3"]
        compare_agents(agent_types, config, models_dir)
    
    # Close loggers
    if 'tensorboard' in loggers:
        loggers['tensorboard'].close()
    if 'wandb' in loggers:
        loggers['wandb'].finish()
    
    loggers['main'].info("Training completed successfully!")


if __name__ == "__main__":
    main()
