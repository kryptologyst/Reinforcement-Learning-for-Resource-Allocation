#!/usr/bin/env python3
"""
Demo script for RL Resource Allocation project.

This script demonstrates the key features of the project with a quick
training example and visualization.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.envs import SimpleResourceEnv
from src.agents import create_agent
from src.utils import plot_training_curves


def main():
    """Run a quick demo of the RL resource allocation project."""
    print("🎯 RL Resource Allocation Demo")
    print("=" * 40)
    
    # Create environment
    print("📦 Creating environment...")
    env = SimpleResourceEnv(seed=42)
    print(f"   Environment: {env.num_jobs} jobs, {env.total_resources} resources")
    
    # Create agent
    print("🤖 Creating PPO agent...")
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 32,
        "hidden_dims": [32, 32]
    }
    agent = create_agent("ppo", env, config)
    
    # Train agent
    print("🚀 Training agent...")
    metrics = agent.train(total_timesteps=2000)
    
    # Show results
    final_reward = metrics["episode_rewards"][-1]
    avg_reward = np.mean(metrics["episode_rewards"][-10:])  # Last 10 episodes
    
    print(f"✅ Training completed!")
    print(f"   Final reward: {final_reward:.3f}")
    print(f"   Average reward (last 10): {avg_reward:.3f}")
    
    # Evaluate agent
    print("🔍 Evaluating agent...")
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    
    # Show allocation
    demands = obs[:env.num_jobs]
    allocation = action * env.total_resources
    
    print(f"   Job demands: {demands}")
    print(f"   Allocation: {allocation}")
    
    # Calculate metrics
    efficiency = np.minimum(allocation, demands).sum() / env.total_resources
    fairness = 1.0 / (1.0 + np.var(allocation))
    
    print(f"   Efficiency: {efficiency:.3f}")
    print(f"   Fairness: {fairness:.3f}")
    
    # Plot training curve
    print("📊 Generating training curve...")
    plot_training_curves(
        metrics,
        title="Demo Training Progress",
        save_path="demo_training_curve.png"
    )
    print("   Training curve saved as 'demo_training_curve.png'")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("  - Run 'python train.py --agent ppo' for full training")
    print("  - Run 'streamlit run app.py' for interactive interface")
    print("  - Check 'README.md' for detailed documentation")


if __name__ == "__main__":
    main()
