"""
Streamlit web interface for RL resource allocation project.

This provides an interactive dashboard for training, evaluation, and visualization
of RL agents for resource allocation problems.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.envs import ResourceAllocationEnv, SimpleResourceEnv, ComplexResourceEnv
from src.agents import create_agent
from src.utils import Config, plot_training_curves, plot_allocation_comparison


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RL Resource Allocation",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Reinforcement Learning for Resource Allocation")
    st.markdown("""
    This interactive dashboard allows you to train, evaluate, and visualize RL agents
    for multi-job resource allocation problems.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Environment configuration
    st.sidebar.subheader("Environment")
    env_type = st.sidebar.selectbox(
        "Environment Type",
        ["standard", "simple", "complex"],
        help="Choose the complexity of the resource allocation environment"
    )
    
    num_jobs = st.sidebar.slider("Number of Jobs", 2, 10, 3)
    total_resources = st.sidebar.slider("Total Resources", 50, 500, 100)
    max_steps = st.sidebar.slider("Max Steps per Episode", 5, 50, 10)
    reward_type = st.sidebar.selectbox(
        "Reward Type",
        ["matching", "efficiency", "fairness", "combined"],
        help="Type of reward function to use"
    )
    
    # Agent configuration
    st.sidebar.subheader("Agent")
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        ["ppo", "sac", "td3", "policy_gradient"],
        help="Reinforcement learning algorithm to use"
    )
    
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.2e")
    total_timesteps = st.sidebar.slider("Training Timesteps", 1000, 100000, 10000)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🏃 Training", "📊 Evaluation", "📈 Visualization", "⚙️ Configuration"])
    
    with tab1:
        st.header("Training")
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training agent..."):
                # Create configuration
                config = Config()
                config.set("environment.env_type", env_type)
                config.set("environment.num_jobs", num_jobs)
                config.set("environment.total_resources", total_resources)
                config.set("environment.max_steps", max_steps)
                config.set("environment.reward_type", reward_type)
                config.set("agent.type", agent_type)
                config.set("agent.learning_rate", learning_rate)
                config.set("training.total_timesteps", total_timesteps)
                
                # Create environment
                if env_type == "simple":
                    env = SimpleResourceEnv(
                        num_jobs=num_jobs,
                        total_resources=total_resources,
                        max_steps=max_steps,
                        reward_type=reward_type
                    )
                elif env_type == "complex":
                    env = ComplexResourceEnv(
                        num_jobs=num_jobs,
                        total_resources=total_resources,
                        max_steps=max_steps,
                        reward_type=reward_type
                    )
                else:
                    env = ResourceAllocationEnv(
                        num_jobs=num_jobs,
                        total_resources=total_resources,
                        max_steps=max_steps,
                        reward_type=reward_type
                    )
                
                # Create and train agent
                agent_config = config.get("agent", {})
                agent = create_agent(agent_type, env, agent_config)
                
                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate training (in real implementation, this would be actual training)
                metrics = {"episode_rewards": [], "episode_lengths": []}
                
                for i in range(100):  # Simplified for demo
                    progress_bar.progress(i / 100)
                    status_text.text(f"Training episode {i+1}/100")
                    
                    # Simulate episode
                    obs, _ = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    
                    done = False
                    while not done:
                        action, _ = agent.predict(obs, deterministic=False)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        episode_length += 1
                        done = terminated or truncated
                    
                    metrics["episode_rewards"].append(episode_reward)
                    metrics["episode_lengths"].append(episode_length)
                
                st.success("Training completed!")
                
                # Display training results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Episode Rewards")
                    fig_rewards = go.Figure()
                    fig_rewards.add_trace(go.Scatter(
                        y=metrics["episode_rewards"],
                        mode='lines',
                        name='Episode Rewards',
                        opacity=0.7
                    ))
                    
                    # Add moving average
                    if len(metrics["episode_rewards"]) > 10:
                        window = min(20, len(metrics["episode_rewards"]) // 5)
                        moving_avg = pd.Series(metrics["episode_rewards"]).rolling(window=window).mean()
                        fig_rewards.add_trace(go.Scatter(
                            y=moving_avg,
                            mode='lines',
                            name=f'Moving Average ({window})',
                            line=dict(width=3)
                        ))
                    
                    fig_rewards.update_layout(
                        title="Training Rewards",
                        xaxis_title="Episode",
                        yaxis_title="Reward"
                    )
                    st.plotly_chart(fig_rewards, use_container_width=True)
                
                with col2:
                    st.subheader("Episode Lengths")
                    fig_lengths = go.Figure()
                    fig_lengths.add_trace(go.Scatter(
                        y=metrics["episode_lengths"],
                        mode='lines',
                        name='Episode Lengths',
                        opacity=0.7
                    ))
                    
                    fig_lengths.update_layout(
                        title="Training Episode Lengths",
                        xaxis_title="Episode",
                        yaxis_title="Length"
                    )
                    st.plotly_chart(fig_lengths, use_container_width=True)
    
    with tab2:
        st.header("Evaluation")
        
        # Environment simulation
        st.subheader("Environment Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current State**")
            
            # Create a simple environment for demonstration
            demo_env = ResourceAllocationEnv(
                num_jobs=num_jobs,
                total_resources=total_resources,
                max_steps=max_steps,
                reward_type=reward_type
            )
            
            obs, _ = demo_env.reset()
            
            # Display job demands
            demands = obs[:num_jobs]
            st.write("Job Demands:")
            for i, demand in enumerate(demands):
                st.write(f"Job {i}: {demand:.1f}")
        
        with col2:
            st.write("**Resource Allocation**")
            
            # Interactive allocation sliders
            allocations = []
            total_allocated = 0
            
            for i in range(num_jobs):
                allocation = st.slider(
                    f"Job {i} Allocation",
                    0.0, 1.0, 0.33,
                    key=f"allocation_{i}"
                )
                allocations.append(allocation)
                total_allocated += allocation
            
            # Normalize allocations
            if total_allocated > 0:
                allocations = [a / total_allocated for a in allocations]
            
            # Calculate metrics
            allocation_array = np.array(allocations) * total_resources
            efficiency = np.minimum(allocation_array, demands).sum() / total_resources
            fairness = 1.0 / (1.0 + np.var(allocation_array))
            
            st.metric("Efficiency", f"{efficiency:.3f}")
            st.metric("Fairness", f"{fairness:.3f}")
        
        # Visualization
        st.subheader("Allocation Visualization")
        
        fig = go.Figure()
        
        x = list(range(num_jobs))
        
        fig.add_trace(go.Bar(
            x=x,
            y=demands,
            name='Demands',
            marker_color='red',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            x=x,
            y=allocation_array,
            name='Allocations',
            marker_color='blue',
            opacity=0.8
        ))
        
        fig.update_layout(
            title="Resource Allocation vs Demands",
            xaxis_title="Job Index",
            yaxis_title="Resources",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Visualization")
        
        # Performance comparison
        st.subheader("Performance Comparison")
        
        # Simulate data for different agents
        agents = ["PPO", "SAC", "TD3", "Policy Gradient"]
        colors = px.colors.qualitative.Set3
        
        fig_comparison = go.Figure()
        
        for i, agent in enumerate(agents):
            # Simulate training data
            episodes = list(range(100))
            rewards = np.random.normal(0, 1, 100).cumsum() + np.random.normal(10, 2, 100)
            
            fig_comparison.add_trace(go.Scatter(
                x=episodes,
                y=rewards,
                mode='lines',
                name=agent,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig_comparison.update_layout(
            title="Learning Curves Comparison",
            xaxis_title="Episode",
            yaxis_title="Cumulative Reward"
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Efficiency vs Fairness tradeoff
        st.subheader("Efficiency vs Fairness Tradeoff")
        
        # Simulate data
        efficiency_scores = np.random.uniform(0.6, 0.95, len(agents))
        fairness_scores = np.random.uniform(0.4, 0.9, len(agents))
        
        fig_tradeoff = go.Figure()
        
        for i, agent in enumerate(agents):
            fig_tradeoff.add_trace(go.Scatter(
                x=[efficiency_scores[i]],
                y=[fairness_scores[i]],
                mode='markers+text',
                text=[agent],
                textposition='top center',
                marker=dict(size=15, color=colors[i % len(colors)]),
                name=agent
            ))
        
        fig_tradeoff.update_layout(
            title="Efficiency vs Fairness Tradeoff",
            xaxis_title="Efficiency",
            yaxis_title="Fairness",
            xaxis=dict(range=[0.5, 1.0]),
            yaxis=dict(range=[0.3, 1.0])
        )
        
        st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    with tab4:
        st.header("Configuration")
        
        st.subheader("Current Configuration")
        
        config_dict = {
            "Environment": {
                "Type": env_type,
                "Number of Jobs": num_jobs,
                "Total Resources": total_resources,
                "Max Steps": max_steps,
                "Reward Type": reward_type
            },
            "Agent": {
                "Type": agent_type,
                "Learning Rate": learning_rate,
                "Training Timesteps": total_timesteps
            }
        }
        
        st.json(config_dict)
        
        # Export configuration
        if st.button("Export Configuration"):
            config_yaml = yaml.dump(config_dict, default_flow_style=False)
            st.download_button(
                label="Download Configuration",
                data=config_yaml,
                file_name="config.yaml",
                mime="text/yaml"
            )
        
        st.subheader("Environment Information")
        st.markdown("""
        **Resource Allocation Environment:**
        - **Standard**: Balanced environment with moderate complexity
        - **Simple**: Simplified environment for quick testing
        - **Complex**: Advanced environment with higher complexity
        
        **Reward Types:**
        - **Matching**: Rewards for matching allocation to demand
        - **Efficiency**: Rewards for efficient resource utilization
        - **Fairness**: Rewards for fair allocation (low variance)
        - **Combined**: Balanced combination of efficiency and fairness
        """)


if __name__ == "__main__":
    main()
