"""
Visualization utilities for RL resource allocation project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_training_curves(
    metrics: Dict[str, List[float]], 
    title: str = "Training Progress",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training curves for rewards and episode lengths.
    
    Args:
        metrics: Dictionary containing training metrics
        title: Plot title
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot episode rewards
    if 'episode_rewards' in metrics:
        rewards = metrics['episode_rewards']
        ax1.plot(rewards, alpha=0.6, label='Episode Rewards')
        
        # Add moving average
        if len(rewards) > 10:
            window = min(100, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            ax1.plot(moving_avg, label=f'Moving Average ({window})', linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    if 'episode_lengths' in metrics:
        lengths = metrics['episode_lengths']
        ax2.plot(lengths, alpha=0.6, label='Episode Lengths')
        
        # Add moving average
        if len(lengths) > 10:
            window = min(100, len(lengths) // 10)
            moving_avg = pd.Series(lengths).rolling(window=window).mean()
            ax2.plot(moving_avg, label=f'Moving Average ({window})', linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_allocation_comparison(
    demands: np.ndarray,
    allocations: List[np.ndarray],
    agent_names: List[str],
    title: str = "Resource Allocation Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of resource allocations across different agents.
    
    Args:
        demands: Job demands array
        allocations: List of allocation arrays from different agents
        agent_names: Names of the agents
        title: Plot title
        save_path: Path to save the plot
    """
    num_jobs = len(demands)
    x = np.arange(num_jobs)
    width = 0.8 / len(agent_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot demands
    ax.bar(x - width/2, demands, width, label='Demands', alpha=0.8, color='red')
    
    # Plot allocations for each agent
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
    for i, (allocation, name) in enumerate(zip(allocations, agent_names)):
        ax.bar(x + width/2 + i*width, allocation, width, 
               label=f'{name} Allocation', alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Job Index')
    ax.set_ylabel('Resources')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_efficiency_fairness_tradeoff(
    efficiency_scores: List[float],
    fairness_scores: List[float],
    agent_names: List[str],
    title: str = "Efficiency vs Fairness Tradeoff",
    save_path: Optional[str] = None
) -> None:
    """
    Plot efficiency vs fairness tradeoff for different agents.
    
    Args:
        efficiency_scores: List of efficiency scores
        fairness_scores: List of fairness scores
        agent_names: Names of the agents
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
    
    for i, (eff, fair, name) in enumerate(zip(efficiency_scores, fairness_scores, agent_names)):
        ax.scatter(eff, fair, s=100, label=name, color=colors[i], alpha=0.7)
        ax.annotate(name, (eff, fair), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Efficiency')
    ax.set_ylabel('Fairness')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add ideal point
    ax.scatter(1.0, 1.0, s=200, marker='*', color='gold', label='Ideal', zorder=5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_interactive_dashboard(
    metrics: Dict[str, List[float]],
    demands: np.ndarray,
    allocations: List[np.ndarray],
    agent_names: List[str]
) -> go.Figure:
    """
    Create an interactive dashboard using Plotly.
    
    Args:
        metrics: Training metrics
        demands: Job demands
        allocations: Agent allocations
        agent_names: Agent names
    
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Rewards', 'Episode Lengths', 
                       'Resource Allocation', 'Efficiency vs Fairness'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot training rewards
    if 'episode_rewards' in metrics:
        rewards = metrics['episode_rewards']
        fig.add_trace(
            go.Scatter(y=rewards, mode='lines', name='Rewards', opacity=0.6),
            row=1, col=1
        )
        
        # Add moving average
        if len(rewards) > 10:
            window = min(100, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(y=moving_avg, mode='lines', name=f'Moving Avg ({window})'),
                row=1, col=1
            )
    
    # Plot episode lengths
    if 'episode_lengths' in metrics:
        lengths = metrics['episode_lengths']
        fig.add_trace(
            go.Scatter(y=lengths, mode='lines', name='Lengths', opacity=0.6),
            row=1, col=2
        )
    
    # Plot resource allocation
    x = list(range(len(demands)))
    fig.add_trace(
        go.Bar(x=x, y=demands, name='Demands', marker_color='red', opacity=0.8),
        row=2, col=1
    )
    
    colors = px.colors.qualitative.Set3
    for i, (allocation, name) in enumerate(zip(allocations, agent_names)):
        fig.add_trace(
            go.Bar(x=x, y=allocation, name=f'{name} Allocation', 
                   marker_color=colors[i % len(colors)], opacity=0.8),
            row=2, col=1
        )
    
    # Plot efficiency vs fairness
    efficiency_scores = [np.minimum(allocation, demands).sum() / demands.sum() 
                        for allocation in allocations]
    fairness_scores = [1.0 / (1.0 + np.var(allocation)) for allocation in allocations]
    
    for i, (eff, fair, name) in enumerate(zip(efficiency_scores, fairness_scores, agent_names)):
        fig.add_trace(
            go.Scatter(x=[eff], y=[fair], mode='markers+text',
                      text=[name], textposition='top center',
                      marker=dict(size=15, color=colors[i % len(colors)])),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="RL Resource Allocation Dashboard",
        showlegend=True
    )
    
    return fig


def plot_learning_curves_comparison(
    all_metrics: Dict[str, Dict[str, List[float]]],
    metric_name: str = "episode_rewards",
    title: str = "Learning Curves Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot learning curves comparison for multiple agents.
    
    Args:
        all_metrics: Dictionary of metrics for each agent
        metric_name: Name of the metric to plot
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_metrics)))
    
    for i, (agent_name, metrics) in enumerate(all_metrics.items()):
        if metric_name in metrics:
            values = metrics[metric_name]
            ax.plot(values, label=agent_name, alpha=0.7, color=colors[i])
            
            # Add moving average
            if len(values) > 10:
                window = min(100, len(values) // 10)
                moving_avg = pd.Series(values).rolling(window=window).mean()
                ax.plot(moving_avg, color=colors[i], linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
