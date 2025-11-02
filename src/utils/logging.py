"""
Logging utilities for RL resource allocation project.
"""

import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    tensorboard_log: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Setup logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        tensorboard_log: Path to tensorboard logs
        wandb_project: Weights & Biases project name
        wandb_config: Weights & Biases configuration
    
    Returns:
        Dictionary containing logger instances
    """
    loggers = {}
    
    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    loggers['main'] = logging.getLogger('main')
    
    # Setup TensorBoard
    if tensorboard_log:
        os.makedirs(tensorboard_log, exist_ok=True)
        loggers['tensorboard'] = SummaryWriter(tensorboard_log)
    
    # Setup Weights & Biases
    if wandb_project:
        wandb.init(
            project=wandb_project,
            config=wandb_config or {}
        )
        loggers['wandb'] = wandb
    
    return loggers


class TrainingLogger:
    """Logger for training metrics and progress."""
    
    def __init__(self, loggers: Dict[str, Any]):
        self.loggers = loggers
        self.metrics = {}
    
    def log_episode(self, episode: int, reward: float, length: int, **kwargs) -> None:
        """Log episode metrics."""
        self.metrics.setdefault('episode_rewards', []).append(reward)
        self.metrics.setdefault('episode_lengths', []).append(length)
        
        # Log to console
        if 'main' in self.loggers:
            self.loggers['main'].info(
                f"Episode {episode}: Reward={reward:.2f}, Length={length}"
            )
        
        # Log to TensorBoard
        if 'tensorboard' in self.loggers:
            self.loggers['tensorboard'].add_scalar('Episode/Reward', reward, episode)
            self.loggers['tensorboard'].add_scalar('Episode/Length', length, episode)
        
        # Log to Weights & Biases
        if 'wandb' in self.loggers:
            self.loggers['wandb'].log({
                'episode': episode,
                'reward': reward,
                'length': length,
                **kwargs
            })
    
    def log_training_step(self, step: int, loss: float, **kwargs) -> None:
        """Log training step metrics."""
        # Log to TensorBoard
        if 'tensorboard' in self.loggers:
            self.loggers['tensorboard'].add_scalar('Training/Loss', loss, step)
        
        # Log to Weights & Biases
        if 'wandb' in self.loggers:
            self.loggers['wandb'].log({
                'step': step,
                'loss': loss,
                **kwargs
            })
    
    def log_evaluation(self, episode: int, eval_reward: float, **kwargs) -> None:
        """Log evaluation metrics."""
        # Log to TensorBoard
        if 'tensorboard' in self.loggers:
            self.loggers['tensorboard'].add_scalar('Evaluation/Reward', eval_reward, episode)
        
        # Log to Weights & Biases
        if 'wandb' in self.loggers:
            self.loggers['wandb'].log({
                'episode': episode,
                'eval_reward': eval_reward,
                **kwargs
            })
    
    def close(self) -> None:
        """Close all loggers."""
        if 'tensorboard' in self.loggers:
            self.loggers['tensorboard'].close()
        
        if 'wandb' in self.loggers:
            self.loggers['wandb'].finish()
