"""
Configuration management utilities for RL resource allocation project.
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class Config:
    """Configuration manager for RL experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "environment": {
                "num_jobs": 3,
                "total_resources": 100,
                "max_steps": 10,
                "reward_type": "matching",
                "noise_level": 0.1,
                "seed": 42
            },
            "agent": {
                "type": "ppo",
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "batch_size": 64,
                "buffer_size": 100000,
                "hidden_dims": [64, 64]
            },
            "training": {
                "total_timesteps": 10000,
                "eval_freq": 1000,
                "save_freq": 5000,
                "log_freq": 100
            },
            "logging": {
                "level": "INFO",
                "tensorboard": True,
                "wandb": False,
                "wandb_project": "rl-resource-allocation"
            },
            "paths": {
                "models": "models/",
                "logs": "logs/",
                "checkpoints": "checkpoints/"
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        self._merge_config(self.config, loaded_config)
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
