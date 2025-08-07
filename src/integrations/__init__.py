"""
External service integrations.

This package contains integrations with external services:
- wandb: Weights & Biases experiment tracking
"""

try:
    from .wandb import WandbExperimentLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbExperimentLogger = None

__all__ = ["WandbExperimentLogger", "WANDB_AVAILABLE"]