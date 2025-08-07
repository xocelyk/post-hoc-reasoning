"""
Core utilities for post-hoc reasoning experiments.

This package contains shared core functionality used across all backends:
- cache_manager: Experiment caching and persistence
- config: Configuration management and validation
- data_loading: Dataset loading and preprocessing
- parsing_utils: Response parsing utilities
- memory_utils: Memory management utilities
- visualizer: Experiment visualization
"""

from .cache_manager import ExperimentCache, ExperimentConfig, ExperimentManager
from .config import (
    ExperimentRunConfig,
    ModelConfig,
    DatasetConfig,
    SteeringConfig,
    ConfigLoader,
    create_experiment_configs,
    save_default_configs,
    create_default_config,
    suggest_backend_optimization
)
from .data_loading import load_all_datasets
from .parsing_utils import parse_response, filter_think_tags
from .memory_utils import smart_empty_cache, memory_cleanup_context
from .visualizer import create_visualizer

__all__ = [
    # Cache manager
    "ExperimentCache",
    "ExperimentConfig", 
    "ExperimentManager",
    # Config
    "ExperimentRunConfig",
    "ModelConfig",
    "DatasetConfig",
    "SteeringConfig",
    "ConfigLoader",
    "create_experiment_configs",
    "save_default_configs",
    "create_default_config",
    "suggest_backend_optimization",
    # Data loading
    "load_all_datasets",
    # Parsing
    "parse_response",
    "filter_think_tags",
    # Memory
    "smart_empty_cache",
    "memory_cleanup_context",
    # Visualizer
    "create_visualizer"
]