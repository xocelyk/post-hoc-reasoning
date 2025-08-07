"""
Methods package for backend-agnostic algorithms.

This package contains shared algorithms that work across different backends:
- probes: Probe training methods (logistic regression, CAA)
- steering: Steering methods for modifying model behavior
"""

from .steering import (
    SteeringMethod,
    create_steering_method,
    compute_contrastive_vectors_all_layers,
    format_steering_results
)

__all__ = [
    "SteeringMethod",
    "create_steering_method",
    "compute_contrastive_vectors_all_layers",
    "format_steering_results"
]