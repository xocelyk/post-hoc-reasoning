"""
Steering and intervention utilities.
"""

from .generation import (
    apply_steering_intervention,
    compute_steering_effects,
    generate_steered_batch,
    generate_with_nnsight_steering,
    generate_with_steering,
    interpolate_steering_vectors,
    normalize_steering_vectors,
    validate_steering_vectors,
)
from .strategies import (
    SteeringDispatcher,
    determine_steering_layers,
    get_method_description,
    get_method_summary,
    get_steering_config,
    list_available_methods,
    prepare_steering_vectors,
    validate_steering_setup,
)

__all__ = [
    # Main steering interface
    "generate_with_steering",
    # Low-level steering functions
    "generate_with_nnsight_steering",
    "generate_steered_batch",
    "apply_steering_intervention",
    # Steering utilities
    "compute_steering_effects",
    "validate_steering_vectors",
    "interpolate_steering_vectors",
    "normalize_steering_vectors",
    # Strategy management
    "SteeringDispatcher",
    "get_steering_config",
    "determine_steering_layers",
    "prepare_steering_vectors",
    "validate_steering_setup",
    "get_method_description",
    "get_method_summary",
    "list_available_methods",
]