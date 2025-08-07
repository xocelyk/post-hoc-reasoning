"""
Steering methods for post-hoc reasoning experiments.

This package contains various steering methods for modifying model behavior:
- CAA Single Layer: Use best-performing layer
- CAA Layer Incremental: Distribute edits across layers
- Logistic Regression: Use classifier coefficients
"""

from .base import SteeringMethod
from .caa_single_layer import CAASingleLayerSteering
from .caa_incremental import CAALayerIncrementalSteering
from .logistic_steering import LogisticRegressionSteering


def create_steering_method(method_name: str, **kwargs) -> SteeringMethod:
    """Factory function to create steering method instances.
    
    Args:
        method_name: Name of the steering method
        **kwargs: Additional arguments for method initialization
        
    Returns:
        SteeringMethod instance
        
    Raises:
        ValueError: If method_name is not recognized
    """
    if method_name == "caa-single-layer":
        similarity_scores = kwargs.get("similarity_scores", [])
        if not similarity_scores:
            raise ValueError("CAA Single Layer method requires 'similarity_scores' parameter")
        return CAASingleLayerSteering(similarity_scores)
    
    elif method_name == "caa-layer-incremental":
        return CAALayerIncrementalSteering()
    
    elif method_name == "logistic-regression":
        required_params = ["train_activations", "train_labels", "test_activations", "test_labels"]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Logistic Regression method requires '{param}' parameter")
        
        return LogisticRegressionSteering(
            kwargs["train_activations"],
            kwargs["train_labels"], 
            kwargs["test_activations"],
            kwargs["test_labels"]
        )
    
    else:
        valid_methods = ["caa-single-layer", "caa-layer-incremental", "logistic-regression"]
        raise ValueError(f"Unknown steering method: {method_name}. Valid options: {valid_methods}")


# Re-export the factory function and helper functions from methods.py
from .methods import (
    compute_contrastive_vectors_all_layers,
    format_steering_results
)


__all__ = [
    "SteeringMethod",
    "CAASingleLayerSteering",
    "CAALayerIncrementalSteering",
    "LogisticRegressionSteering",
    "create_steering_method",
    "compute_contrastive_vectors_all_layers",
    "format_steering_results"
]