"""
Steering strategy registry and dispatch logic.

This module defines the mapping between probe methods and steering strategies,
ensuring that the correct steering approach is used for each probe type.
"""

from typing import Dict, List, Optional, Callable, Any
import numpy as np

from ..probes.base import ProbeResult


# Registry mapping probe methods to steering configurations
STEERING_REGISTRY = {
    "logistic-regression": {
        "description": "Steer at all layers using logistic regression coefficients",
        "layers": "all",
        "vector_source": "coefficients",
        "normalization": "none"
    },
    "caa-single-layer": {
        "description": "Steer only at the layer with strongest mean difference",
        "layers": "best_only", 
        "vector_source": "mean_diff",
        "normalization": "none"
    },
    "caa-multi-layer": {
        "description": "Steer at all layers using incremental difference vectors",
        "layers": "all",
        "vector_source": "incremental", 
        "normalization": "rms"
    }
}


def get_steering_config(method_name: str) -> Dict[str, str]:
    """
    Get steering configuration for a given probe method.
    
    Args:
        method_name: Name of the probe method
        
    Returns:
        Dictionary with steering configuration
        
    Raises:
        ValueError: If method is not registered
    """
    if method_name not in STEERING_REGISTRY:
        available = list(STEERING_REGISTRY.keys())
        raise ValueError(f"Unknown steering method: {method_name}. Available: {available}")
    
    return STEERING_REGISTRY[method_name].copy()


def determine_steering_layers(
    probe_result: ProbeResult,
    config: Dict[str, str]
) -> List[int]:
    """
    Determine which layers to apply steering to based on probe results and config.
    
    Args:
        probe_result: Results from probe training
        config: Steering configuration
        
    Returns:
        List of layer indices to steer
    """
    if config["layers"] == "all":
        return sorted(probe_result.vectors.keys())
    elif config["layers"] == "best_only":
        if probe_result.best_layer is not None:
            return [probe_result.best_layer]
        else:
            # Fallback: use highest scoring layer
            best_layer = max(probe_result.scores.keys(), key=lambda k: probe_result.scores[k])
            return [best_layer]
    else:
        raise ValueError(f"Unknown layer selection strategy: {config['layers']}")


def prepare_steering_vectors(
    probe_result: ProbeResult,
    config: Optional[Dict[str, str]] = None
) -> Dict[int, np.ndarray]:
    """
    Prepare steering vectors based on probe method and configuration.
    
    Args:
        probe_result: Results from probe training
        config: Optional steering configuration (auto-detected if None)
        
    Returns:
        Dictionary mapping layer indices to steering vectors
    """
    if config is None:
        config = get_steering_config(probe_result.method)
    
    if probe_result.method == "logistic-regression":
        # Use ALL layers with coefficient vectors
        return probe_result.vectors.copy()
    
    elif probe_result.method == "caa-single-layer":
        # Use ONLY the best layer
        if probe_result.best_layer is not None:
            return {probe_result.best_layer: probe_result.vectors[probe_result.best_layer]}
        else:
            # Fallback: find best layer from scores (earliest wins ties)
            best_layer = max(probe_result.scores.keys(), key=lambda k: (probe_result.scores[k], -k))
            return {best_layer: probe_result.vectors[best_layer]}
    
    elif probe_result.method == "caa-multi-layer":
        # Use ALL layers with incremental vectors
        return probe_result.vectors.copy()
    
    else:
        raise ValueError(f"Unknown steering method: {probe_result.method}")


def validate_steering_setup(
    probe_result: ProbeResult,
    layers_to_steer: List[int]
) -> bool:
    """
    Validate that the steering setup is consistent and valid.
    
    Args:
        probe_result: Results from probe training
        layers_to_steer: List of layers to apply steering to
        
    Returns:
        True if setup is valid
        
    Raises:
        ValueError: If setup is invalid
    """
    # Check that all requested layers have vectors
    missing_layers = []
    for layer in layers_to_steer:
        if layer not in probe_result.vectors:
            missing_layers.append(layer)
    
    if missing_layers:
        raise ValueError(f"Missing steering vectors for layers: {missing_layers}")
    
    # Check method-specific constraints
    if probe_result.method == "caa-single-layer":
        if len(layers_to_steer) > 1:
            raise ValueError(
                f"CAA single-layer method should only steer one layer, "
                f"but got: {layers_to_steer}"
            )
    
    # Check vector dimensions are consistent
    vector_dims = [probe_result.vectors[layer].shape[0] for layer in layers_to_steer]
    if len(set(vector_dims)) > 1:
        raise ValueError(f"Inconsistent vector dimensions: {vector_dims}")
    
    return True


def get_method_description(method_name: str) -> str:
    """
    Get human-readable description of a steering method.
    
    Args:
        method_name: Name of the probe method
        
    Returns:
        Description string
    """
    config = get_steering_config(method_name)
    return config["description"]


def list_available_methods() -> List[str]:
    """
    List all available steering methods.
    
    Returns:
        List of method names
    """
    return list(STEERING_REGISTRY.keys())


def get_method_summary() -> Dict[str, str]:
    """
    Get summary of all available methods.
    
    Returns:
        Dictionary mapping method names to descriptions
    """
    return {
        method: config["description"] 
        for method, config in STEERING_REGISTRY.items()
    }


class SteeringDispatcher:
    """
    Helper class to manage steering dispatch logic.
    """
    
    def __init__(self):
        self.registry = STEERING_REGISTRY.copy()
    
    def register_method(
        self,
        method_name: str,
        description: str,
        layers: str,
        vector_source: str,
        normalization: str = "none"
    ):
        """
        Register a new steering method.
        
        Args:
            method_name: Unique method identifier
            description: Human-readable description
            layers: Layer selection strategy ("all", "best_only")
            vector_source: Vector source type
            normalization: Normalization approach
        """
        self.registry[method_name] = {
            "description": description,
            "layers": layers,
            "vector_source": vector_source,
            "normalization": normalization
        }
    
    def dispatch(self, probe_result: ProbeResult) -> Dict[str, Any]:
        """
        Create complete steering configuration for a probe result.
        
        Args:
            probe_result: Results from probe training
            
        Returns:
            Dictionary with complete steering setup
        """
        config = get_steering_config(probe_result.method)
        layers = determine_steering_layers(probe_result, config)
        vectors = prepare_steering_vectors(probe_result, config)
        
        # Validate setup
        validate_steering_setup(probe_result, layers)
        
        return {
            "method": probe_result.method,
            "config": config,
            "layers": layers,
            "vectors": vectors,
            "num_layers": len(layers),
            "vector_dim": list(vectors.values())[0].shape[0] if vectors else 0
        }