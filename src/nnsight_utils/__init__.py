"""
NNsight utilities for post-hoc reasoning experiments.

This package provides a unified interface for working with NNsight models,
including activation extraction, probe training, and steering interventions.
"""

# Core functionality
from .core import (
    NNsightChatModel,
    batch_get_resid_activations,
    extract_layer_activations,
    generate_text,
    batch_generate_text,
    get_model_response,
)

# Probe training
from .probes import (
    ProbeResult,
    train_logistic_probe,
    train_caa_single_layer,
    train_caa_multi_layer,
    evaluate_probe_performance,
    compare_probe_methods,
)

# Steering
from .steering import (
    generate_with_steering,
    generate_with_nnsight_steering,
    get_steering_config,
    list_available_methods,
)

# Memory management
from .memory import (
    smart_empty_cache,
    memory_cleanup_context,
    suggest_batch_size,
)

# Utilities
from .utils import (
    batch_tokenize,
    create_message,
    create_conversation,
)

# Caching (new)
from .caching import (
    KVCache,
    SharedPrefixCache,
    SteeringVectorCache,
)

# High-level convenience functions
def extract_activations(
    model: NNsightChatModel,
    prompts: list,
    **kwargs
):
    """
    Extract activations from model for given prompts.
    
    Args:
        model: NNsightChatModel instance
        prompts: List of prompt strings
        **kwargs: Additional arguments for batch_get_resid_activations
        
    Returns:
        Activation array
    """
    return batch_get_resid_activations(model, prompts, **kwargs)


def train_probes(
    method: str,
    activations,
    labels: list,
    **kwargs
) -> ProbeResult:
    """
    Train probes using specified method.
    
    Args:
        method: Probe method ("logistic", "caa-single-layer", "caa-multi-layer")
        activations: Activation array
        labels: List of labels
        **kwargs: Method-specific arguments
        
    Returns:
        ProbeResult with trained probe data
    """
    if method == "logistic" or method == "logistic-regression":
        return train_logistic_probe(activations, labels, **kwargs)
    elif method == "caa-single-layer":
        return train_caa_single_layer(activations, labels, **kwargs)
    elif method == "caa-multi-layer":
        return train_caa_multi_layer(activations, labels, **kwargs)
    else:
        raise ValueError(f"Unknown probe method: {method}")


def load_model(model_name: str, **kwargs) -> NNsightChatModel:
    """
    Load an NNsight model.
    
    Args:
        model_name: Name/path of model to load
        **kwargs: Additional arguments for model loading
        
    Returns:
        NNsightChatModel instance
    """
    return NNsightChatModel(model_name, **kwargs)


# Export version
__version__ = "0.1.0"

# Main public API
__all__ = [
    # High-level functions
    "extract_activations",  
    "train_probes",
    "generate_with_steering",
    "load_model",
    
    # Core classes and functions
    "NNsightChatModel",
    "ProbeResult",
    
    # Specific probe methods
    "train_logistic_probe",
    "train_caa_single_layer", 
    "train_caa_multi_layer",
    
    # Evaluation
    "evaluate_probe_performance",
    "compare_probe_methods",
    
    # Activation extraction
    "batch_get_resid_activations",
    "extract_layer_activations",
    
    # Generation
    "generate_text",
    "batch_generate_text",
    "get_model_response",
    "generate_with_nnsight_steering",
    
    # Steering configuration
    "get_steering_config",
    "list_available_methods",
    
    # Memory management
    "smart_empty_cache",
    "memory_cleanup_context",
    "suggest_batch_size",
    
    # Utilities
    "batch_tokenize",
    "create_message",
    "create_conversation",
    
    # Caching (new)
    "KVCache",
    "SharedPrefixCache", 
    "SteeringVectorCache",
]