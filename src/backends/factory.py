"""
Backend factory for creating appropriate backend adapters.
"""

from typing import Optional
from .base_adapter import BackendAdapter
from .transformer_lens.adapter import TransformerLensAdapter
from .nnsight.adapter import NNsightAdapter


def create_backend_adapter(backend_name: str) -> BackendAdapter:
    """Create a backend adapter based on the backend name.
    
    Args:
        backend_name: Name of the backend ("transformer_lens" or "nnsight")
        
    Returns:
        BackendAdapter instance
        
    Raises:
        ValueError: If backend_name is not recognized
    """
    if backend_name == "transformer_lens":
        return TransformerLensAdapter()
    elif backend_name == "nnsight":
        return NNsightAdapter()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Valid options: transformer_lens, nnsight")


def get_backend_for_model(model_name: str) -> str:
    """Determine the best backend for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Backend name ("transformer_lens" or "nnsight")
    """
    # Models that require nnsight
    nnsight_models = [
        "deepseek",
        "DeepSeek",
        "mistral",
        "Mistral",
        "mixtral",
        "Mixtral"
    ]
    
    for pattern in nnsight_models:
        if pattern in model_name:
            return "nnsight"
    
    # Default to transformer_lens for compatibility
    return "transformer_lens"