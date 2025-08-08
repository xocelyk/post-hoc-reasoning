"""
Model factory for creating models with different backends.

This module provides a unified interface for creating models using either
transformer_lens or nnsight backends, with automatic fallback and compatibility
detection.
"""

import logging
from typing import Dict, Optional, Union, Any

from models import ChatModel
from nnsight_models import NNsightChatModel, is_model_supported_by_nnsight, get_model_info

logger = logging.getLogger(__name__)


class ModelCreationError(Exception):
    """Exception raised when model creation fails."""
    pass


def create_model(
    model_name: str,
    backend: str = "auto",
    **kwargs
) -> Union[ChatModel, NNsightChatModel]:
    """
    Create a model using the specified backend.
    
    Args:
        model_name: Name of the model to load
        backend: Backend to use ("auto", "nnsight", "transformer_lens")
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        Model instance (ChatModel or NNsightChatModel)
        
    Raises:
        ModelCreationError: If model creation fails with all attempted backends
    """
    logger.info(f"Creating model '{model_name}' with backend '{backend}'")
    
    if backend == "auto":
        return _create_model_auto(model_name, **kwargs)
    elif backend == "nnsight":
        return _create_model_nnsight(model_name, **kwargs)
    elif backend == "transformer_lens":
        return _create_model_transformer_lens(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: auto, nnsight, transformer_lens")


def _create_model_auto(model_name: str, **kwargs) -> Union[ChatModel, NNsightChatModel]:
    """
    Automatically select the best backend for the given model.
    Try nnsight first for broader compatibility, fallback to transformer_lens.
    """
    errors = []
    
    # Special case: For Llama models, try transformer_lens first
    if "llama" in model_name.lower():
        try:
            logger.info(f"Llama model detected. Attempting to load '{model_name}' with transformer_lens backend first")
            model = ChatModel(model_name, **kwargs)
            logger.info(f"Successfully loaded '{model_name}' with transformer_lens backend")
            return model
        except Exception as e:
            error_msg = f"transformer_lens backend failed: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
    
    # Strategy 1: Try nnsight first for broader model support
    if is_model_supported_by_nnsight(model_name):
        try:
            logger.info(f"Attempting to load '{model_name}' with nnsight backend")
            model = NNsightChatModel(model_name, **kwargs)
            logger.info(f"Successfully loaded '{model_name}' with nnsight backend")
            return model
        except Exception as e:
            error_msg = f"nnsight backend failed: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
    
    # Strategy 2: Fallback to transformer_lens
    try:
        logger.info(f"Attempting to load '{model_name}' with transformer_lens backend")
        model = ChatModel(model_name, **kwargs)
        logger.info(f"Successfully loaded '{model_name}' with transformer_lens backend")
        return model
    except Exception as e:
        error_msg = f"transformer_lens backend failed: {str(e)}"
        logger.warning(error_msg)
        errors.append(error_msg)
    
    # Both backends failed
    error_summary = "\n".join([f"  - {error}" for error in errors])
    raise ModelCreationError(
        f"Failed to create model '{model_name}' with any backend:\n{error_summary}"
    )


def _create_model_nnsight(model_name: str, **kwargs) -> NNsightChatModel:
    """Create model using nnsight backend."""
    try:
        # Map common kwargs to nnsight equivalents
        nnsight_kwargs = _map_kwargs_for_nnsight(kwargs)
        
        model = NNsightChatModel(model_name, **nnsight_kwargs)
        logger.info(f"Created '{model_name}' with nnsight backend")
        return model
    except Exception as e:
        raise ModelCreationError(f"Failed to create '{model_name}' with nnsight: {str(e)}")


def _create_model_transformer_lens(model_name: str, **kwargs) -> ChatModel:
    """Create model using transformer_lens backend."""
    try:
        # Map common kwargs to transformer_lens equivalents
        tl_kwargs = _map_kwargs_for_transformer_lens(kwargs)
        
        model = ChatModel(model_name, **tl_kwargs)
        logger.info(f"Created '{model_name}' with transformer_lens backend")
        return model
    except Exception as e:
        raise ModelCreationError(f"Failed to create '{model_name}' with transformer_lens: {str(e)}")


def _map_kwargs_for_nnsight(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Map common kwargs to nnsight-compatible arguments."""
    nnsight_kwargs = {}
    
    # Direct mappings
    if "device" in kwargs:
        nnsight_kwargs["device_map"] = kwargs["device"]
    
    if "dtype" in kwargs:
        nnsight_kwargs["dtype"] = kwargs["dtype"]
    
    if "n_devices" in kwargs:
        # nnsight handles multi-device through device_map
        logger.warning("n_devices not directly supported by nnsight, using device_map instead")
    
    # Additional nnsight-specific arguments
    for key in ["trust_remote_code", "device_map", "torch_dtype"]:
        if key in kwargs:
            nnsight_kwargs[key] = kwargs[key]
    
    return nnsight_kwargs


def _map_kwargs_for_transformer_lens(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Map common kwargs to transformer_lens-compatible arguments."""
    tl_kwargs = {}
    
    # Direct mappings
    for key in ["device", "dtype", "n_devices"]:
        if key in kwargs:
            tl_kwargs[key] = kwargs[key]
    
    # Map device_map to device for transformer_lens
    if "device_map" in kwargs and "device" not in kwargs:
        device_map = kwargs["device_map"]
        if isinstance(device_map, str):
            tl_kwargs["device"] = device_map
        else:
            logger.warning("Complex device_map not supported by transformer_lens")
    
    return tl_kwargs


def get_recommended_backend(model_name: str) -> str:
    """
    Get the recommended backend for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Recommended backend name
    """
    # Model-specific recommendations
    model_lower = model_name.lower()
    
    # Models that work better with specific backends
    if "deepseek" in model_lower:
        return "nnsight"  # DeepSeek models require nnsight
    elif "gpt2" in model_lower and "openai-community" in model_lower:
        return "transformer_lens"  # GPT-2 works well with transformer_lens
    elif any(pattern in model_lower for pattern in ["gemma", "llama", "mistral"]):
        return "auto"  # These models work with both, try nnsight first
    else:
        return "auto"  # Default to auto-detection


def list_supported_models() -> Dict[str, Dict[str, Any]]:
    """
    List models with their backend support information.
    
    Returns:
        Dictionary mapping model names to support information
    """
    supported_models = {
        # Transformer_lens supported models
        "gpt2": {
            "backends": ["transformer_lens", "nnsight"],
            "recommended": "transformer_lens",
            "notes": "Well-supported by transformer_lens"
        },
        "google/gemma-2-9b-it": {
            "backends": ["transformer_lens", "nnsight"],
            "recommended": "auto",
            "notes": "Works with both backends"
        },
        # add llama-2-7b-chat
        "Llama-2-7b-chat": {
            "backends": ["transformer_lens", "nnsight"],
            "recommended": "auto",
            "notes": "Works with both backends"
        },
        # add phi-3-mini-4k-instruct
        "Phi-3-mini-4k-instruct": {
            "backends": ["transformer_lens", "nnsight"],
            "recommended": "auto",
            "notes": "Works with both backends"
        },
        
        # NNsight-only models
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
            "backends": ["nnsight"],
            "recommended": "nnsight",
            "notes": "Requires nnsight for chat template support"
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
            "backends": ["nnsight"],
            "recommended": "nnsight",
            "notes": "Requires nnsight for chat template support"
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
            "backends": ["nnsight"],
            "recommended": "nnsight",
            "notes": "Requires nnsight for chat template support"
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "backends": ["nnsight"],
            "recommended": "nnsight",
            "notes": "Requires nnsight for chat template support"
        },
    }
    
    return supported_models


def validate_model_config(model_name: str, backend: str, **kwargs) -> Dict[str, Any]:
    """
    Validate model configuration before creation.
    
    Args:
        model_name: Name of the model
        backend: Backend to use
        **kwargs: Model arguments
        
    Returns:
        Dictionary with validation results and warnings
    """
    result = {
        "valid": True,
        "warnings": [],
        "recommendations": []
    }
    
    # Check backend compatibility
    supported_models = list_supported_models()
    if model_name in supported_models:
        model_info = supported_models[model_name]
        if backend != "auto" and backend not in model_info["backends"]:
            result["warnings"].append(
                f"Backend '{backend}' may not be supported for '{model_name}'"
            )
            result["recommendations"].append(
                f"Consider using backend '{model_info['recommended']}' instead"
            )
    
    # Check for common configuration issues
    if backend == "transformer_lens" and "device_map" in kwargs:
        result["warnings"].append(
            "device_map not supported by transformer_lens, will be mapped to device"
        )
    
    if backend == "nnsight" and "n_devices" in kwargs:
        result["warnings"].append(
            "n_devices not directly supported by nnsight, use device_map instead"
        )
    
    # Check for DeepSeek models with transformer_lens
    if "deepseek" in model_name.lower() and backend == "transformer_lens":
        result["valid"] = False
        result["warnings"].append(
            "DeepSeek models are not supported by transformer_lens"
        )
        result["recommendations"].append(
            "Use backend 'nnsight' for DeepSeek models"
        )
    
    return result


def get_backend_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available backends.
    
    Returns:
        Dictionary with backend information
    """
    return {
        "transformer_lens": {
            "description": "TransformerLens-based models with hook system",
            "strengths": ["Well-established", "Good for interpretability", "Detailed documentation"],
            "limitations": ["Limited model support", "Complex hook system"],
            "supported_models": ["GPT-2", "Some Gemma models", "Selected models only"]
        },
        "nnsight": {
            "description": "NNsight-based models with broad HuggingFace compatibility",
            "strengths": ["Broad model support", "Clean API", "Any HuggingFace model"],
            "limitations": ["Newer framework", "Less community usage"],
            "supported_models": ["Any HuggingFace model", "DeepSeek", "Llama", "Mistral", "etc."]
        },
        "auto": {
            "description": "Automatic backend selection with fallback",
            "strengths": ["Best compatibility", "Graceful fallbacks", "Easy to use"],
            "limitations": ["Less predictable", "May mask configuration issues"],
            "supported_models": ["All models supported by either backend"]
        }
    }