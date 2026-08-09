"""
Core NNsight utilities for models, activation extraction, and generation.
"""

from .activation_extraction import (
    analyze_layer_differences,
    batch_get_resid_activations,
    compare_activations,
    extract_layer_activations,
    get_activation_statistics,
    prepare_activations_for_probing,
)
from .generation import (
    batch_generate_text,
    continue_generation,
    generate_text,
    get_model_response,
)
from .models import NNsightChatModel

__all__ = [
    # Models
    "NNsightChatModel",
    # Activation extraction
    "batch_get_resid_activations",
    "extract_layer_activations",
    "compare_activations",
    "get_activation_statistics",
    "prepare_activations_for_probing",
    "analyze_layer_differences",
    # Generation
    "generate_text",
    "batch_generate_text",
    "get_model_response",
    "continue_generation",
]