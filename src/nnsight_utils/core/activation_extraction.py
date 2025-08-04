"""
Activation extraction utilities using NNsight.

This module provides functions for extracting model activations using nnsight's
clean tracing API, maintaining compatibility with the existing caching system.
"""

import gc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .models import NNsightChatModel


def batch_get_resid_activations(
    model: NNsightChatModel, 
    prompts: List[str],
    layers: Optional[List[int]] = None,
    position: str = "last"
) -> np.ndarray:
    """
    Extract residual stream activations for a batch of prompts using nnsight.
    
    Args:
        model: NNsightChatModel instance
        prompts: List of prompt strings
        layers: List of layer indices to extract (None for all layers)
        position: Position to extract ("last" for final token, "all" for all positions)
        
    Returns:
        Numpy array of shape (n_prompts, n_layers, d_model) containing activations
    """
    # Tokenize prompts
    tokens = model.to_tokens(prompts)
    
    # Determine layers to extract
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    
    # Extract activations using nnsight tracing
    with model.model.trace(tokens):
        # Extract residual activations from specified layers
        if position == "last":
            # Extract only the final position (most common case)
            residuals = {
                layer: model.model.model.layers[layer].output[0][:, -1].save()
                for layer in layers
            }
        elif position == "all":
            # Extract all positions
            residuals = {
                layer: model.model.model.layers[layer].output[0].save()
                for layer in layers
            }
        else:
            raise ValueError(f"Unknown position: {position}. Use 'last' or 'all'")
    
    # Convert to numpy format matching original implementation
    n_prompts = len(prompts)
    n_layers = len(layers)
    d_model = model.cfg.d_model
    
    if position == "last":
        activations = np.zeros((n_prompts, n_layers, d_model))
        for i, layer in enumerate(layers):
            layer_acts = residuals[layer].detach().float().cpu().numpy()
            activations[:, i, :] = layer_acts
    else:  # position == "all"
        # For "all" positions, we return the full sequence
        seq_len = tokens.shape[1]
        activations = np.zeros((n_prompts, seq_len, n_layers, d_model))
        for i, layer in enumerate(layers):
            layer_acts = residuals[layer].detach().float().cpu().numpy()
            activations[:, :, i, :] = layer_acts
    
    # Clean up GPU memory
    del residuals
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    
    return activations


def extract_layer_activations(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    layer: int,
    position_slice: Optional[slice] = None
) -> torch.Tensor:
    """
    Extract activations from a specific layer.
    
    Args:
        model: NNsightChatModel instance
        tokens: Input token tensor
        layer: Layer index to extract from
        position_slice: Slice of positions to extract (None for all)
        
    Returns:
        Tensor of activations from the specified layer
    """
    with model.model.trace(tokens):
        if position_slice is not None:
            activations = model.model.model.layers[layer].output[0][:, position_slice].save()
        else:
            activations = model.model.model.layers[layer].output[0].save()
    
    return activations.detach()


def compare_activations(
    activations1: np.ndarray,
    activations2: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compare two sets of activations using various similarity metrics.
    
    Args:
        activations1: First set of activations
        activations2: Second set of activations  
        metric: Similarity metric ("cosine", "euclidean", "dot")
        
    Returns:
        Array of similarity scores
    """
    if activations1.shape != activations2.shape:
        raise ValueError(f"Activation shapes must match: {activations1.shape} vs {activations2.shape}")
    
    if metric == "cosine":
        # Cosine similarity
        dot_product = np.sum(activations1 * activations2, axis=-1)
        norm1 = np.linalg.norm(activations1, axis=-1)
        norm2 = np.linalg.norm(activations2, axis=-1)
        similarities = dot_product / (norm1 * norm2 + 1e-8)
    elif metric == "euclidean":
        # Negative Euclidean distance (higher = more similar)
        similarities = -np.linalg.norm(activations1 - activations2, axis=-1)
    elif metric == "dot":
        # Dot product
        similarities = np.sum(activations1 * activations2, axis=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarities


def get_activation_statistics(activations: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute statistics for activations across different dimensions.
    
    Args:
        activations: Activation array of shape (n_samples, n_layers, d_model)
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {}
    
    # Per-layer statistics
    stats["layer_means"] = np.mean(activations, axis=(0, 2))  # Average across samples and features
    stats["layer_stds"] = np.std(activations, axis=(0, 2))
    stats["layer_norms"] = np.linalg.norm(activations, axis=2).mean(axis=0)
    
    # Per-sample statistics
    stats["sample_norms"] = np.linalg.norm(activations, axis=(1, 2))  # Norm across layers and features
    stats["sample_means"] = np.mean(activations, axis=(1, 2))
    
    # Global statistics
    stats["global_mean"] = np.mean(activations)
    stats["global_std"] = np.std(activations)
    stats["global_min"] = np.min(activations)
    stats["global_max"] = np.max(activations)
    
    return stats


def prepare_activations_for_probing(
    activations: np.ndarray,
    labels: List[str],
    layer: int,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare activations for probe training.
    
    Args:
        activations: Activation array of shape (n_samples, n_layers, d_model)
        labels: List of string labels
        layer: Layer index to extract
        normalize: Whether to normalize activations
        
    Returns:
        Tuple of (X, y) for training a probe
    """
    # Extract activations from specific layer
    X = activations[:, layer, :]
    
    # Normalize if requested
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    # Convert labels to binary (assuming "yes"/"no" labels)
    y = np.array([1 if label == "yes" else 0 for label in labels])
    
    return X, y


def analyze_layer_differences(
    activations: np.ndarray,
    labels: List[str],
    layers: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Analyze differences between layers for different label classes.
    
    Args:
        activations: Activation array of shape (n_samples, n_layers, d_model)
        labels: List of string labels  
        layers: List of layer indices to analyze (None for all)
        
    Returns:
        Dictionary with analysis results
    """
    if layers is None:
        layers = list(range(activations.shape[1]))
    
    # Split by labels
    yes_mask = np.array([label == "yes" for label in labels])
    no_mask = np.array([label == "no" for label in labels])
    
    yes_acts = activations[yes_mask]
    no_acts = activations[no_mask]
    
    results = {}
    
    # Compute layer-wise differences
    layer_diffs = []
    layer_distances = []
    
    for layer in layers:
        yes_layer = yes_acts[:, layer, :]
        no_layer = no_acts[:, layer, :]
        
        # Mean difference
        mean_diff = np.mean(yes_layer, axis=0) - np.mean(no_layer, axis=0)
        layer_diffs.append(np.linalg.norm(mean_diff))
        
        # Distance between centroids
        yes_centroid = np.mean(yes_layer, axis=0)
        no_centroid = np.mean(no_layer, axis=0)
        distance = np.linalg.norm(yes_centroid - no_centroid)
        layer_distances.append(distance)
    
    results["layer_differences"] = np.array(layer_diffs)
    results["layer_distances"] = np.array(layer_distances)
    results["best_layer"] = layers[np.argmax(layer_distances)]
    
    return results