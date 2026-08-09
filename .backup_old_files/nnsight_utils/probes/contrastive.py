"""
Contrastive probe methods (CAA - Contrastive Activation Analysis).

This module implements various contrastive probe training methods including
single-layer and multi-layer CAA approaches.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .base import BaseProbe, ProbeResult, compute_similarity_score, split_activations_by_label


def compute_mean_differences(
    activations: np.ndarray,
    labels: List[str],
    layers: Optional[List[int]] = None
) -> Dict[int, np.ndarray]:
    """
    Compute mean difference vectors (mean_yes - mean_no) for each layer.
    
    Args:
        activations: Activation array of shape (n_samples, n_layers, d_model)
        labels: List of string labels
        layers: Specific layers to compute (None = all)
        
    Returns:
        Dictionary mapping layer indices to difference vectors
    """
    if layers is None:
        layers = list(range(activations.shape[1]))
    
    mean_diffs = {}
    
    for layer in layers:
        # Split activations by label
        pos_acts, neg_acts = split_activations_by_label(activations, labels, layer)
        
        # Compute mean difference
        if len(pos_acts) > 0 and len(neg_acts) > 0:
            mean_pos = np.mean(pos_acts, axis=0)
            mean_neg = np.mean(neg_acts, axis=0)
            mean_diffs[layer] = mean_pos - mean_neg
        else:
            # Handle edge case where one class is missing
            mean_diffs[layer] = np.zeros(activations.shape[2])
    
    return mean_diffs


def evaluate_mean_differences(
    mean_diffs: Dict[int, np.ndarray],
    test_activations: np.ndarray,
    test_labels: List[str]
) -> Dict[int, float]:
    """
    Evaluate mean difference vectors using similarity scores.
    
    Args:
        mean_diffs: Dictionary of mean difference vectors
        test_activations: Test activation array
        test_labels: Test labels
        
    Returns:
        Dictionary mapping layer indices to AUC scores
    """
    scores = {}
    binary_labels = np.array([1 if label == "yes" else 0 for label in test_labels])
    
    for layer, vector in mean_diffs.items():
        layer_acts = test_activations[:, layer, :]
        score = compute_similarity_score(layer_acts, vector, binary_labels)
        scores[layer] = score
    
    return scores


def find_best_layer(
    mean_diffs: Dict[int, np.ndarray],
    test_activations: np.ndarray,
    test_labels: List[str]
) -> Tuple[int, Dict[int, float]]:
    """
    Find the layer with the highest similarity score.
    
    Args:
        mean_diffs: Dictionary of mean difference vectors
        test_activations: Test activation array
        test_labels: Test labels
        
    Returns:
        Tuple of (best_layer_index, all_scores)
    """
    scores = evaluate_mean_differences(mean_diffs, test_activations, test_labels)
    
    # Find best layer (with tiebreaker favoring EARLIER layers)
    best_layer = max(scores.keys(), key=lambda k: (scores[k], -k))
    
    return best_layer, scores


def compute_incremental_vectors(
    mean_diffs: Dict[int, np.ndarray],
    normalize: bool = False
) -> Dict[int, np.ndarray]:
    """
    Create incremental difference vectors WITHOUT normalization.
    
    This implements the CAA multi-layer approach where each layer gets
    an incremental update from the previous layer.
    
    Args:
        mean_diffs: Dictionary of mean difference vectors
        normalize: Whether to apply RMS normalization per layer (DISABLED for CAA)
        
    Returns:
        Dictionary of incremental vectors
    """
    layers = sorted(mean_diffs.keys())
    incremental_vectors = {}
    
    # First layer is just the mean difference
    incremental_vectors[layers[0]] = mean_diffs[layers[0]].copy()
    
    # Subsequent layers get the difference from previous layer
    for i in range(1, len(layers)):
        curr_layer = layers[i]
        prev_layer = layers[i-1]
        
        incremental_vectors[curr_layer] = (
            mean_diffs[curr_layer] - mean_diffs[prev_layer]
        )
    
    # CAA multi-layer should NOT use normalization
    # Apply RMS normalization if requested (but defaults to False for CAA)
    if normalize:
        for layer, vector in incremental_vectors.items():
            rms_norm = np.sqrt(np.mean(vector ** 2))
            if rms_norm > 0:
                incremental_vectors[layer] = vector / rms_norm
    
    return incremental_vectors


class CAASingleLayerProbe(BaseProbe):
    """Contrastive probe that selects the single best layer."""
    
    def train(
        self,
        activations: np.ndarray,
        labels: List[str],
        layers: Optional[List[int]] = None
    ) -> ProbeResult:
        """Train CAA single-layer probe."""
        # Compute mean differences
        mean_diffs = compute_mean_differences(activations, labels, layers)
        
        # For training, we'll use the training data itself for layer selection
        # In practice, you might want to use a validation set
        best_layer, scores = find_best_layer(mean_diffs, activations, labels)
        
        # Return only the best layer's vector
        return ProbeResult(
            method="caa-single-layer",
            vectors={best_layer: mean_diffs[best_layer]},
            scores=scores,
            best_layer=best_layer,
            metadata={"all_mean_diffs": mean_diffs}
        )
    
    def evaluate(
        self,
        probe_result: ProbeResult,
        test_activations: np.ndarray,
        test_labels: List[str]
    ) -> Dict[int, float]:
        """Evaluate CAA single-layer probe."""
        # For single-layer, we only evaluate the selected layer
        return evaluate_mean_differences(
            probe_result.vectors,
            test_activations,
            test_labels
        )


class CAAMultiLayerProbe(BaseProbe):
    """Contrastive probe using incremental vectors across all layers."""
    
    def __init__(self, normalize: bool = False):
        self.normalize = normalize
    
    def train(
        self,
        activations: np.ndarray,
        labels: List[str],
        layers: Optional[List[int]] = None
    ) -> ProbeResult:
        """Train CAA multi-layer probe."""
        # Compute mean differences
        mean_diffs = compute_mean_differences(activations, labels, layers)
        
        # Compute incremental vectors
        incremental_vectors = compute_incremental_vectors(
            mean_diffs, 
            normalize=self.normalize
        )
        
        # Evaluate performance
        scores = evaluate_mean_differences(
            incremental_vectors,
            activations,
            labels
        )
        
        return ProbeResult(
            method="caa-multi-layer",
            vectors=incremental_vectors,
            scores=scores,
            metadata={
                "mean_diffs": mean_diffs,
                "normalized": self.normalize
            }
        )
    
    def evaluate(
        self,
        probe_result: ProbeResult,
        test_activations: np.ndarray,
        test_labels: List[str]
    ) -> Dict[int, float]:
        """Evaluate CAA multi-layer probe."""
        return evaluate_mean_differences(
            probe_result.vectors,
            test_activations,
            test_labels
        )


# Convenience functions for direct use
def train_caa_single_layer(
    activations: np.ndarray,
    labels: List[str],
    test_activations: Optional[np.ndarray] = None,
    test_labels: Optional[List[str]] = None
) -> ProbeResult:
    """
    Train a CAA single-layer probe.
    
    Args:
        activations: Training activations
        labels: Training labels
        test_activations: Optional test data for layer selection
        test_labels: Optional test labels
        
    Returns:
        ProbeResult with single best layer
    """
    probe = CAASingleLayerProbe()
    result = probe.train(activations, labels)
    
    # If test data provided, use it to select best layer
    if test_activations is not None and test_labels is not None:
        mean_diffs = result.metadata["all_mean_diffs"]
        best_layer, scores = find_best_layer(
            mean_diffs, 
            test_activations, 
            test_labels
        )
        result.best_layer = best_layer
        result.vectors = {best_layer: mean_diffs[best_layer]}
        result.scores = scores
    
    return result


def train_caa_multi_layer(
    activations: np.ndarray,
    labels: List[str],
    normalize: bool = False
) -> ProbeResult:
    """
    Train a CAA multi-layer probe with incremental vectors.
    
    Args:
        activations: Training activations
        labels: Training labels
        normalize: Whether to apply RMS normalization (disabled for CAA)
        
    Returns:
        ProbeResult with incremental vectors for all layers
    """
    probe = CAAMultiLayerProbe(normalize=normalize)
    return probe.train(activations, labels)