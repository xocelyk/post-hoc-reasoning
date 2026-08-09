"""
Base classes and interfaces for probe training.

This module defines the common interfaces and data structures used by
different probe training methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProbeResult:
    """Standardized probe training results."""
    
    method: str  # "logistic", "caa-single-layer", "caa-multi-layer"
    vectors: Dict[int, np.ndarray]  # layer -> steering vector
    scores: Dict[int, float]  # layer -> evaluation score
    best_layer: Optional[int] = None  # For single-layer methods
    metadata: Dict[str, Any] = None  # Additional method-specific data
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseProbe(ABC):
    """Abstract base class for probe training methods."""
    
    @abstractmethod
    def train(
        self,
        activations: np.ndarray,
        labels: List[str],
        layers: Optional[List[int]] = None
    ) -> ProbeResult:
        """
        Train probe on activations.
        
        Args:
            activations: Activation array of shape (n_samples, n_layers, d_model)
            labels: List of string labels ("yes"/"no")
            layers: Specific layers to train on (None = all)
            
        Returns:
            ProbeResult containing trained probe data
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        probe_result: ProbeResult,
        test_activations: np.ndarray,
        test_labels: List[str]
    ) -> Dict[int, float]:
        """
        Evaluate probe performance on test data.
        
        Args:
            probe_result: Trained probe results
            test_activations: Test activation array
            test_labels: Test labels
            
        Returns:
            Dictionary mapping layer indices to evaluation scores
        """
        pass


def convert_labels_to_binary(labels: List[str], positive_label: str = "yes") -> np.ndarray:
    """
    Convert string labels to binary format.
    
    Args:
        labels: List of string labels
        positive_label: Label to treat as positive class (1)
        
    Returns:
        Binary numpy array
    """
    return np.array([1 if label == positive_label else 0 for label in labels])


def split_activations_by_label(
    activations: np.ndarray,
    labels: List[str],
    layer: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split activations into positive and negative classes.
    
    Args:
        activations: Activation array
        labels: String labels
        layer: Specific layer to extract (None = keep all layers)
        
    Returns:
        Tuple of (positive_activations, negative_activations)
    """
    pos_mask = np.array([label == "yes" for label in labels])
    neg_mask = np.array([label == "no" for label in labels])
    
    if layer is not None:
        pos_acts = activations[pos_mask, layer, :]
        neg_acts = activations[neg_mask, layer, :]
    else:
        pos_acts = activations[pos_mask]
        neg_acts = activations[neg_mask]
    
    return pos_acts, neg_acts


def compute_similarity_score(
    activations: np.ndarray,
    vector: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute similarity-based evaluation score.
    
    Args:
        activations: Test activations for a single layer
        vector: Probe vector
        labels: Binary labels
        
    Returns:
        AUC score based on dot product similarities
    """
    from sklearn.metrics import roc_auc_score
    
    # Compute similarities
    similarities = activations @ vector
    
    # Calculate AUC
    try:
        auc = roc_auc_score(labels, similarities)
    except ValueError:
        # Handle case where only one class is present
        auc = 0.5
    
    return auc