"""
Logistic regression probe training.

This module implements traditional logistic regression probes using sklearn,
extracting coefficient vectors for use as steering vectors.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .base import BaseProbe, ProbeResult, convert_labels_to_binary


def train_logistic_probes(
    activations: np.ndarray,
    labels: List[str],
    layers: Optional[List[int]] = None,
    C: float = 1.0,
    max_iter: int = 1000,
    normalize_activations: bool = False
) -> Dict[int, LogisticRegression]:
    """
    Train logistic regression classifiers for each layer.
    
    Args:
        activations: Activation array of shape (n_samples, n_layers, d_model)
        labels: List of string labels
        layers: Specific layers to train on (None = all)
        C: Inverse regularization strength
        max_iter: Maximum iterations for solver
        normalize_activations: Whether to normalize activations before training
        
    Returns:
        Dictionary mapping layer indices to trained LogisticRegression models
    """
    if layers is None:
        layers = list(range(activations.shape[1]))
    
    # Convert labels to binary
    binary_labels = convert_labels_to_binary(labels)
    
    probes = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for layer in layers:
            # Extract layer activations
            X = activations[:, layer, :]
            
            # Normalize if requested
            if normalize_activations:
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            
            # Train logistic regression
            clf = LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="lbfgs",
                n_jobs=-1
            )
            clf.fit(X, binary_labels)
            
            probes[layer] = clf
    
    return probes


def extract_steering_vectors(
    probes: Dict[int, LogisticRegression],
    normalize: bool = False
) -> Dict[int, np.ndarray]:
    """
    Extract coefficient vectors from trained logistic regression probes.
    
    Args:
        probes: Dictionary of trained LogisticRegression models
        normalize: Whether to normalize the coefficient vectors
        
    Returns:
        Dictionary mapping layer indices to coefficient vectors
    """
    vectors = {}
    
    for layer, clf in probes.items():
        # Extract coefficients (shape: (1, d_model) for binary classification)
        coef = clf.coef_[0]
        
        # Normalize if requested
        if normalize:
            coef = coef / (np.linalg.norm(coef) + 1e-8)
        
        vectors[layer] = coef
    
    return vectors


def evaluate_logistic_probes(
    probes: Dict[int, LogisticRegression],
    test_activations: np.ndarray,
    test_labels: List[str],
    normalize_activations: bool = False
) -> Dict[int, float]:
    """
    Evaluate logistic regression probes on test data.
    
    Args:
        probes: Dictionary of trained probes
        test_activations: Test activation array
        test_labels: Test labels
        normalize_activations: Whether to normalize activations (should match training)
        
    Returns:
        Dictionary mapping layer indices to AUC scores
    """
    binary_labels = convert_labels_to_binary(test_labels)
    scores = {}
    
    for layer, clf in probes.items():
        # Extract layer activations
        X = test_activations[:, layer, :]
        
        # Normalize if needed (should match training)
        if normalize_activations:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        # Get predictions
        try:
            y_scores = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(binary_labels, y_scores)
        except ValueError:
            # Handle case where only one class is present
            auc = 0.5
        
        scores[layer] = auc
    
    return scores


class LogisticRegressionProbe(BaseProbe):
    """Logistic regression probe implementation."""
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        normalize_activations: bool = False,
        normalize_vectors: bool = False
    ):
        self.C = C
        self.max_iter = max_iter
        self.normalize_activations = normalize_activations
        self.normalize_vectors = normalize_vectors
    
    def train(
        self,
        activations: np.ndarray,
        labels: List[str],
        layers: Optional[List[int]] = None
    ) -> ProbeResult:
        """Train logistic regression probes."""
        # Train probes
        probes = train_logistic_probes(
            activations=activations,
            labels=labels,
            layers=layers,
            C=self.C,
            max_iter=self.max_iter,
            normalize_activations=self.normalize_activations
        )
        
        # Extract coefficient vectors
        vectors = extract_steering_vectors(
            probes,
            normalize=self.normalize_vectors
        )
        
        # Evaluate on training data
        scores = evaluate_logistic_probes(
            probes,
            activations,
            labels,
            normalize_activations=self.normalize_activations
        )
        
        return ProbeResult(
            method="logistic-regression",
            vectors=vectors,
            scores=scores,
            metadata={
                "probes": probes,
                "C": self.C,
                "normalize_activations": self.normalize_activations,
                "normalize_vectors": self.normalize_vectors
            }
        )
    
    def evaluate(
        self,
        probe_result: ProbeResult,
        test_activations: np.ndarray,
        test_labels: List[str]
    ) -> Dict[int, float]:
        """Evaluate logistic regression probes."""
        probes = probe_result.metadata["probes"]
        normalize = probe_result.metadata["normalize_activations"]
        
        return evaluate_logistic_probes(
            probes,
            test_activations,
            test_labels,
            normalize_activations=normalize
        )


# Convenience function for direct use
def train_logistic_probe(
    activations: np.ndarray,
    labels: List[str],
    C: float = 1.0,
    normalize_activations: bool = False,
    normalize_vectors: bool = False
) -> ProbeResult:
    """
    Train logistic regression probes with default settings.
    
    Args:
        activations: Training activations
        labels: Training labels
        C: Inverse regularization strength
        normalize_activations: Whether to normalize activations
        normalize_vectors: Whether to normalize coefficient vectors
        
    Returns:
        ProbeResult with coefficient vectors for all layers
    """
    probe = LogisticRegressionProbe(
        C=C,
        normalize_activations=normalize_activations,
        normalize_vectors=normalize_vectors
    )
    return probe.train(activations, labels)