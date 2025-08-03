"""
Steering methods for contrastive activation analysis.

This module implements different approaches to computing and applying steering vectors
from contrastive activations, including single-layer and layer-incremental methods.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod


class SteeringMethod(ABC):
    """Abstract base class for steering methods."""
    
    @abstractmethod
    def compute_steering_vectors(self, layer_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Compute steering vectors from layer-wise contrastive vectors.
        
        Args:
            layer_vectors: List of contrastive vectors for each layer
            
        Returns:
            List of steering vectors to be applied at each layer
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this steering method."""
        pass


class CAASingleLayerSteering(SteeringMethod):
    """CAA Single Layer Steering: Use best-performing layer without normalization."""
    
    def __init__(self, similarity_scores: List[float]):
        """Initialize with similarity scores for each layer.
        
        Args:
            similarity_scores: Performance scores for each layer
        """
        self.similarity_scores = similarity_scores
        
    def compute_steering_vectors(self, layer_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Select best layer without normalization."""
        # Find best layer with tiebreaker (latest layer wins ties)
        best_score = max(self.similarity_scores)
        best_layers = [i for i, score in enumerate(self.similarity_scores) if score == best_score]
        best_layer_idx = max(best_layers)  # Latest layer wins ties
        
        # Get the best vector without normalization
        best_vector = layer_vectors[best_layer_idx].copy()
        
        # Replicate best vector for all layers (maintains compatibility)
        steering_vectors = [best_vector] * len(layer_vectors)
        
        return steering_vectors
    
    def get_method_name(self) -> str:
        return "caa-single-layer"


class CAALayerIncrementalSteering(SteeringMethod):
    """CAA Layer Incremental Steering: Distribute concept edits across layers."""
    
    def compute_steering_vectors(self, layer_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Compute incremental vectors with RMS normalization."""
        incremental_vectors = []
        
        for i, vec in enumerate(layer_vectors):
            if i == 0:
                # First layer: use the vector as-is
                delta_v = vec.copy()
            else:
                # Later layers: compute incremental difference
                delta_v = vec - layer_vectors[i-1]
            
            # Apply RMS normalization to each incremental vector
            rms = np.sqrt(np.mean(delta_v**2))
            if rms > 0:
                delta_v = delta_v / rms
            
            incremental_vectors.append(delta_v)
        
        return incremental_vectors
    
    def get_method_name(self) -> str:
        return "caa-layer-incremental"


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
    
    else:
        valid_methods = ["caa-single-layer", "caa-layer-incremental"]
        raise ValueError(f"Unknown steering method: {method_name}. Valid options: {valid_methods}")


def compute_contrastive_vectors_all_layers(
    train_activations: List[List[np.ndarray]], 
    train_labels: List[str],
    test_activations: List[List[np.ndarray]], 
    test_labels: List[str],
    normalize_individual: bool = True
) -> Tuple[List[np.ndarray], List[float]]:
    """Compute contrastive vectors and similarity scores for all layers.
    
    Args:
        train_activations: List of activation lists (one per sample, one per layer)
        train_labels: Training labels ("yes" or "no")
        test_activations: List of activation lists for test data
        test_labels: Test labels ("yes" or "no")
        normalize_individual: Whether to normalize individual layer vectors (L2 norm)
        
    Returns:
        Tuple of (layer_vectors, similarity_scores)
    """
    num_layers = len(train_activations[0])
    layer_vectors = []
    similarity_scores = []
    
    for layer_idx in range(num_layers):
        # Extract activations for this layer
        layer_train_activations = [sample[layer_idx] for sample in train_activations]
        layer_test_activations = [sample[layer_idx] for sample in test_activations]
        
        # Compute contrastive vector for this layer
        contrastive_vector = extract_contrastive_vector(
            layer_train_activations, train_labels, normalize=normalize_individual
        )
        layer_vectors.append(contrastive_vector)
        
        # Evaluate this layer's performance
        similarity_score = evaluate_contrastive_vector(
            contrastive_vector, layer_test_activations, test_labels
        )
        similarity_scores.append(similarity_score)
    
    return layer_vectors, similarity_scores


def extract_contrastive_vector(activations: List[np.ndarray], labels: List[str], normalize: bool = True) -> np.ndarray:
    """Extract contrastive activation vector from activations and labels.
    
    Args:
        activations: List of activation arrays for this layer
        labels: List of labels ("yes" or "no")
        normalize: Whether to L2 normalize the vector
        
    Returns:
        Contrastive difference vector (mean_yes - mean_no)
    """
    activations_array = np.array(activations)
    labels_array = np.array(labels)
    
    # Separate activations by class
    yes_mask = labels_array == "yes"
    no_mask = labels_array == "no"
    
    if not np.any(yes_mask) or not np.any(no_mask):
        # If we don't have both classes, return zero vector
        return np.zeros(activations_array.shape[1])
    
    # Compute mean activations for each class
    mean_yes = np.mean(activations_array[yes_mask], axis=0)
    mean_no = np.mean(activations_array[no_mask], axis=0)
    
    # Compute difference vector: mean(yes) - mean(no)
    difference_vector = mean_yes - mean_no
    
    # Normalize the vector if requested (L2 normalization)
    if normalize:
        vector_norm = np.linalg.norm(difference_vector)
        if vector_norm > 0:  # Avoid division by zero
            difference_vector = difference_vector / vector_norm
    
    return difference_vector


def evaluate_contrastive_vector(contrastive_vector: np.ndarray, activations: List[np.ndarray], labels: List[str]) -> float:
    """Evaluate contrastive vector using similarity scores.
    
    Args:
        contrastive_vector: The computed contrastive vector
        activations: Test activations for this layer
        labels: Test labels ("yes" or "no")
        
    Returns:
        Similarity score (higher is better)
    """
    if len(activations) == 0:
        return 0.0
        
    activations_array = np.array(activations)
    labels_array = np.array(labels)
    
    # Compute dot product of each activation with contrastive vector
    similarities = np.dot(activations_array, contrastive_vector)
    
    # Convert labels to binary (1 for "yes", 0 for "no")
    binary_labels = (labels_array == "yes").astype(int)
    
    # Compute correlation between similarities and labels
    if len(set(binary_labels)) < 2:
        # If all labels are the same, return 0
        return 0.0
        
    correlation = np.corrcoef(similarities, binary_labels)[0, 1]
    
    # Return absolute correlation (we care about separation, not direction)
    return abs(correlation) if not np.isnan(correlation) else 0.0


def apply_rms_normalization(vector: np.ndarray) -> np.ndarray:
    """Apply RMS (Root Mean Square) normalization to a vector.
    
    Args:
        vector: Input vector to normalize
        
    Returns:
        RMS-normalized vector
    """
    rms = np.sqrt(np.mean(vector**2))
    if rms > 0:
        return vector / rms
    else:
        return vector


def format_steering_results(steering_vectors: List[np.ndarray], method_name: str) -> Dict[str, Any]:
    """Format steering results for caching and analysis.
    
    Args:
        steering_vectors: Computed steering vectors
        method_name: Name of the steering method used
        
    Returns:
        Dictionary with formatted results (JSON serializable)
    """
    return {
        "method": method_name,
        "num_layers": len(steering_vectors),
        "vector_shapes": [list(vec.shape) for vec in steering_vectors],
        "vector_norms": [float(np.linalg.norm(vec)) for vec in steering_vectors],
        "vector_rms": [float(np.sqrt(np.mean(vec**2))) for vec in steering_vectors],
        # Note: steering_vectors themselves are saved separately via pickle
    }