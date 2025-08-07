"""
CAA Single Layer Steering implementation.

This method uses the best-performing layer without normalization.
"""

from typing import List
import numpy as np
from .base import SteeringMethod


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
        # Find best layer with tiebreaker (earliest layer wins ties)
        best_score = max(self.similarity_scores)
        best_layers = [i for i, score in enumerate(self.similarity_scores) if score == best_score]
        best_layer_idx = min(best_layers)  # EARLIEST layer wins ties
        
        # Get the best vector without normalization
        best_vector = layer_vectors[best_layer_idx].copy()
        
        # Replicate best vector for all layers (maintains compatibility)
        steering_vectors = [best_vector] * len(layer_vectors)
        
        return steering_vectors
    
    def get_method_name(self) -> str:
        return "caa-single-layer"