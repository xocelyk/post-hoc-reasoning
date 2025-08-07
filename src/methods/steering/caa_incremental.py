"""
CAA Layer Incremental Steering implementation.

This method distributes concept edits across layers using incremental differences.
"""

from typing import List
import numpy as np
from .base import SteeringMethod


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