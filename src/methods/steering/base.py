"""
Abstract base class for steering methods.

This module defines the interface that all steering methods must implement.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


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