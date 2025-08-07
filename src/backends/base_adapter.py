"""
Backend adapter interface for experiment runners.

This module defines the interface that backend adapters must implement
to work with the unified experiment runner.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class BackendAdapter(ABC):
    """Abstract base class for backend adapters."""
    
    @abstractmethod
    def load_model(self, model_name: str, device: str = "auto", dtype: str = "bfloat16", **kwargs) -> Any:
        """Load a model using this backend.
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on
            dtype: Data type for model weights
            **kwargs: Additional backend-specific arguments
            
        Returns:
            Model instance for this backend
        """
        pass
    
    @abstractmethod
    def apply_chat_template(self, model: Any, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages.
        
        Args:
            model: Model instance from this backend
            messages: List of chat messages
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def extract_activations(
        self, 
        model: Any, 
        prompts: List[str], 
        layers: Optional[List[int]] = None
    ) -> List[List[np.ndarray]]:
        """Extract activations from model for given prompts.
        
        Args:
            model: Model instance from this backend
            prompts: List of prompts to process
            layers: Layers to extract from (None = all layers)
            
        Returns:
            List of activation arrays per prompt, per layer
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        model: Any,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            model: Model instance from this backend
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_with_steering(
        self,
        model: Any,
        prompt: str,
        steering_vectors: List[np.ndarray],
        alpha: float,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text with steering applied.
        
        Args:
            model: Model instance from this backend
            prompt: Input prompt
            steering_vectors: Steering vectors per layer
            alpha: Steering strength
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text with steering applied
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get information about the model.
        
        Args:
            model: Model instance from this backend
            
        Returns:
            Dictionary with model information (num_layers, hidden_size, etc.)
        """
        pass
    
    @abstractmethod
    def cleanup(self, model: Any) -> None:
        """Clean up resources for a model.
        
        Args:
            model: Model instance to clean up
        """
        pass