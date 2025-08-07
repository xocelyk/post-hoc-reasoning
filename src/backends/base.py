"""
Abstract base classes for backend implementations.

This module defines the interfaces that all backends must implement,
ensuring consistent behavior across TransformerLens and NNsight implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np


class BaseModel(ABC):
    """Abstract base class for model wrappers."""
    
    @abstractmethod
    def __init__(self, model_name: str, device: str = "auto", dtype: str = "bfloat16", **kwargs):
        """Initialize the model wrapper."""
        pass
    
    @abstractmethod
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        pass
    
    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of layers in the model."""
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden size of the model."""
        pass


class BaseActivationExtractor(ABC):
    """Abstract base class for activation extraction."""
    
    @abstractmethod
    def extract_activations(
        self,
        model: BaseModel,
        prompts: List[str],
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """Extract activations from specified layers."""
        pass


class BaseSteering(ABC):
    """Abstract base class for steering implementations."""
    
    @abstractmethod
    def generate_with_steering(
        self,
        model: BaseModel,
        prompt: str,
        steering_vectors: List[np.ndarray],
        alpha: float,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text with steering applied."""
        pass