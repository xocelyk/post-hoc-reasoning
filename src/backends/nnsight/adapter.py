"""
NNsight backend adapter implementation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import gc

from ..base_adapter import BackendAdapter
from .models import NNsightChatModel
from .utils import batch_get_resid_activations
from .steering import generate_with_nnsight_steering


class NNsightAdapter(BackendAdapter):
    """Adapter for NNsight backend."""
    
    def load_model(self, model_name: str, device: str = "auto", dtype: str = "bfloat16", **kwargs) -> NNsightChatModel:
        """Load a model using NNsight."""
        # NNsight uses device_map instead of device
        device_map = device if device != "auto" else "auto"
        return NNsightChatModel(model_name, device_map=device_map, dtype=dtype, **kwargs)
    
    def apply_chat_template(self, model: NNsightChatModel, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages."""
        return model.apply_chat_template(messages)
    
    def extract_activations(
        self, 
        model: NNsightChatModel, 
        prompts: List[str], 
        layers: Optional[List[int]] = None
    ) -> List[List[np.ndarray]]:
        """Extract activations from model for given prompts."""
        # Get model info
        model_info = self.get_model_info(model)
        num_layers = model_info["num_layers"]
        
        if layers is None:
            layers = list(range(num_layers))
        
        # Use batch_get_resid_activations for all prompts at once
        # Returns shape: (num_prompts, num_layers, hidden_size)
        batch_activations = batch_get_resid_activations(model, prompts, layers)
        
        # Reorganize to List[List[np.ndarray]] format
        # From (batch, layers, hidden) to List of prompts, each with List of layer activations
        all_activations = []
        for prompt_idx in range(len(prompts)):
            prompt_activations = []
            for layer_idx in range(len(layers)):
                prompt_activations.append(batch_activations[prompt_idx, layer_idx])
            all_activations.append(prompt_activations)
        
        return all_activations
    
    def generate(
        self,
        model: NNsightChatModel,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        return model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
    
    def generate_with_steering(
        self,
        model: NNsightChatModel,
        prompt: str,
        steering_vectors: List[np.ndarray],
        alpha: float,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text with steering applied."""
        return generate_with_nnsight_steering(
            model,
            prompt,
            steering_vectors,
            list(range(len(steering_vectors))),  # Apply to all layers
            alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def get_model_info(self, model: NNsightChatModel) -> Dict[str, Any]:
        """Get information about the model."""
        # Get number of layers and hidden size from the model
        config = model.model.config
        
        # Different models use different attribute names
        if hasattr(config, 'n_layers'):
            num_layers = config.n_layers
        elif hasattr(config, 'num_hidden_layers'):
            num_layers = config.num_hidden_layers
        else:
            num_layers = len(model.model.model.layers)
        
        if hasattr(config, 'd_model'):
            hidden_size = config.d_model
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            hidden_size = config.hidden_size
        
        return {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "model_name": model.model_name,
            "device": str(model.device_map),
            "dtype": str(model.dtype)
        }
    
    def cleanup(self, model: NNsightChatModel) -> None:
        """Clean up resources for a model."""
        # Garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()