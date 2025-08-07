"""
TransformerLens backend adapter implementation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import gc

from ..base_adapter import BackendAdapter
from .models import ChatModel
from .utils import generate_with_steering


class TransformerLensAdapter(BackendAdapter):
    """Adapter for TransformerLens backend."""
    
    def load_model(self, model_name: str, device: str = "auto", dtype: str = "bfloat16", **kwargs) -> ChatModel:
        """Load a model using TransformerLens."""
        return ChatModel(model_name, device=device, dtype=dtype, **kwargs)
    
    def apply_chat_template(self, model: ChatModel, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages."""
        return model.apply_chat_template(messages)
    
    def extract_activations(
        self, 
        model: ChatModel, 
        prompts: List[str], 
        layers: Optional[List[int]] = None
    ) -> List[List[np.ndarray]]:
        """Extract activations from model for given prompts."""
        if layers is None:
            layers = list(range(model.cfg.n_layers))
        
        with torch.no_grad():
            tokens = model.to_tokens(prompts, prepend_bos=True)
            _, cache = model.run_with_cache(tokens, pos_slice=-1)
            
            # Extract activations for each prompt and layer
            all_activations = []
            for prompt_idx in range(len(prompts)):
                prompt_activations = []
                for layer in layers:
                    layer_act = cache["resid_post", layer][prompt_idx]
                    # Convert to float32 numpy array
                    if layer_act.dtype == torch.bfloat16:
                        layer_act = layer_act.float()
                    prompt_activations.append(layer_act.cpu().numpy())
                all_activations.append(prompt_activations)
            
            # Clean up cache
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return all_activations
    
    def generate(
        self,
        model: ChatModel,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        return model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
    
    def generate_with_steering(
        self,
        model: ChatModel,
        prompt: str,
        steering_vectors: List[np.ndarray],
        alpha: float,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text with steering applied."""
        return generate_with_steering(
            model.model,  # Use the underlying HookedTransformer
            prompt,
            steering_vectors,
            list(range(len(steering_vectors))),  # Apply to all layers
            alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def get_model_info(self, model: ChatModel) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "num_layers": model.cfg.n_layers,
            "hidden_size": model.cfg.d_model,
            "model_name": model.model_name,
            "device": str(model.device),
            "dtype": str(model.dtype)
        }
    
    def cleanup(self, model: ChatModel) -> None:
        """Clean up resources for a model."""
        # Clear any hooks
        if hasattr(model.model, 'reset_hooks'):
            model.model.reset_hooks()
        
        # Garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()