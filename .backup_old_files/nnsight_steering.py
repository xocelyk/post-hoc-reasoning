"""
NNsight-based steering implementation using clean intervention API.

This module provides steering functionality using nnsight's intervention system,
offering a cleaner alternative to the transformer_lens hook-based approach.
"""

from typing import List, Optional, Union
import logging
import numpy as np
import torch

from nnsight_models import NNsightChatModel

# Set up logger
logger = logging.getLogger(__name__)


def generate_with_nnsight_steering(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    steering_vectors: np.ndarray,
    alpha: float = 1.0,
    instruction_pos: Optional[int] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    layers: Optional[List[int]] = None,
    do_sample: bool = True,
) -> str:
    """
    Generate text with steering using nnsight interventions.
    
    Args:
        model: NNsightChatModel instance
        tokens: Input token tensor
        steering_vectors: Numpy array of steering vectors for each layer
        alpha: Scaling factor for steering vectors
        instruction_pos: Position after which to apply steering (None = end of prompt) 
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        layers: List of layer indices to steer (None = all layers with vectors)
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        
    Returns:
        Generated text string
    """
    # Set default instruction position to end of prompt
    if instruction_pos is None:
        instruction_pos = tokens.size(1)
    
    # Determine which layers to steer
    if layers is None:
        layers = list(range(len(steering_vectors)))
    
    # Convert steering vectors to tensors on the right device
    steering_tensors = {
        layer: torch.tensor(
            steering_vectors[layer], 
            device=tokens.device,
            dtype=model.dtype
        )
        for layer in layers
    }
    
    # Use the custom generation method from nnsight_models.py
    # which already has a generation loop we can modify
    # Clone tokens to avoid modifying input
    toks = tokens.clone().to(tokens.device)
    
    # Convert steering vectors to tensors
    steering_tensors_list = [steering_tensors[layer] for layer in layers]
    
    # Custom generation loop with steering
    for _ in range(max_new_tokens):
        with model.model.trace(toks):
            # Apply steering to each layer
            for i, layer in enumerate(layers):
                layer_output = model.model.model.layers[layer].output
                steering_vec = steering_tensors_list[i]
                
                # Add steering vector to the last position
                # Note: layer_output shape is [batch, seq_len, hidden]
                if steering_vec.dim() == 1:
                    steering_vec = steering_vec.unsqueeze(0).unsqueeze(0)
                
                # Only steer the last token position (the one being generated)
                layer_output[:, -1:, :] = layer_output[:, -1:, :] + alpha * steering_vec
            
            # Get logits from the language model head
            if hasattr(model.model, 'lm_head'):
                logits = model.model.lm_head.output.save()
            elif hasattr(model.model, 'embed_out'):
                logits = model.model.embed_out.output.save()
            else:
                # For other architectures, try to find the output
                logits = model.model.output.logits.save()
        
        # Get logits for the last position
        next_token_logits = logits[:, -1, :]
        
        if do_sample and temperature > 0:
            # Apply temperature and sample
            probs = (next_token_logits / temperature).softmax(dim=-1)
            next_tok = torch.multinomial(probs, 1)
        else:
            # Greedy decoding
            next_tok = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Ensure next_tok is on the same device as toks
        next_tok = next_tok.to(toks.device)
        
        # Append to sequence
        toks = torch.cat([toks, next_tok], dim=-1)
        
        # Check for EOS
        if model.tokenizer.eos_token_id is not None and (next_tok == model.tokenizer.eos_token_id).all():
            break
    
    # Return only the generated portion as string
    generated_tokens = toks[:, tokens.size(1):]
    result = model.to_string(generated_tokens)
    
    # Handle both single string and list outputs
    if isinstance(result, list):
        return result[0] if result else ""
    return result


def generate_steered_batch(
    model: NNsightChatModel,
    prompts: List[str],
    steering_vectors: np.ndarray,
    alpha: float = 1.0,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    layers: Optional[List[int]] = None,
    batch_size: int = 1,
) -> List[str]:
    """
    Generate steered text for a batch of prompts.
    
    Args:
        model: NNsightChatModel instance
        prompts: List of prompt strings
        steering_vectors: Numpy array of steering vectors
        alpha: Steering strength
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        layers: Layers to apply steering to
        batch_size: Number of prompts to process at once
        
    Returns:
        List of generated text strings
    """
    results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        for prompt in batch_prompts:
            # Tokenize individual prompt
            tokens = model.to_tokens(prompt)
            
            # Generate with steering
            generated = generate_with_nnsight_steering(
                model=model,
                tokens=tokens,
                steering_vectors=steering_vectors,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                layers=layers,
            )
            
            results.append(generated)
    
    return results


def apply_steering_intervention(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    layer: int,
    steering_vector: torch.Tensor,
    alpha: float = 1.0,
    positions: Optional[slice] = None,
) -> torch.Tensor:
    """
    Apply a steering intervention to a specific layer and return activations.
    
    Args:
        model: NNsightChatModel instance
        tokens: Input tokens
        layer: Layer index to intervene on
        steering_vector: Steering vector to add
        alpha: Steering strength
        positions: Slice of positions to affect (None = all positions)
        
    Returns:
        Modified activations from the specified layer
    """
    with model.model.trace(tokens):
        # Get residual activations
        residual = model.model.model.layers[layer].output
        
        # Apply intervention
        def steer_activation(activation_tensor):
            if positions is not None:
                activation_tensor[:, positions, :] += alpha * steering_vector
            else:
                activation_tensor += alpha * steering_vector
            return activation_tensor
        
        # Apply intervention and save result
        steered_residual = residual.intervene(steer_activation).save()
    
    return steered_residual


def compute_steering_effects(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    steering_vectors: np.ndarray,
    alphas: List[float],
    layers: Optional[List[int]] = None,
) -> dict:
    """
    Analyze the effects of different steering strengths and layers.
    
    Args:
        model: NNsightChatModel instance
        tokens: Input tokens
        steering_vectors: Array of steering vectors
        alphas: List of alpha values to test
        layers: Layers to test (None = all)
        
    Returns:
        Dictionary with analysis results
    """
    if layers is None:
        layers = list(range(len(steering_vectors)))
    
    results = {
        'baseline': {},
        'steered': {},
        'differences': {}
    }
    
    # Get baseline activations (no steering)
    with model.model.trace(tokens):
        baseline_activations = {
            layer: model.model.model.layers[layer].output.save()
            for layer in layers
        }
    
    results['baseline'] = {
        layer: act.detach().cpu().numpy() 
        for layer, act in baseline_activations.items()
    }
    
    # Test different alpha values
    for alpha in alphas:
        results['steered'][alpha] = {}
        results['differences'][alpha] = {}
        
        for layer in layers:
            steering_vector = torch.tensor(
                steering_vectors[layer],
                device=tokens.device,
                dtype=model.dtype
            )
            
            # Apply steering and get activations
            steered_acts = apply_steering_intervention(
                model, tokens, layer, steering_vector, alpha
            )
            
            steered_numpy = steered_acts.detach().cpu().numpy()
            baseline_numpy = results['baseline'][layer]
            
            results['steered'][alpha][layer] = steered_numpy
            results['differences'][alpha][layer] = steered_numpy - baseline_numpy
    
    return results


def validate_steering_vectors(
    steering_vectors: np.ndarray,
    expected_layers: int,
    expected_dim: int,
) -> bool:
    """
    Validate that steering vectors have the correct shape and properties.
    
    Args:
        steering_vectors: Array of steering vectors
        expected_layers: Expected number of layers
        expected_dim: Expected dimension size
        
    Returns:
        True if vectors are valid
        
    Raises:
        ValueError: If vectors are invalid
    """
    if steering_vectors.ndim != 2:
        raise ValueError(f"Steering vectors must be 2D, got shape {steering_vectors.shape}")
    
    n_layers, dim = steering_vectors.shape
    
    if n_layers != expected_layers:
        raise ValueError(f"Expected {expected_layers} layers, got {n_layers}")
    
    if dim != expected_dim:
        raise ValueError(f"Expected dimension {expected_dim}, got {dim}")
    
    # Check for NaN or infinite values
    if not np.isfinite(steering_vectors).all():
        raise ValueError("Steering vectors contain NaN or infinite values")
    
    # Check if vectors are all zeros (might indicate a problem)
    if np.allclose(steering_vectors, 0):
        logger.debug("Warning: All steering vectors are close to zero")
    
    return True


def interpolate_steering_vectors(
    vector1: np.ndarray,
    vector2: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Interpolate between two steering vectors.
    
    Args:
        vector1: First steering vector
        vector2: Second steering vector  
        alpha: Interpolation factor (0 = vector1, 1 = vector2)
        
    Returns:
        Interpolated steering vector
    """
    if vector1.shape != vector2.shape:
        raise ValueError(f"Vector shapes must match: {vector1.shape} vs {vector2.shape}")
    
    return (1 - alpha) * vector1 + alpha * vector2


def normalize_steering_vectors(
    steering_vectors: np.ndarray,
    method: str = "layer_norm"
) -> np.ndarray:
    """
    Normalize steering vectors using different methods.
    
    Args:
        steering_vectors: Array of steering vectors
        method: Normalization method ("layer_norm", "global_norm", "unit")
        
    Returns:
        Normalized steering vectors
    """
    if method == "layer_norm":
        # Normalize each layer's vector to unit length
        norms = np.linalg.norm(steering_vectors, axis=1, keepdims=True)
        return steering_vectors / (norms + 1e-8)
    
    elif method == "global_norm":
        # Normalize by the global norm across all layers
        global_norm = np.linalg.norm(steering_vectors)
        return steering_vectors / (global_norm + 1e-8)
    
    elif method == "unit":
        # Normalize to unit vectors per layer
        return steering_vectors / (np.linalg.norm(steering_vectors, axis=1, keepdims=True) + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")