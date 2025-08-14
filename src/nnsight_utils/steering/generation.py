"""
NNsight-based steering implementation using clean intervention API.

This module provides steering functionality using nnsight's intervention system,
offering a cleaner alternative to the transformer_lens hook-based approach.
"""

from typing import List, Optional, Union, Tuple
import logging
import numpy as np
import torch

from ..core.activation_extraction import _get_layer_output
from ..core.models import NNsightChatModel
from ..probes.base import ProbeResult
from .strategies import SteeringDispatcher, prepare_steering_vectors, determine_steering_layers, get_steering_config
from .kv_cached_generation import generate_with_kv_cached_steering, generate_with_steering_single
from ..caching.steering_cache import get_cached_steering_tensors

# Set up logger
logger = logging.getLogger(__name__)


def generate_with_steering(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    probe_result: ProbeResult,
    alpha: float = 1.0,
    instruction_pos: Optional[int] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    **kwargs
) -> str:
    """
    Generate text with steering using probe results (method-aware).
    
    This is the main steering interface that automatically configures
    itself based on the probe method and results. Uses optimized
    tensor caching for better performance.
    
    Args:
        model: NNsightChatModel instance
        tokens: Input token tensor
        probe_result: Results from probe training (contains method and vectors)
        alpha: Scaling factor for steering vectors
        instruction_pos: Position after which to apply steering (None = end of prompt)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text string
    """
    # Use the optimized single-prompt generation function
    return generate_with_steering_single(
        model=model,
        tokens=tokens,
        probe_result=probe_result,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        instruction_pos=instruction_pos,
        **kwargs
    )


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
    attention_mask: Optional[torch.Tensor] = None,
) -> List[str]:
    """
    Generate text with steering using nnsight interventions.
    Handles any batch size uniformly with left-padding.
    
    Args:
        model: NNsightChatModel instance
        tokens: Input token tensor (batch_size, seq_len) - should be left-padded
        steering_vectors: Numpy array of steering vectors for each layer
        alpha: Scaling factor for steering vectors
        instruction_pos: Position after which to apply steering (None = end of prompt)
                        Same for all sequences due to left-padding
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        layers: List of layer indices to steer (None = all layers with vectors)
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        attention_mask: Optional attention mask for batched inputs
        
    Returns:
        List of generated text strings (always returns a list)
    """
    batch_size = tokens.size(0)
    
    # Set default instruction position to end of prompt (same for all due to left-padding)
    if instruction_pos is None:
        instruction_pos = tokens.size(1)
    
    # Determine which layers to steer
    if layers is None:
        layers = list(range(len(steering_vectors)))
    
    # Convert steering vectors to tensors on the right device (use tokens device)
    steering_tensors = {
        layer: torch.tensor(
            steering_vectors[i], 
            dtype=model.dtype
        )
        for i, layer in enumerate(layers)
    }
    
    # Prepare generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": model.tokenizer.eos_token_id,
    }
    
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    
    # Add DeepSeek-specific stopping criteria
    if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
        stop_tokens = []
        vocab = model.tokenizer.get_vocab()
        
        if '<｜end▁of▁sentence｜>' in vocab:
            stop_tokens.append(vocab['<｜end▁of▁sentence｜>'])
        if '<｜User｜>' in vocab:
            stop_tokens.append(vocab['<｜User｜>'])
            
        if stop_tokens:
            gen_kwargs["eos_token_id"] = stop_tokens
    
    # Generate with interventions
    with model.model.generate(tokens, **gen_kwargs) as generator:
        
        # Apply steering interventions during generation
        for layer in layers:
            # Get the residual stream output for this layer using architecture detection
            residual = _get_layer_output(model.model, layer)
            
            # Apply steering to all sequences in batch at the same instruction position
            steering_vector = steering_tensors[layer].to(residual.device)
            if steering_vector.dim() == 1:
                steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            
            # Apply to all batch elements at once (same instruction_pos due to left-padding)
            residual[:, instruction_pos:, :] += alpha * steering_vector.squeeze(0).squeeze(0)
        
        # Get the generated output
        output = model.model.generator.output.save()
    
    # Decode all outputs uniformly
    input_length = tokens.size(1)
    results = []
    
    for i in range(batch_size):
        generated_tokens = output[i, input_length:]
        if len(generated_tokens) > 0:
            generated_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            generated_text = ""
        results.append(generated_text)
    
    return results


def generate_steered_batch(
    model: NNsightChatModel,
    prompts: List[str],
    probe_result: ProbeResult,
    alpha: float = 1.0,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    batch_size: Optional[int] = None,
) -> List[str]:
    """
    Generate steered text for a batch of prompts using efficient batching with left-padding.
    
    Args:
        model: NNsightChatModel instance
        prompts: List of prompt strings
        probe_result: Results from probe training (contains method and vectors)
        alpha: Steering strength
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        batch_size: Number of prompts to process at once (None = all at once)
        
    Returns:
        List of generated text strings
    """
    if batch_size is None:
        batch_size = len(prompts)
    
    results = []
    
    config = get_steering_config(probe_result.method)
    layers = determine_steering_layers(probe_result, config)
    vectors = prepare_steering_vectors(probe_result, config)
    
    # Convert and cache
    steering_tensors = {}
    for layer, vector in vectors.items():
        tensor = torch.tensor(
            alpha * vector,  # Apply alpha scaling
            dtype=model.dtype
        )
        steering_tensors[layer] = tensor
    
    # Set up progress bar
    from tqdm import tqdm
    
    progress_bar = tqdm(
        total=len(prompts),
        desc=f"Steering (batch_size={batch_size})",
        unit="prompts",
        ncols=100
    )
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Set pad token (use eos_token_id as pad_token_id if not set)
        if model.tokenizer.pad_token_id is None:
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        
        # Use tokenizer's built-in padding with left-padding for causal models
        model.tokenizer.padding_side = 'left'
        
        # Tokenize all prompts with padding
        batch_encoding = model.tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=False
        )
        
        batch_tokens = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']
        
        # Use the updated batched steering function
        batch_results = generate_with_nnsight_steering(
            model=model,
            tokens=batch_tokens,
            steering_vectors=steering_tensors,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            layers=layers,
            do_sample=True,
            attention_mask=attention_mask,
        )
        
        # Results are already a list
        results.extend(batch_results)
        
        # Update progress bar
        progress_bar.update(len(batch_prompts))
    
    # Close progress bar
    progress_bar.close()
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
        residual = model.model.model.layers[layer].output[0]
        
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
            layer: model.model.model.layers[layer].output[0].save()
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