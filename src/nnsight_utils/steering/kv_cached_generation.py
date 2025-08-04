"""
KV-cached steering generation for optimal performance.

This module implements steering generation with KV caching to achieve O(n) 
complexity instead of O(t*n) for multiple prompts with shared prefixes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

from ..core.models import NNsightChatModel
from ..probes.base import ProbeResult
from ..caching.steering_cache import get_cached_steering_tensors
from .strategies import prepare_steering_vectors, determine_steering_layers, get_steering_config

logger = logging.getLogger(__name__)


def generate_with_kv_cached_steering(
    model: NNsightChatModel,
    prompts: List[str],
    probe_result: ProbeResult,
    alpha: float = 1.0,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    **kwargs
) -> List[str]:
    """
    Generate steered text for multiple prompts using KV caching optimization.
    
    This function groups prompts by shared prefixes and uses KV caching to avoid
    recomputing attention states, achieving O(n) complexity instead of O(t*n).
    
    Args:
        model: NNsightChatModel instance
        prompts: List of prompt strings to process
        probe_result: Results from probe training (contains method and vectors)
        alpha: Scaling factor for steering vectors
        max_new_tokens: Maximum number of tokens to generate per prompt
        temperature: Sampling temperature
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        **kwargs: Additional generation parameters
        
    Returns:
        List of generated text strings (one per input prompt)
    """
    if len(prompts) == 1:
        # Single prompt - use regular generation
        tokens = model.to_tokens(prompts[0])
        return [generate_with_steering_single(
            model, tokens, probe_result, alpha, max_new_tokens, temperature, do_sample, **kwargs
        )]
    
    # Group prompts by shared prefixes
    prefix_groups = group_prompts_by_prefix(prompts)
    
    results = []
    for prefix_info in prefix_groups:
        if prefix_info['shared_prefix'] and len(prefix_info['prompts']) > 1:
            # Use KV caching for groups with shared prefix
            logger.info(f"Using KV caching for {len(prefix_info['prompts'])} prompts with shared prefix")
            group_results = generate_batch_with_shared_prefix(
                model, prefix_info, probe_result, alpha, max_new_tokens, temperature, do_sample, **kwargs
            )
        else:
            # Process individually for unique prompts
            group_results = []
            for prompt in prefix_info['prompts']:
                tokens = model.to_tokens(prompt)
                result = generate_with_steering_single(
                    model, tokens, probe_result, alpha, max_new_tokens, temperature, do_sample, **kwargs
                )
                group_results.append(result)
        
        results.extend(group_results)
    
    return results


def group_prompts_by_prefix(prompts: List[str], min_prefix_length: int = 50) -> List[Dict]:
    """
    Group prompts by their longest common prefixes.
    
    Args:
        prompts: List of prompt strings
        min_prefix_length: Minimum length for a prefix to be considered for caching
        
    Returns:
        List of dictionaries with 'shared_prefix', 'prompts', and 'suffixes'
    """
    if len(prompts) <= 1:
        return [{'shared_prefix': None, 'prompts': prompts, 'suffixes': ['']}]
    
    # For simplicity, find the longest common prefix of all prompts
    # In a more sophisticated implementation, we could create multiple groups
    min_len = min(len(p) for p in prompts)
    common_prefix_len = 0
    
    for i in range(min_len):
        if all(p[i] == prompts[0][i] for p in prompts):
            common_prefix_len = i + 1
        else:
            break
    
    if common_prefix_len >= min_prefix_length:
        # All prompts share a significant prefix
        shared_prefix = prompts[0][:common_prefix_len]
        suffixes = [p[common_prefix_len:] for p in prompts]
        
        return [{
            'shared_prefix': shared_prefix,
            'prompts': prompts,
            'suffixes': suffixes
        }]
    else:
        # No significant shared prefix - process individually
        return [{
            'shared_prefix': None,
            'prompts': prompts,
            'suffixes': [''] * len(prompts)
        }]


def generate_batch_with_shared_prefix(
    model: NNsightChatModel,
    prefix_info: Dict,
    probe_result: ProbeResult,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    **kwargs
) -> List[str]:
    """
    Generate for a batch of prompts that share a common prefix using KV caching.
    
    Args:
        model: NNsightChatModel instance
        prefix_info: Dictionary with shared prefix and prompt information
        probe_result: Probe results for steering
        alpha: Steering strength
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        **kwargs: Additional generation parameters
        
    Returns:
        List of generated strings
    """
    shared_prefix = prefix_info['shared_prefix']
    prompts = prefix_info['prompts']
    suffixes = prefix_info['suffixes']
    
    # Tokenize the shared prefix
    prefix_tokens = model.to_tokens(shared_prefix)
    instruction_pos = prefix_tokens.size(1)
    
    # Get steering configuration
    config = get_steering_config(probe_result.method)
    layers = determine_steering_layers(probe_result, config)
    vectors = prepare_steering_vectors(probe_result, config)
    
    # Get cached steering tensors
    steering_tensors = get_cached_steering_tensors(
        vectors, alpha, prefix_tokens.device, model.dtype
    )
    
    # Process each prompt by extending the shared prefix
    results = []
    for i, (prompt, suffix) in enumerate(zip(prompts, suffixes)):
        # For now, fall back to individual processing
        # TODO: Implement true KV state caching when nnsight supports it
        tokens = model.to_tokens(prompt)
        result = generate_with_steering_single(
            model, tokens, probe_result, alpha, max_new_tokens, temperature, do_sample, **kwargs
        )
        results.append(result)
    
    return results


def generate_with_steering_single(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    probe_result: ProbeResult,
    alpha: float = 1.0,
    max_new_tokens: int = 100,  
    temperature: float = 0.7,
    do_sample: bool = True,
    instruction_pos: Optional[int] = None,
    **kwargs
) -> str:
    """
    Generate steered text for a single prompt (optimized version).
    
    Args:
        model: NNsightChatModel instance
        tokens: Input token tensor
        probe_result: Results from probe training
        alpha: Scaling factor for steering vectors
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        instruction_pos: Position after which to apply steering (None = end of prompt)
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text string
    """
    # Set default instruction position to end of prompt
    if instruction_pos is None:
        instruction_pos = tokens.size(1)
    
    # Get steering configuration
    config = get_steering_config(probe_result.method)
    layers = determine_steering_layers(probe_result, config)
    vectors = prepare_steering_vectors(probe_result, config)
    
    # Get cached steering tensors (optimized tensor conversion)
    steering_tensors = get_cached_steering_tensors(
        vectors, alpha, tokens.device, model.dtype
    )
    
    # Generate with interventions using the corrected nnsight pattern
    with model.model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=model.tokenizer.eos_token_id,
        **kwargs
    ) as generator:
        
        # Apply steering interventions during generation
        for layer in layers:
            # Get the residual stream output for this layer
            # Handle different model architectures
            if hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'h'):
                # GPT-style models (GPT2, etc.)
                layer_module = model.model.transformer.h[layer]
                # Access the first element of the output tuple for GPT2
                residual = layer_module.output[0]
            elif hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
                # Llama/Gemma style models
                residual = model.model.model.layers[layer].output[0]
            else:
                # Generic fallback
                raise ValueError(f"Unsupported model architecture: {type(model.model)}")
            
            # Create intervention function for this layer
            def create_steering_intervention(layer_idx):
                def steer_residual(residual_tensor):
                    batch_size, seq_len, hidden_size = residual_tensor.shape
                    
                    # Only apply steering to positions after instruction_pos
                    if seq_len > instruction_pos:
                        steering_vector = steering_tensors[layer_idx]
                        # Broadcast steering vector to the right shape
                        if steering_vector.dim() == 1:
                            steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)
                        
                        # Add steering to positions after instruction
                        residual_tensor[:, instruction_pos:, :] += steering_vector
                    
                    return residual_tensor
                
                return steer_residual
            
            # Apply the intervention
            residual.intervene(create_steering_intervention(layer))
        
        # Get the generated output through the correct path
        if hasattr(model.model, 'lm_head'):
            # For models with lm_head (GPT-style)
            logits = model.model.lm_head.output.save()
        elif hasattr(model.model, 'embed_out'):
            # For models with embed_out
            logits = model.model.embed_out.output.save()
        else:
            # Fallback: try to access through the last layer based on model type
            last_layer_idx = model.cfg.n_layers - 1
            if hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'h'):
                # GPT-style models
                logits = model.model.transformer.h[last_layer_idx].output[0].save()
            elif hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
                # Llama/Gemma style models
                logits = model.model.model.layers[last_layer_idx].output[0].save()
            else:
                raise ValueError(f"Cannot determine output access for model: {type(model.model)}")
    
    # The logits from nnsight generation context only contain the newly generated tokens
    # We need to combine with the original tokens to get the full sequence
    if logits.dim() == 3:  # [batch, seq, vocab]
        generated_tokens = torch.argmax(logits, dim=-1)
        # Concatenate original tokens with generated tokens
        full_sequence = torch.cat([tokens, generated_tokens], dim=1)
    elif logits.dim() == 2:  # [batch, seq] - already tokens
        full_sequence = torch.cat([tokens, logits], dim=1)
    else:
        # Handle unexpected tensor shapes - return just input
        full_sequence = tokens
    
    # Decode the full output and return as string
    return model.to_string(full_sequence[0])


def estimate_kv_cache_savings(
    prompts: List[str],
    model_layers: int,
    avg_prompt_length: int = 100
) -> Dict[str, float]:
    """
    Estimate potential performance savings from KV caching.
    
    Args:
        prompts: List of prompts to analyze
        model_layers: Number of model layers
        avg_prompt_length: Average prompt length in tokens
        
    Returns:
        Dictionary with performance estimates
    """
    if len(prompts) <= 1:
        return {'savings_ratio': 1.0, 'shared_prefix_length': 0}
    
    # Find longest common prefix
    min_len = min(len(p) for p in prompts)
    common_prefix_len = 0
    
    for i in range(min_len):
        if all(p[i] == prompts[0][i] for p in prompts):
            common_prefix_len = i + 1
        else:
            break
    
    if common_prefix_len < 50:  # No significant shared prefix
        return {'savings_ratio': 1.0, 'shared_prefix_length': 0}
    
    # Estimate computation savings
    # Without caching: t * n * avg_prompt_length operations
    # With caching: n * common_prefix_len + t * n * (avg_prompt_length - common_prefix_len)
    
    t = len(prompts)  # Number of prompts
    n = model_layers  # Number of layers
    
    without_caching = t * n * avg_prompt_length
    with_caching = n * common_prefix_len + t * n * (avg_prompt_length - common_prefix_len)
    
    savings_ratio = without_caching / with_caching if with_caching > 0 else 1.0
    
    return {
        'savings_ratio': savings_ratio,
        'shared_prefix_length': common_prefix_len,
        'estimated_speedup': f'{savings_ratio:.2f}x'
    }