"""
Text generation utilities using NNsight.

This module provides basic text generation functions without steering,
used as building blocks for more complex steering operations.
"""

from typing import List, Optional, Union

import torch

from .models import NNsightChatModel


def generate_text(
    model: NNsightChatModel,
    prompt: Union[str, torch.Tensor],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    **kwargs
) -> str:
    """
    Generate text using the model without any steering.
    
    Args:
        model: NNsightChatModel instance
        prompt: Input prompt string or token tensor
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling (vs greedy decoding)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text string
    """
    # Convert prompt to tokens if needed
    if isinstance(prompt, str):
        tokens = model.to_tokens(prompt, prepend_bos=False)
    else:
        tokens = prompt
    
    # Prepare generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": model.tokenizer.eos_token_id,
    }
    
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
        
    # Update with any additional kwargs
    gen_kwargs.update(kwargs)
    
    # Generate using nnsight
    with model.model.generate(tokens, **gen_kwargs) as generator:
        output = generator.output.save()
    
    # Decode and return
    return model.to_string(output[0])


def batch_generate_text(
    model: NNsightChatModel,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    batch_size: Optional[int] = None,
    **kwargs
) -> List[str]:
    """
    Generate text for multiple prompts with optional batching.
    
    Args:
        model: NNsightChatModel instance
        prompts: List of input prompts
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        batch_size: Process prompts in batches of this size (None = all at once)
        **kwargs: Additional generation parameters
        
    Returns:
        List of generated text strings
    """
    if batch_size is None:
        batch_size = len(prompts)
    
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Process each prompt in the batch
        # Note: NNsight may require individual processing for complex generation
        for prompt in batch_prompts:
            generated = generate_text(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
            results.append(generated)
    
    return results


def get_model_response(
    model: NNsightChatModel,
    messages: List[dict],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Get model response for a chat-formatted conversation.
    
    Args:
        model: NNsightChatModel instance
        messages: List of message dictionaries with 'role' and 'content'
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional generation parameters
        
    Returns:
        Generated response text
    """
    # Apply chat template
    prompt = model.apply_chat_template(messages)
    
    # Generate response
    response = generate_text(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs
    )
    
    return response


def continue_generation(
    model: NNsightChatModel,
    tokens: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    **kwargs
) -> torch.Tensor:
    """
    Continue generation from existing token tensor.
    
    Args:
        model: NNsightChatModel instance
        tokens: Existing token tensor to continue from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional generation parameters
        
    Returns:
        Extended token tensor including both input and generated tokens
    """
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "pad_token_id": model.tokenizer.eos_token_id,
    }
    gen_kwargs.update(kwargs)
    
    with model.model.generate(tokens, **gen_kwargs) as generator:
        output = generator.output.save()
    
    return output