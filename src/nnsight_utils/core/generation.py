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
        try:
            tokens = model.to_tokens(prompt)
        except Exception as e:
            print(f"Tokenization error: {e}")
            print(f"Prompt type: {type(prompt)}")
            print(f"Prompt: {prompt[:100]}...")  # First 100 chars
            # If tokenization fails, try direct tokenizer call
            # This handles cases where the model expects different input format
            tokens = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
    else:
        tokens = prompt
    
    # Prepare generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": model.tokenizer.eos_token_id,
    }
    
    # Add DeepSeek-specific stopping criteria
    if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
        # Stop at end of sentence or user turn tokens
        stop_tokens = []
        vocab = model.tokenizer.get_vocab()
        
        # Add DeepSeek specific stopping tokens
        if '<｜end▁of▁sentence｜>' in vocab:
            stop_tokens.append(vocab['<｜end▁of▁sentence｜>'])
        if '<｜User｜>' in vocab:
            stop_tokens.append(vocab['<｜User｜>'])
            
        if stop_tokens:
            gen_kwargs["eos_token_id"] = stop_tokens
    
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
        
    # Update with any additional kwargs
    gen_kwargs.update(kwargs)
    
    # Generate using the model's generate method
    # This calls the NNsightChatModel.generate() which handles nnsight properly
    output = model.generate(tokens, **gen_kwargs)
    
    # Decode the full output (including prompt)
    decoded = model.to_string(output)
    if isinstance(decoded, list):
        full_text = decoded[0]
    else:
        full_text = decoded
    
    # Extract only the generated part (not including the input prompt)
    # This is done at the string level to match TransformerLens behavior
    if full_text.startswith(prompt) and isinstance(prompt, str):
        return full_text[len(prompt):]
    else:
        # If we can't cleanly extract, return the full text
        return full_text


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
    Generate text for multiple prompts with efficient batching and left-padding.
    Works for any batch size, including batch_size=1.
    
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
        
        # Set pad token (use eos_token_id as pad_token_id if not set)
        if model.tokenizer.pad_token_id is None:
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        
        # Use tokenizer's built-in padding with left-padding for causal models
        model.tokenizer.padding_side = 'left'  # Left-pad for causal generation
        
        # Tokenize all prompts with padding
        batch_encoding = model.tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=False
        )
        
        batch_tokens = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": model.tokenizer.pad_token_id,
            "attention_mask": attention_mask,
        }
        
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
        
        # Update with additional kwargs
        gen_kwargs.update(kwargs)
        
        # Generate using the model's batch generate method
        output = model.generate(batch_tokens, **gen_kwargs)
        
        # Decode outputs and extract only generated parts
        input_length = batch_tokens.size(1)
        
        for original_prompt, output_tokens in zip(batch_prompts, output):
            # Extract only the newly generated tokens (after input)
            generated_tokens = output_tokens[input_length:]
            
            if len(generated_tokens) > 0:
                generated_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = ""
            
            # Fallback: string-based extraction if token-based fails
            if not generated_text:
                full_decoded = model.tokenizer.decode(output_tokens, skip_special_tokens=True)
                
                # Remove the original prompt from the decoded output
                if full_decoded.startswith(original_prompt):
                    generated_text = full_decoded[len(original_prompt):]
                else:
                    generated_text = full_decoded
            
            results.append(generated_text)
    
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