"""
Tokenization utilities for NNsight models.
"""

from typing import List, Union, Optional
import torch

from ..core.models import NNsightChatModel


def batch_tokenize(
    model: NNsightChatModel,
    texts: List[str],
    prepend_bos: bool = False,
    max_length: Optional[int] = None,
    padding: str = "longest",
    truncation: bool = True
) -> torch.Tensor:
    """
    Tokenize a batch of texts with consistent formatting.
    
    Args:
        model: NNsightChatModel instance
        texts: List of text strings to tokenize
        prepend_bos: Whether to add BOS token
        max_length: Maximum sequence length
        padding: Padding strategy ("longest", "max_length", or None)
        truncation: Whether to truncate long sequences
        
    Returns:
        Batch tensor of tokenized sequences
    """
    # Use model's tokenizer for batch processing
    tokenizer = model.tokenizer
    
    # Prepare tokenization arguments
    tokenize_kwargs = {
        "padding": padding,
        "truncation": truncation,
        "return_tensors": "pt"
    }
    
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length
    
    # Add BOS token if requested
    if prepend_bos and tokenizer.bos_token is not None:
        texts = [tokenizer.bos_token + text for text in texts]
    
    # Tokenize
    batch_tokens = tokenizer(texts, **tokenize_kwargs)["input_ids"]
    
    # Move to model device
    return batch_tokens.to(model.device)


def get_token_lengths(
    model: NNsightChatModel,
    texts: List[str],
    include_special_tokens: bool = True
) -> List[int]:
    """
    Get token lengths for a list of texts.
    
    Args:
        model: NNsightChatModel instance
        texts: List of text strings
        include_special_tokens: Whether to count special tokens
        
    Returns:
        List of token lengths
    """
    lengths = []
    
    for text in texts:
        tokens = model.tokenizer.encode(
            text, 
            add_special_tokens=include_special_tokens
        )
        lengths.append(len(tokens))
    
    return lengths


def find_token_positions(
    model: NNsightChatModel,
    text: str,
    target_tokens: Union[str, List[str]],
    return_text_positions: bool = False
) -> dict:
    """
    Find positions of specific tokens in text.
    
    Args:
        model: NNsightChatModel instance
        text: Input text string
        target_tokens: Token(s) to find
        return_text_positions: Whether to return character positions too
        
    Returns:
        Dictionary with token positions and optionally text positions
    """
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens]
    
    # Tokenize with offsets
    tokenizer = model.tokenizer
    encoding = tokenizer(text, return_offsets_mapping=True)
    
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    
    results = {}
    
    for target in target_tokens:
        target_id = tokenizer.encode(target, add_special_tokens=False)[0]
        
        positions = []
        text_positions = []
        
        for i, token_id in enumerate(tokens):
            if token_id == target_id:
                positions.append(i)
                if return_text_positions and offsets:
                    text_positions.append(offsets[i])
        
        results[target] = {
            "token_positions": positions,
            "count": len(positions)
        }
        
        if return_text_positions:
            results[target]["text_positions"] = text_positions
    
    return results


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Rough estimate of token count without actual tokenization.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token
        
    Returns:
        Estimated token count
    """
    return max(1, int(len(text) / chars_per_token))


def truncate_to_tokens(
    model: NNsightChatModel,
    text: str,
    max_tokens: int,
    truncate_from: str = "end"
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        model: NNsightChatModel instance
        text: Input text
        max_tokens: Maximum number of tokens
        truncate_from: Where to truncate ("end" or "start")
        
    Returns:
        Truncated text
    """
    tokens = model.tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    if truncate_from == "end":
        truncated_tokens = tokens[:max_tokens]
    elif truncate_from == "start":
        truncated_tokens = tokens[-max_tokens:]
    else:
        raise ValueError(f"Invalid truncate_from: {truncate_from}")
    
    return model.tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def pad_tokens(
    tokens: torch.Tensor,
    target_length: int,
    pad_token_id: int,
    pad_side: str = "left"
) -> torch.Tensor:
    """
    Pad token sequences to target length.
    
    Args:
        tokens: Token tensor
        target_length: Target sequence length
        pad_token_id: Token ID to use for padding
        pad_side: Side to pad ("left" or "right")
        
    Returns:
        Padded token tensor
    """
    current_length = tokens.shape[-1]
    
    if current_length >= target_length:
        return tokens
    
    padding_length = target_length - current_length
    
    if pad_side == "left":
        padding = torch.full(
            (*tokens.shape[:-1], padding_length),
            pad_token_id,
            dtype=tokens.dtype,
            device=tokens.device
        )
        return torch.cat([padding, tokens], dim=-1)
    elif pad_side == "right":
        padding = torch.full(
            (*tokens.shape[:-1], padding_length),
            pad_token_id,
            dtype=tokens.dtype,
            device=tokens.device
        )
        return torch.cat([tokens, padding], dim=-1)
    else:
        raise ValueError(f"Invalid pad_side: {pad_side}")


def get_tokenizer_info(model: NNsightChatModel) -> dict:
    """
    Get information about the model's tokenizer.
    
    Args:
        model: NNsightChatModel instance
        
    Returns:
        Dictionary with tokenizer information
    """
    tokenizer = model.tokenizer
    
    info = {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "has_bos_token": tokenizer.bos_token is not None,
        "has_eos_token": tokenizer.eos_token is not None,
        "has_pad_token": tokenizer.pad_token is not None,
    }
    
    if tokenizer.bos_token is not None:
        info["bos_token"] = tokenizer.bos_token
        info["bos_token_id"] = tokenizer.bos_token_id
    
    if tokenizer.eos_token is not None:
        info["eos_token"] = tokenizer.eos_token
        info["eos_token_id"] = tokenizer.eos_token_id
    
    if tokenizer.pad_token is not None:
        info["pad_token"] = tokenizer.pad_token
        info["pad_token_id"] = tokenizer.pad_token_id
    
    return info