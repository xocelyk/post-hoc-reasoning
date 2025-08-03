"""
Consolidated parsing utilities for model responses.
"""
import re
from typing import Tuple


def parse_response(response: str, thinking: bool = True) -> Tuple[str, str]:
    """
    Parse model response to extract answer letter and text.
    
    Args:
        response: Raw model response string
        thinking: Whether the response includes reasoning (default True)
        
    Returns:
        Tuple of (letter, text_answer) where:
        - letter: The choice letter (A, B, etc.) or empty string if not found
        - text_answer: The text answer or empty string if not found
    """
    # Clean response by removing special tokens
    response = (
        response.strip()
        .replace("<eos>", "")
        .replace("<pad>", "")
        .replace("<end_of_turn>", "")
        .strip()
    )
    
    if thinking:
        # For responses with reasoning, look for "the best answer is:" pattern
        start_answer_string = "the best answer is:"
        response_lower = response.lower()
        
        if start_answer_string not in response_lower:
            return "", ""
        
        # Find the answer part using case-insensitive search but preserve original case
        start_idx = response_lower.find(start_answer_string)
        answer_part = response[start_idx + len(start_answer_string):]
    else:
        # For non-reasoning responses, use the entire cleaned response
        answer_part = response
    
    # Extract letter using robust regex pattern
    letter_match = re.search(r"\(\s*([A-Za-z])\s*\)", answer_part)
    if not letter_match:
        return "", ""
    
    letter = letter_match.group(1).upper()
    
    # Extract text answer after the parentheses
    match_end = letter_match.end()
    text_after = answer_part[match_end:].strip()
    
    # Clean up text answer
    text_answer = (
        text_after
        .split(".")[0]  # Take first sentence
        .split(",")[0]  # Take first clause
        .strip()
        .lower()
    )
    
    return letter, text_answer


def parse_response_simple(response: str) -> str:
    """
    Simple parsing that returns only the text answer.
    Used by reasoning_probes.py for compatibility.
    
    Args:
        response: Raw model response string
        
    Returns:
        Text answer or empty string if not found
    """
    # Try the new pattern-based approach first (for plausible/implausible)
    match = re.search(r"\(\s*[A-Za-z]\s*\)\s*(plausible|implausible)", response, re.IGNORECASE)
    if match:
        return "yes" if match.group(1).lower() == "plausible" else "no"
    
    # Fall back to standard parsing
    letter, text_answer = parse_response(response, thinking=True)
    return text_answer