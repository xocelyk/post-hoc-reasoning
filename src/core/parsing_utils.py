"""
Consolidated parsing utilities for model responses.
"""
import re
from typing import Tuple, Union, List


def filter_think_tags(response: str) -> str:
    """
    Remove content between <think> and </think> tags from response.
    
    Args:
        response: Raw model response that may contain think tags
        
    Returns:
        Response with think content removed
    """
    # Remove content between <think> and </think> tags
    pattern = r'<think>.*?</think>\s*'
    filtered = re.sub(pattern, '', response, flags=re.DOTALL)
    
    # Clean up any double newlines left behind
    filtered = re.sub(r'\n\n+', '\n\n', filtered)
    
    return filtered.strip()


def parse_response(response: Union[str, List[str]], thinking: bool = True) -> Tuple[str, str]:
    """
    Parse model response to extract answer letter and text.
    
    Args:
        response: Raw model response string or list of strings
        thinking: Whether the response includes reasoning (default True)
        
    Returns:
        Tuple of (letter, text_answer) where:
        - letter: The choice letter (A, B, etc.) or empty string if not found
        - text_answer: The text answer or empty string if not found
    """
    # Handle list input - take the first element
    if isinstance(response, list):
        if len(response) == 0:
            return "", ""
        response = response[0]
    
    # Clean response by removing special tokens
    response = (
        response.strip()
        .replace("<eos>", "")
        .replace("<pad>", "")
        .replace("<end_of_turn>", "")
        .strip()
    )
    
    if thinking:
        # For responses with reasoning, look for "the best answer is:" or "the best answer is" patterns
        start_answer_patterns = ["the best answer is:", "the best answer is"]
        response_lower = response.lower()
        
        # Try each pattern and use the one that appears last in the response
        start_idx = -1
        matching_pattern = None
        
        for pattern in start_answer_patterns:
            idx = response_lower.rfind(pattern)
            if idx > start_idx:
                start_idx = idx
                matching_pattern = pattern
        
        if start_idx == -1:
            return "", ""
        
        # Find the answer part using case-insensitive search but preserve original case
        answer_part = response[start_idx + len(matching_pattern):].strip()
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