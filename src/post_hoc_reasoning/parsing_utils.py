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


def parse_deepseek_response(response: Union[str, List[str]]) -> Tuple[str, str]:
    """
    Parse DeepSeek model response by finding the last occurrence of (A) or (B).
    
    Args:
        response: Raw model response string or list of strings
        
    Returns:
        Tuple of (letter, text_answer) where:
        - letter: 'A' or 'B' if found, empty string otherwise
        - text_answer: 'yes' or 'no' based on the letter and context
    """
    # Handle list input - take the first element
    if isinstance(response, list):
        if len(response) == 0:
            return "", ""
        response = response[0]
    
    # Find all occurrences of (A) or (B) - case insensitive
    matches = list(re.finditer(r'\(\s*([AaBb])\s*\)', response))
    
    if not matches:
        return "", ""
    
    # Get the last match
    last_match = matches[-1]
    letter = last_match.group(1).upper()
    
    # Look for answer text near the matched letter
    # Get broader context to understand the mapping
    start_pos = max(0, last_match.start() - 100)
    end_pos = min(len(response), last_match.end() + 100)
    context = response[start_pos:end_pos]
    
    # Extract the actual choice text associated with the letter
    # Look for patterns like "(A) Yes" or "(A) No" or "(A) Yes, contains"
    text_answer = ""
    
    # Try to find the answer text immediately after the letter
    after_letter = context[last_match.start() - start_pos + len(last_match.group(0)):].strip()
    
    # Check if it starts with Yes or No
    if after_letter.lower().startswith("yes"):
        text_answer = "yes"
    elif after_letter.lower().startswith("no"):
        text_answer = "no"
    else:
        # Look for yes/no in the first few words after the letter
        words = after_letter.split()[:5]  # Check first 5 words
        words_lower = [w.lower() for w in words]
        
        if "yes" in words_lower:
            text_answer = "yes"
        elif "no" in words_lower:
            text_answer = "no"
        else:
            # Last resort: check if the choice mentions containing/not containing
            if "contains" in after_letter.lower() and "not" not in after_letter.lower()[:20]:
                text_answer = "yes"  # Contains anachronistic elements = yes
            elif "does not" in after_letter.lower() or "doesn't" in after_letter.lower():
                text_answer = "no"   # Does not contain = no
            else:
                # Final fallback - this is dataset specific
                # For anachronisms: (A) often = yes, (B) often = no
                # But this is not reliable
                text_answer = ""
    
    return letter, text_answer


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