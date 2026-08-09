"""
Utility functions for NNsight operations.
"""

from .chat_templates import (
    apply_chat_template_batch,
    create_conversation,
    create_message,
    extract_assistant_responses,
    format_conversation_for_display,
    get_conversation_stats,
    is_incomplete_assistant_message,
    validate_conversation_format,
)
from .tokenization import (
    batch_tokenize,
    estimate_tokens,
    find_token_positions,
    get_token_lengths,
    get_tokenizer_info,
    pad_tokens,
    truncate_to_tokens,
)

__all__ = [
    # Chat templates
    "apply_chat_template_batch",
    "create_message",
    "create_conversation",
    "is_incomplete_assistant_message",
    "extract_assistant_responses",
    "format_conversation_for_display",
    "validate_conversation_format",
    "get_conversation_stats",
    # Tokenization
    "batch_tokenize",
    "get_token_lengths",
    "find_token_positions",
    "estimate_tokens",
    "truncate_to_tokens",
    "pad_tokens",
    "get_tokenizer_info",
]