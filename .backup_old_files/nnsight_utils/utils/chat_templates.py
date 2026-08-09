"""
Chat template utilities for NNsight models.
"""

from typing import List, Dict, Any, Optional
from ..core.models import NNsightChatModel


def apply_chat_template_batch(
    model: NNsightChatModel,
    conversations: List[List[Dict[str, str]]],
    tokenize: bool = False,
    add_generation_prompt: bool = True
) -> List[str]:
    """
    Apply chat template to multiple conversations.
    
    Args:
        model: NNsightChatModel instance
        conversations: List of conversation message lists
        tokenize: Whether to return tokens instead of strings
        add_generation_prompt: Whether to add generation prompt
        
    Returns:
        List of formatted chat strings or token tensors
    """
    results = []
    
    for messages in conversations:
        formatted = model.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt
        )
        results.append(formatted)
    
    return results


def create_message(role: str, content: str) -> Dict[str, str]:
    """
    Create a properly formatted message dictionary.
    
    Args:
        role: Message role ("user", "assistant", "system")
        content: Message content
        
    Returns:
        Message dictionary
    """
    return {"role": role, "content": content}


def create_conversation(
    user_message: str,
    system_message: Optional[str] = None,
    assistant_history: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Create a conversation from components.
    
    Args:
        user_message: Current user message
        system_message: Optional system prompt
        assistant_history: Optional list of previous assistant messages
        
    Returns:
        List of message dictionaries
    """
    conversation = []
    
    if system_message:
        conversation.append(create_message("system", system_message))
    
    if assistant_history:
        for i, assistant_msg in enumerate(assistant_history):
            # Add alternating user/assistant messages
            if i == 0:
                conversation.append(create_message("user", f"Previous context"))
            conversation.append(create_message("assistant", assistant_msg))
    
    conversation.append(create_message("user", user_message))
    
    return conversation


def is_incomplete_assistant_message(messages: List[Dict[str, str]]) -> bool:
    """
    Check if the conversation ends with an incomplete assistant message.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        True if last message is incomplete assistant message
    """
    if not messages:
        return False
    
    last_message = messages[-1]
    
    # Check if it's an assistant message that doesn't end with a completion token
    if last_message.get("role") == "assistant":
        content = last_message.get("content", "")
        # Simple heuristic: incomplete if doesn't end with punctuation or newline
        return not content.strip().endswith(('.', '!', '?', '\n'))
    
    return False


def extract_assistant_responses(
    model: NNsightChatModel,
    generated_text: str,
    original_messages: List[Dict[str, str]]
) -> str:
    """
    Extract just the assistant's response from generated text.
    
    Args:
        model: NNsightChatModel instance
        generated_text: Full generated text including prompt
        original_messages: Original conversation messages
        
    Returns:
        Just the assistant's response portion
    """
    # Get the original prompt
    original_prompt = model.apply_chat_template(original_messages, tokenize=False)
    
    # Remove the original prompt to get just the response
    if generated_text.startswith(original_prompt):
        response = generated_text[len(original_prompt):]
    else:
        # Fallback: assume response starts after the last user message
        response = generated_text
    
    return response.strip()


def format_conversation_for_display(
    messages: List[Dict[str, str]],
    max_content_length: int = 100
) -> str:
    """
    Format conversation for human-readable display.
    
    Args:
        messages: List of message dictionaries
        max_content_length: Maximum length for content display
        
    Returns:
        Formatted conversation string
    """
    lines = []
    
    for i, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        
        # Truncate long content
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Format with role indicator
        role_indicator = {
            "system": "🔧",
            "user": "👤", 
            "assistant": "🤖"
        }.get(role, "❓")
        
        lines.append(f"{role_indicator} {role.title()}: {content}")
    
    return "\n".join(lines)


def validate_conversation_format(messages: List[Dict[str, str]]) -> bool:
    """
    Validate that conversation follows proper format.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        True if conversation is properly formatted
    """
    if not messages:
        return False
    
    valid_roles = {"system", "user", "assistant"}
    
    for message in messages:
        # Check required fields
        if not isinstance(message, dict):
            return False
        
        if "role" not in message or "content" not in message:
            return False
        
        # Check valid role
        if message["role"] not in valid_roles:
            return False
        
        # Check content is string
        if not isinstance(message["content"], str):
            return False
    
    # Check conversation structure (system can only be first)
    system_indices = [
        i for i, msg in enumerate(messages) 
        if msg["role"] == "system"
    ]
    
    if system_indices and system_indices != [0]:
        return False
    
    return True


def get_conversation_stats(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Get statistics about a conversation.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary with conversation statistics
    """
    if not messages:
        return {"total_messages": 0}
    
    role_counts = {}
    total_chars = 0
    message_lengths = []
    
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        
        role_counts[role] = role_counts.get(role, 0) + 1
        total_chars += len(content)
        message_lengths.append(len(content))
    
    return {
        "total_messages": len(messages),
        "role_counts": role_counts,
        "total_characters": total_chars,
        "average_message_length": total_chars / len(messages),
        "longest_message": max(message_lengths) if message_lengths else 0,
        "shortest_message": min(message_lengths) if message_lengths else 0,
        "has_system_message": any(msg.get("role") == "system" for msg in messages)
    }