import os
import platform
import logging
from typing import Dict, List

import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


class ChatModel:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        n_devices: int = 1,
        dtype: str = "bfloat16",
    ):
        """
        Initialize the ChatModel.

        Args:
            model_name: Name of the model to load via transformer_lens.
            device: Device to run the model on.
            dtype: Data type for model weights.
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=self.device,
            dtype=self.dtype,
            n_devices=n_devices,
        )
        
        # Validate that the model has a chat template
        if not hasattr(self.model.tokenizer, 'chat_template') or self.model.tokenizer.chat_template is None:
            raise ValueError(
                f"Model '{model_name}' does not have a chat template. "
                f"Chat templates are required for proper conversation formatting. "
                f"Please use a model that supports chat templates or implement custom formatting."
            )

    def _resolve_device(self, device: str) -> str:
        """Resolve device setting to actual device."""
        if device != "auto":
            return device
        
        # Auto-detect best available device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and platform.processor() == "arm":
            # M1/M2 Mac with MPS support
            return "mps"
        else:
            return "cpu"
    
    def _resolve_dtype(self, dtype: str):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,  # For 8-bit quantization
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(dtype_map.keys())}")
        
        return dtype_map[dtype]

    def _is_incomplete_assistant_message(self, messages: List[Dict[str, str]]) -> bool:
        """
        Check if the final message is an incomplete assistant response that should continue.
        
        Args:
            messages: List of chat messages
            
        Returns:
            True if final message is an incomplete assistant response
        """
        if not messages:
            return False
            
        final_message = messages[-1]
        if final_message.get("role") != "assistant":
            return False
            
        content = final_message.get("content", "").strip()
        # Check for common continuation patterns
        continuation_patterns = [
            "Let's think step by step:",
            "A: Let's think step by step:",
            "Let me think about this:",
            "A:",
        ]
        
        return any(content.endswith(pattern) for pattern in continuation_patterns)

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a list of chat messages according to the model's chat template.
        Automatically detects incomplete assistant messages and uses continue_final_message.
        """
        try:
            # Check if we need to continue the final message
            if self._is_incomplete_assistant_message(messages):
                # Try with continue_final_message first
                try:
                    result = self.model.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        continue_final_message=True
                    )
                except TypeError as e:
                    # If continue_final_message is not supported, fall back to default
                    # This handles older tokenizers that don't support this parameter
                    result = self.model.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                result = self.model.tokenizer.apply_chat_template(messages, tokenize=False)
            return result
        except Exception as e:
            # If there's an error with the chat template, provide more context
            if "[/INST]" in str(e) and "doesn't match" in str(e):
                # Log the messages to help debug
                logger.error(f"Chat template error for Llama model. Messages: {messages}")
                logger.error(f"Model name: {self.model_name}")
                logger.error(f"Error: {str(e)}")
                
                # Try a workaround for Llama models
                try:
                    # For Llama, ensure messages don't have malformed content
                    cleaned_messages = []
                    for msg in messages:
                        cleaned_msg = dict(msg)
                        # Remove any [INST] or [/INST] tags from content
                        if 'content' in cleaned_msg:
                            cleaned_msg['content'] = cleaned_msg['content'].replace('[INST]', '').replace('[/INST]', '')
                        cleaned_messages.append(cleaned_msg)
                    
                    result = self.model.tokenizer.apply_chat_template(cleaned_messages, tokenize=False)
                    logger.warning("Applied Llama chat template workaround - removed [INST]/[/INST] tags from message content")
                    return result
                except Exception as e2:
                    logger.error(f"Llama workaround also failed: {str(e2)}")
                    raise ValueError(
                        f"Chat template error for model '{self.model_name}': {str(e)}. "
                        f"This often happens with Llama models when the chat template is malformed. "
                        f"Messages passed: {messages}"
                    )
            raise

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying transformer lens model.
        return getattr(self.model, attr)
