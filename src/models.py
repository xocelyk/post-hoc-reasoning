import os
import platform
from typing import Dict, List

import torch
from transformer_lens import HookedTransformer


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

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a list of chat messages according to the model's chat template.
        """
        try:
            result = self.model.tokenizer.apply_chat_template(messages, tokenize=False)
            return result
        except Exception as e:
            raise

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying transformer lens model.
        return getattr(self.model, attr)
