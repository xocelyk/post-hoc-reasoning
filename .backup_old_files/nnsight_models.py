"""
NNsight-based model wrapper for broader model compatibility.

This module provides a ChatModel-compatible interface using nnsight's LanguageModel,
enabling support for any HuggingFace model including DeepSeek and other models
not supported by transformer_lens.
"""

import platform
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
from nnsight import LanguageModel


class NNsightChatModel:
    """
    NNsight-based chat model wrapper with broad HuggingFace compatibility.
    
    This class provides the same interface as the transformer_lens ChatModel
    but uses nnsight's LanguageModel for broader model support.
    """
    
    def __init__(
        self,
        model_name: str,
        device_map: Union[str, Dict] = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        **kwargs
    ):
        """
        Initialize the NNsightChatModel.
        
        Args:
            model_name: Name of the HuggingFace model to load
            device_map: Device mapping strategy ("auto", "cpu", "cuda", or custom dict)
            dtype: Data type for model weights ("bfloat16", "float16", "float32")
            trust_remote_code: Whether to trust remote code in model files
            **kwargs: Additional arguments passed to LanguageModel
        """
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = self._resolve_dtype(dtype)
        
        # Load model using nnsight
        self.model = LanguageModel(
            model_name,
            device_map=device_map,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        # Dispatch model to load actual weights (nnsight loads with meta tensors by default)
        if hasattr(self.model, 'dispatch'):
            self.model.dispatch()
        
        # Model-specific formatting registry
        self.format_registry = {
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": self._format_turns_deepseek,
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": self._format_turns_deepseek,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": self._format_turns_deepseek,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": self._format_turns_deepseek,
            "google/gemma-2-9b-it": self._format_turns_gemma,
            "google/gemma-3-12b-it": self._format_turns_gemma,
        }
        
        # Store config-like attributes for compatibility
        self.cfg = self._create_config_compatible_object()
    
    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(dtype_map.keys())}")
        
        return dtype_map[dtype]
    
    def _create_config_compatible_object(self):
        """Create a config-like object for compatibility with existing code."""
        class ConfigCompatible:
            def __init__(self, model):
                self.n_layers = model.config.num_hidden_layers or model.config.n_layers
                self.d_model = model.config.hidden_size
                
        return ConfigCompatible(self.model)
    
    def _format_turns_deepseek(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format chat turns for DeepSeek models (convert 'model' role to 'assistant')."""
        formatted = []
        for msg in deepcopy(messages):
            if msg["role"] == "model":
                msg["role"] = "assistant"
            formatted.append(msg)
        return formatted
    
    def _format_turns_gemma(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format chat turns for Gemma models (currently no special formatting needed)."""
        return deepcopy(messages)
    
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

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Format a list of chat messages according to the model's chat template.
        Automatically detects incomplete assistant messages and uses continue_final_message.
        
        Args:
            messages: List of chat messages with 'role' and 'content' keys
            **kwargs: Additional arguments passed to tokenizer.apply_chat_template
            
        Returns:
            Formatted chat string
        """
        # Apply model-specific formatting first
        if self.model_name in self.format_registry:
            messages = self.format_registry[self.model_name](messages)
        
        # Use tokenizer's chat template
        try:
            # Check if we need to continue the final message
            if self._is_incomplete_assistant_message(messages) and 'continue_final_message' not in kwargs:
                kwargs['continue_final_message'] = True
                
            result = self.model.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                **kwargs
            )
            return result
        except Exception as e:
            raise
    
    def to_tokens(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """
        Tokenize text input.
        
        Args:
            text: Input text string or list of strings
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenized input as tensor
        """
        # Ensure text is a list for consistent handling
        if isinstance(text, str):
            text = [text]
        
        # Default tokenizer arguments for compatibility
        tokenizer_kwargs = {
            "return_tensors": "pt",
            **kwargs
        }
        
        # Only add padding if we have multiple strings
        if len(text) > 1:
            tokenizer_kwargs["padding"] = True
        
        result = self.model.tokenizer(text, **tokenizer_kwargs)
        tokens = result["input_ids"]
        
        # Ensure tokens are on a real device (not meta)
        # NNsight models are often loaded with meta tensors
        if tokens.is_meta or str(tokens.device) == 'meta':
            if torch.cuda.is_available():
                tokens = tokens.to('cuda')
            else:
                tokens = tokens.to('cpu')
        
        return tokens
    
    def to_string(self, tokens: torch.Tensor, **kwargs) -> Union[str, List[str]]:
        """
        Decode tokens back to text.
        
        Args:
            tokens: Token tensor to decode
            **kwargs: Additional arguments passed to tokenizer.decode
            
        Returns:
            Decoded text string or list of strings
        """
        # Handle batch dimension
        if tokens.dim() == 1:
            # Single sequence - ensure it's a proper tensor
            return self.model.tokenizer.decode(tokens.tolist(), **kwargs)
        else:
            # Batch of sequences
            return [
                self.model.tokenizer.decode(seq.tolist(), **kwargs) 
                for seq in tokens
            ]
    
    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate text using the model with custom sampling loop.
        
        Args:
            tokens: Input token tensor
            **kwargs: Generation arguments (max_new_tokens, temperature, do_sample, etc.)
            
        Returns:
            Generated token tensor
        """
        # Extract generation parameters
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        temperature = kwargs.get('temperature', 0.7)
        do_sample = kwargs.get('do_sample', True)
        eos_token_id = kwargs.get('pad_token_id', self.tokenizer.eos_token_id)
        
        # Clone tokens to avoid modifying input and ensure correct device
        toks = tokens.clone()
        
        # Ensure tokens is 2D tensor (batch_size, seq_len)
        if toks.dim() == 1:
            toks = toks.unsqueeze(0)
        
        # Determine the device from model parameters
        try:
            model_device = next(self.model.parameters()).device
        except:
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move tokens to the same device as the model
        toks = toks.to(model_device)
        
        # Custom generation loop using model.trace()
        for _ in range(max_new_tokens):
            with self.model.trace(toks):
                # Get logits from the language model head
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head.output.save()
                elif hasattr(self.model, 'embed_out'):
                    logits = self.model.embed_out.output.save()
                else:
                    # Fallback for other architectures
                    raise ValueError(f"Cannot determine output layer for model: {type(self.model)}")
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :]
            
            if do_sample and temperature > 0:
                # Apply temperature and sample
                probs = (next_token_logits / temperature).softmax(dim=-1)
                next_tok = torch.multinomial(probs, 1)
            else:
                # Greedy decoding
                next_tok = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Ensure next_tok is on the same device as toks
            next_tok = next_tok.to(toks.device)
            
            # Append to sequence
            toks = torch.cat([toks, next_tok], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
            
            # Clean up to save memory
            del logits, next_token_logits
            if do_sample and temperature > 0:
                del probs
            torch.cuda.empty_cache()
        
        return toks
    
    @property
    def tokenizer(self):
        """Access to the underlying tokenizer."""
        return self.model.tokenizer
    
    @property
    def config(self):
        """Access to the underlying model config."""
        return self.model.config
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying nnsight model."""
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def is_model_supported_by_nnsight(model_name: str) -> bool:
    """
    Check if a model is likely to be supported by nnsight.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        True if the model should work with nnsight
    """
    # nnsight generally supports any HuggingFace model
    # This function can be extended with specific compatibility checks
    
    # Known supported model patterns
    supported_patterns = [
        "google/gemma",
        "deepseek-ai/",
        "microsoft/",
        "meta-llama/",
        "mistralai/",
        "anthropic/",
        "openai-community/gpt2",
    ]
    
    return any(pattern in model_name.lower() for pattern in supported_patterns)


def get_model_info(model_name: str) -> Dict[str, str]:
    """
    Get information about model compatibility and requirements.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Dictionary with model information
    """
    info = {
        "name": model_name,
        "backend": "nnsight",
        "supports_chat_template": True,
        "requires_trust_remote_code": False,
    }
    
    # Model-specific requirements
    if "deepseek" in model_name.lower():
        info["chat_formatting"] = "deepseek"
        info["requires_trust_remote_code"] = True
    elif "gemma" in model_name.lower():
        info["chat_formatting"] = "gemma"
    else:
        info["chat_formatting"] = "default"
    
    return info