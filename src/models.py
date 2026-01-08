import os
import platform
import logging
import traceback
from typing import Dict, List, Union, Optional
import numpy as np
import gc
import torch
from nnsight import LanguageModel
from transformer_lens import HookedTransformer, utils
from utils import _steer_generated_token, _ace_debias_generated_token
from functools import partial
from copy import deepcopy

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

class ChatModel(ABC):
    """
    Abstract base class for chat models.
    Defines the interface for both NNsightChatModel and TransformerLens-based models.
    """

    @abstractmethod
    def to_tokens(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Tokenize text input."""
        pass

    @abstractmethod
    def to_string(self, tokens: torch.Tensor, **kwargs) -> Union[str, List[str]]:
        """Decode tokens back to text."""
        pass

    @abstractmethod
    def generate(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate text using the model."""
        pass

    @property
    @abstractmethod
    def tokenizer(self):
        """Access to the underlying tokenizer."""
        pass

    @property
    @abstractmethod
    def config(self):
        """Access to the underlying model config."""
        pass

    @abstractmethod
    def batch_get_resid_activations(
        self,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        position: str = "last",
        batch_size: int = 4
    ) -> np.ndarray:
        """Extract residual stream activations for a batch of prompts."""
        pass


class NNSightChatModel(ChatModel):
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
        # Handle both single strings and lists
        if isinstance(text, str):
            text = [text]
        
        # Default tokenizer arguments for compatibility
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            **kwargs
        }
        
        result = self.model.tokenizer(text, **tokenizer_kwargs)
        return result["input_ids"]
    
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
            # Single sequence
            return self.model.tokenizer.decode(tokens, **kwargs)
        else:
            # Batch of sequences
            return [
                self.model.tokenizer.decode(seq, **kwargs) 
                for seq in tokens
            ]
    
    def generate(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            tokens: Input token tensor
            **kwargs: Generation arguments
            
        Returns:
            Generated token tensor
        """
        # For basic generation, we need to use nnsight's context manager properly
        # The model.generate() returns a tracer, not actual output
        
        # Ensure we have pad_token_id set
        if 'pad_token_id' not in kwargs and hasattr(self.tokenizer, 'eos_token_id'):
            kwargs['pad_token_id'] = self.tokenizer.eos_token_id
        
        # Use nnsight's generate context manager like in the steering code
        # We need to save some output to force materialization
        with self.model.generate(tokens, **kwargs) as generator:
            # Get the generated output from the model's generator
            output = self.model.generator.output.save()
        
        # The output contains the full sequence (input + generated)
        # Return it as-is since generate_text expects the full sequence
        return output
    
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

    def _get_layer_output(self, layer_idx: int) -> any:
        """
        Get the layer output from the model.
        
        Args:
            model: The underlying model object (model.model)
            layer_idx: Index of the layer to get output from
            
        Returns:
            The layer output tensor (handles tuple outputs automatically)
            
        Raises:
            ValueError: If the model architecture is not supported
        """
        # 1. Find the list of layers object depending on the arch
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-style models (GPT2, etc.)
            layer_container = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama/Gemma style models
            layer_container = self.model.model.layers
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")
        
        # 2. Select the layer at layer_idx and get the .output
        layer_output = layer_container[layer_idx].output
        
        # 3. Check if .output is a tuple of len 1 and if so select the first item
        if isinstance(layer_output, tuple) and len(layer_output) == 1:
            return layer_output[0]
        else:
            return layer_output

    def batch_get_resid_activations(
        self, 
        prompts: List[str],
        layers: Optional[List[int]] = None,
        position: str = "last",
        batch_size: int = 4
    ) -> np.ndarray:
        """
        Extract residual stream activations for a batch of prompts using nnsight.
        
        Args:
            model: NNsightChatModel instance
            prompts: List of prompt strings
            layers: List of layer indices to extract (None for all layers)
            position: Position to extract ("last" for final token, "all" for all positions)
            batch_size: Number of prompts to process at once (default: 4)
            
        Returns:
            Numpy array of shape (n_prompts, n_layers, d_model) containing activations
        """
        # Determine layers to extract
        if layers is None:
            layers = list(range(self.cfg.n_layers))
        
        n_prompts = len(prompts)
        n_layers = len(layers)
        d_model = self.cfg.d_model
        
        # Pre-allocate output array
        if position == "last":
            all_activations = np.zeros((n_prompts, n_layers, d_model))
        else:
            # We'll collect these and stack at the end for "all" positions
            all_activations = []
        
        # Process in batches with progress bar
        from tqdm import tqdm
        
        progress_bar = tqdm(
            total=n_prompts,
            desc=f"Extracting activations (batch_size={batch_size})",
            unit="prompts",
            ncols=100
        )
        
        for i in range(0, n_prompts, batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_tokens = self.to_tokens(batch_prompts)
            
            # Add max length truncation to prevent very long sequences
            max_length = 1024  # Reasonable max length
            # curr_length = batch_tokens.shape[1]
            # to_pad = max(0, max_length - curr_length)
            # batch_tokens = torch.cat([torch.ones(batch_tokens.shape[0], to_pad, dtype=batch_tokens.dtype) * self.tokenizer.pad_token_id, batch_tokens[:, -max_length:]], dim=1)
            batch_tokens = batch_tokens[:, -max_length:]
            
            # Extract activations using nnsight tracing
            with self.model.trace(batch_tokens):
                # Extract residual activations from specified layers
                if position == "last":
                    # Extract only the final position (most common case)
                    residuals = {
                        layer: self._get_layer_output(layer)[:, -1]
                        for layer in layers
                    }.save()
                elif position == "all":
                    # Extract all positions
                    residuals = {
                        layer: self._get_layer_output(layer)
                        for layer in layers
                    }.save()
                else:
                    raise ValueError(f"Unknown position: {position}. Use 'last' or 'all'")
        
            # Process batch results
            batch_size_actual = len(batch_prompts)
            
            if position == "last":
                # Store activations for this batch
                for j, layer in enumerate(layers):
                    layer_acts = residuals[layer].detach().float().cpu().numpy()
                    all_activations[i:i + batch_size_actual, j, :] = layer_acts
            else:  # position == "all"
                # Collect batch activations
                batch_acts = np.zeros((batch_size_actual, batch_tokens.shape[1], n_layers, d_model))
                for j, layer in enumerate(layers):
                    layer_acts = residuals[layer].detach().float().cpu().numpy()
                    batch_acts[:, :, j, :] = layer_acts
                all_activations.append(batch_acts)
            
            # Clean up GPU memory after each batch
            del residuals
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            # Update progress bar
            progress_bar.update(batch_size_actual)
        
        # Close progress bar
        progress_bar.close()
        
        # Final processing
        if position == "all":
            # Stack all batch results
            all_activations = np.concatenate(all_activations, axis=0)
        
        return all_activations

    @torch.no_grad() 
    def batch_get_generations(self, prompts, temperature, max_new_tokens):
        if hasattr(self.model, 'model_name') and self.model.model_name.lower().startswith('deepseek'):
            max_new_tokens = 2000
        elif hasattr(self.model, 'model_name') and self.model.model_name.lower().startswith('openai'):
            max_new_tokens = 128
        generations = []
        batch_encoding = self.model.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=False
        )
        batch_tokens = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "pad_token_id": self.model.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        with self.model.generate(batch_tokens, **gen_kwargs) as generator:
            output = self.model.generator.output.save()
        generations = [self.model.tokenizer.decode(o) for o in output]
        return generations

    @torch.no_grad()
    def generate_with_steering(self, prompts, temperature, max_new_tokens, alpha, steering_vectors, layers):
        steering_tensors = {}
        for layer in layers:
            steering_tensors[layer] = torch.as_tensor(
                steering_vectors[layer],
                dtype=torch.float32,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )

        # Set pad token (use eos_token_id as pad_token_id if not set)
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id
        
        # Use tokenizer's built-in padding with left-padding for causal models
        self.model.tokenizer.padding_side = 'left'
        
        # Tokenize all prompts with padding
        batch_encoding = self.model.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=False
        )
        
        batch_tokens = batch_encoding['input_ids']
        attention_mask = batch_encoding['attention_mask']

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "pad_token_id": self.model.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }

        # Add DeepSeek-specific stopping criteria
        if hasattr(self.model, 'model_name') and self.model.model_name.lower().startswith('deepseek'):
            stop_tokens = []
            vocab = self.model.tokenizer.get_vocab()
            
            if '<｜end▁of▁sentence｜>' in vocab:
                stop_tokens.append(vocab['<｜end▁of▁sentence｜>'])
            if '<｜User｜>' in vocab:
                stop_tokens.append(vocab['<｜User｜>'])
                
            if stop_tokens:
                gen_kwargs["eos_token_id"] = stop_tokens

        with self.model.generate(batch_tokens, **gen_kwargs) as generator:
            # output = self.model.generator.output.save()
            residual = self.model.model.layers[layers[0]].output
            steering_vector = steering_tensors[layers[0]].to(residual.device)
            residual += alpha * steering_vector
            output = self.model.generator.output.save()
        
        # Return only the freshly generated part
        gen_only = output[:, batch_tokens.size(1):]
        generations = [self.model.tokenizer.decode(o) for o in gen_only]
        del batch_tokens, output
        gc.collect()
        return generations

    def generate_with_ace_debiasing(self, 
        prompts: List[str],
        ace_unit_direction: np.ndarray,
        ace_bias: float,
        layer: int,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text with ACE debiasing intervention using nnsight.
        
        Args:
            model: NNsightChatModel instance
            tokens: Input tokens
            ace_unit_direction: ACE unit direction vector
            ace_bias: ACE bias value for centering
            layer: Layer index to apply debiasing
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text with ACE debiasing applied
        """
        # Convert prompts to tokens
        prompt_tokens = self.to_tokens(prompts)
        batch_size = prompt_tokens.shape[0]
        prompt_len = prompt_tokens.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.model.tokenizer.eos_token_id,
        }
        # Convert to torch tensor if needed
        if isinstance(ace_unit_direction, np.ndarray):
            ace_unit_direction = torch.tensor(
                ace_unit_direction, 
                dtype=torch.float32,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        
        # Define ACE intervention function
        def ace_debias_activation(activation_tensor):
            """Apply ACE debiasing: x_debiased = x - (<x, unit_direction> - bias) * unit_direction"""
            # activation_tensor shape: [batch_size, seq_len, d_model]
            
            # Compute projection: <x, unit_direction>
            projection = torch.sum(activation_tensor * ace_unit_direction, dim=-1, keepdim=True)  # [B, seq_len, 1]
            
            # Compute intervention: (projection - bias) * unit_direction
            intervention = (projection - ace_bias) * ace_unit_direction  # [B, seq_len, d_model]
            
            # Apply intervention
            return intervention
        
        # Apply debiasing during generation
        with self.model.generate(
            prompt_tokens,
            **gen_kwargs
        ) as generator:
            # Apply ACE intervention to the specified layer
            residual = self._get_layer_output(layer)
            intervention = ace_debias_activation(residual)
            residual -= intervention
            
            # Save the output
            output = self.model.generator.output.save()
        
        # Decode the generated text
        assert output.shape[0] == batch_size
        generated_text = [self.model.tokenizer.decode(o[prompt_len:]) for o in output]
        
        return generated_text
            


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

class TransformerLensChatModel(ChatModel):
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
        # For Llama-2 models, preemptively clean messages to avoid template errors
        is_llama2 = "llama-2" in self.model_name.lower() or "llama2" in self.model_name.lower()
        if is_llama2:
            # Clean any [INST]/[/INST] tags from message content before applying template
            cleaned_messages = []
            for msg in messages:
                cleaned_msg = dict(msg)
                if 'content' in cleaned_msg:
                    cleaned_msg['content'] = cleaned_msg['content'].replace('[INST]', '').replace('[/INST]', '')
                cleaned_messages.append(cleaned_msg)
            messages = cleaned_messages
        
        try:
            # Check if we need to continue the final message
            if self._is_incomplete_assistant_message(messages):
                # For Llama-2 models, don't use continue_final_message as it often causes issues
                if is_llama2:
                    # Add generation prompt manually for Llama-2
                    result = self.model.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    # Llama-2 format expects responses after [/INST]
                    if not result.endswith(" "):
                        result += " "
                else:
                    # Try with continue_final_message first for other models
                    try:
                        result = self.model.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False,
                            continue_final_message=True
                        )
                    except TypeError as e:
                        # If continue_final_message is not supported, fall back to default
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

    def batch_get_resid_activations(self, prompts: List[str]):
        """Get residual stream activations for a batch of prompts with memory optimization."""
        layers = list(range(self.cfg.n_layers))
        
        with torch.no_grad():  # Ensure no gradients are computed
            tokens = self.to_tokens(prompts, prepend_bos=True)
            _, cache = self.model.run_with_cache(tokens, pos_slice=-1)

            # Pre-allocate with float32 to save memory
            activations = np.zeros((len(prompts), self.cfg.n_layers, self.cfg.d_model), dtype=np.float32)

            for layer in layers:
                layer_activations = cache["resid_post", layer]
                # Convert to float32 before converting to numpy to avoid BFloat16 issues on MPS
                layer_activations = layer_activations.squeeze().detach().float().cpu().numpy().astype(np.float32)
                activations[:, layer, :] = layer_activations
                
                # Immediate cleanup
                del layer_activations
                
                # More aggressive memory cleanup
                if layer % 5 == 0:  # Every 5 layers
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
            
            # Final cleanup
            del cache, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        return activations

    @torch.inference_mode()
    def batch_get_generations(self, prompts, temperature, max_new_tokens):
        token_outs = self.model.tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
        prompt_tokens = token_outs.input_ids.to(self.model.W_E.device)
        full_tokens = self.model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            prepend_bos=False,
        )
        gen_only = full_tokens[:, prompt_tokens.size(1):]
        generations = self.model.to_string(gen_only)
        return generations

    @torch.inference_mode()
    def generate_with_steering(self, prompts, temperature, max_new_tokens, alpha, steering_vectors, layers):
        token_outs = self.model.tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
        prompt_tokens = token_outs.input_ids.to(self.model.W_E.device)

        steer_hook = partial(_steer_generated_token,
                            steering_vectors=steering_vectors,
                            alpha=alpha)

        # 3. Register hooks once, generate, then clear hooks
        for l in layers:
            name = utils.get_act_name("resid_post", l)
            self.model.add_hook(name, steer_hook, dir="fwd")   # returns None on PyPI build

        # run generate (steering active)
        full_tokens = self.model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            prepend_bos=False,
        )

        self.model.reset_hooks()            # ← one call clears every registered hook

        # Return only the freshly generated part
        gen_only = full_tokens[:, prompt_tokens.size(1):]
        generations = self.model.to_string(gen_only)
        del prompt_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return generations

    @torch.inference_mode()
    def generate_with_ace_debiasing(
        self,
        prompts,
        ace_unit_direction,                   # NumPy or torch, [d_model]
        ace_bias: float,                      # bias value for centering
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        layer: int = None,                    # specific layer to apply debiasing
    ):
        """Generate text with ACE debiasing intervention."""
        token_outs = self.model.tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
        prompt_tokens = token_outs.input_ids.to(self.model.W_E.device)
        # 1. Normalise ace_unit_direction to correct dtype / device
        ace_unit_direction = torch.as_tensor(
            ace_unit_direction,
            dtype=self.model.W_E.dtype,
            device=self.model.W_E.device,
        )

        # 2. Decide which layer to debias (should be specified)
        if layer is None:
            raise ValueError("Layer must be specified for ACE debiasing")

        debias_hook = partial(_ace_debias_generated_token,
                            ace_unit_direction=ace_unit_direction,
                            ace_bias=ace_bias)

        # 3. Register hook for the specific layer
        name = utils.get_act_name("resid_post", layer)
        self.model.add_hook(name, debias_hook, dir="fwd")

        # run generate (debiasing active)
        full_tokens = self.model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            prepend_bos=False,
        )

        self.model.reset_hooks()            # ← clear registered hooks

        # Return only the freshly generated part
        gen_only = full_tokens[:, prompt_tokens.size(1):]
        generations = self.model.to_string(gen_only)
        del prompt_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return generations