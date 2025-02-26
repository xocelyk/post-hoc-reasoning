from typing import Dict, List, Optional, Tuple, Union
import torch
from transformer_lens import HookedTransformer

class ChatModel:
    """
    Enhanced ChatModel class that supports different model architectures.
    """
    # Mapping of model families to their known model names
    MODEL_FAMILIES = {
        "gemma": ["google/gemma-2-9b-it", "google/gemma-2-2b-it", "google/gemma-7b-it"],
        "llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"],
        "mistral": ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "phi": ["microsoft/phi-2", "microsoft/phi-1_5"],
        "falcon": ["tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b-instruct"],
    }

    def __init__(self, model_name: str, device: str = "cpu", dtype: str = "bfloat16"):
        """
        Initialize the ChatModel.

        Args:
            model_name: Name of the model to load via transformer_lens.
            device: Device to run the model on.
            dtype: Data type for model weights.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model_family = self._detect_model_family()
        
        # Convert dtype string to torch dtype
        if isinstance(dtype, str):
            if dtype == "float16":
                dtype = torch.float16
            elif dtype == "bfloat16":
                dtype = torch.bfloat16
            elif dtype == "float32":
                dtype = torch.float32
        
        print(f"Loading model: {model_name} (family: {self.model_family})")
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name, device=self.device, dtype=dtype
        )
        
    def _detect_model_family(self) -> str:
        """Detect the model family based on the model name."""
        for family, models in self.MODEL_FAMILIES.items():
            if any(model in self.model_name for model in models):
                return family
            
        # If model family not found in our mappings, try to guess from the name
        model_name_lower = self.model_name.lower()
        for family in self.MODEL_FAMILIES.keys():
            if family in model_name_lower:
                return family
                
        # Default fallback
        return "unknown"
    
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a list of chat messages according to the model's chat template.
        This handles different models with their specific templates.
        """
        try:
            # Use the model's built-in chat template if available
            return self.model.tokenizer.apply_chat_template(messages, tokenize=False)
        except (AttributeError, NotImplementedError):
            # Fallback to custom template based on model family
            return self._custom_chat_template(messages)
    
    def _custom_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Custom chat template implementation for models without built-in support."""
        result = ""
        
        if self.model_family == "gemma":
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    result += f"{content}\n\n"
                elif role == "user":
                    result += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif role == "assistant" or role == "model":
                    result += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            
            # Add final turn for model to complete
            if messages and messages[-1]["role"] == "user":
                result += "<start_of_turn>model\n"
        
        elif self.model_family == "llama":
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            if system_msg:
                result += f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
            else:
                result += "<s>[INST] "
            
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    continue
                
                if msg["role"] == "user":
                    if i > 0 and messages[i-1]["role"] != "system":
                        result += "[/INST] "
                        result += f"{messages[i-1]['content']} "
                        result += "[INST] "
                    result += f"{msg['content']} "
                elif msg["role"] == "assistant" or msg["role"] == "model":
                    if i == len(messages) - 1:
                        result += "[/INST] "
                    result += f"{msg['content']} "
        
        elif self.model_family == "mistral":
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    result += f"<s>[INST] {msg['content']} [/INST]"
                elif msg["role"] == "user":
                    result += f"<s>[INST] {msg['content']} [/INST]"
                elif msg["role"] == "assistant" or msg["role"] == "model":
                    result += f" {msg['content']} </s>"
        
        else:
            # Generic template as fallback
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                result += f"{role}: {content}\n\n"
            
            result += "Assistant: "
        
        return result

    def parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the model's response based on the model family.
        Returns the extracted letter and text answer.
        """
        if self.model_family == "gemma":
            return self._parse_gemma_response(response)
        elif self.model_family == "llama":
            return self._parse_llama_response(response)
        elif self.model_family == "mistral":
            return self._parse_mistral_response(response)
        else:
            return self._parse_generic_response(response)
    
    def _parse_gemma_response(self, response: str) -> Tuple[str, str]:
        """Parse Gemma-specific response format."""
        response = (
            response.strip()
            .replace("<eos>", "")
            .replace("<pad>", "")
            .replace("<end_of_turn>", "")
            .strip()
        )
        start_answer_string = "the best answer is:"
        if start_answer_string not in response.lower():
            return "", ""
        answer_part = response.lower().split(start_answer_string)[-1]
        import re
        letter_match = re.search(r"\((.)\)", answer_part)
        if not letter_match:
            return "", ""
        letter = letter_match.group(1)
        text_answer = (
            answer_part.split(")")[-1]
            .strip()
            .split(", ")[0]
            .lower()
            .replace(".", "")
            .strip()
        )
        return letter, text_answer
    
    def _parse_llama_response(self, response: str) -> Tuple[str, str]:
        """Parse Llama-specific response format."""
        response = response.strip().replace("</s>", "").strip()
        start_answer_string = "the best answer is:"
        if start_answer_string not in response.lower():
            return "", ""
        answer_part = response.lower().split(start_answer_string)[-1]
        import re
        letter_match = re.search(r"\((.)\)", answer_part)
        if not letter_match:
            return "", ""
        letter = letter_match.group(1)
        text_answer = (
            answer_part.split(")")[-1]
            .strip()
            .split(", ")[0]
            .lower()
            .replace(".", "")
            .strip()
        )
        return letter, text_answer
    
    def _parse_mistral_response(self, response: str) -> Tuple[str, str]:
        """Parse Mistral-specific response format."""
        response = response.strip().replace("</s>", "").strip()
        start_answer_string = "the best answer is:"
        if start_answer_string not in response.lower():
            return "", ""
        answer_part = response.lower().split(start_answer_string)[-1]
        import re
        letter_match = re.search(r"\((.)\)", answer_part)
        if not letter_match:
            return "", ""
        letter = letter_match.group(1)
        text_answer = (
            answer_part.split(")")[-1]
            .strip()
            .split(", ")[0]
            .lower()
            .replace(".", "")
            .strip()
        )
        return letter, text_answer
    
    def _parse_generic_response(self, response: str) -> Tuple[str, str]:
        """Generic response parser for unknown model families."""
        response = response.strip()
        start_answer_string = "the best answer is:"
        if start_answer_string not in response.lower():
            return "", ""
        answer_part = response.lower().split(start_answer_string)[-1]
        import re
        letter_match = re.search(r"\((.)\)", answer_part)
        if not letter_match:
            return "", ""
        letter = letter_match.group(1)
        text_answer = (
            answer_part.split(")")[-1]
            .strip()
            .split(", ")[0]
            .lower()
            .replace(".", "")
            .strip()
        )
        return letter, text_answer

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying transformer lens model.
        return getattr(self.model, attr)