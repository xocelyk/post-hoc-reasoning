from transformer_lens import HookedTransformer
from typing import List, Dict

class ChatModel:
    def __init__(self, model_name: str, device: str = 'cpu', dtype: str = 'bfloat16', chat_type: str = "gemma"):
        """
        Initialize the ChatModel.

        :param model_name: Name of the model to load via transformer_lens.
        :param device: Device to run the model on.
        :param dtype: Data type for model weights.
        :param chat_type: Chat formatting style. Options are "gemma" for Gemma 2 IT style and "llama" for Llama 2 Chat style.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.chat_type = chat_type.lower()
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=self.device,
            dtype=self.dtype
        )

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a list of chat messages according to the specified chat_type.
        """
        return self.model.tokenizer.apply_chat_template(messages, tokenize=False)

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying transformer lens model.
        return getattr(self.model, attr)