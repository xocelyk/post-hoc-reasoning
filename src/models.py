from transformer_lens import HookedTransformer
from typing import List, Dict

class ChatModel:
    def __init__(self, model_name, device='cpu', dtype='bfloat16'):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Initialize the transformer lens model
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=self.device,
            dtype=self.dtype
        )

    def apply_chat_template(self, user_input: List[Dict[str, str]]) -> str:
        """
        Provide a custom chat template based on the model_name.
        """
        # TODO
        if self.model_name.lower().startswith("google/gemma"):
            return self.apply_chat_template(user_input)
        elif self.model_name.lower().startswith("meta-llama"):
            return self.apply_chat_template_llama(user_input)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
