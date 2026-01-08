# %%
from nnsight import LanguageModel
from transformer_lens import HookedTransformer

model_name = "google/gemma-2-9b-it"

model = LanguageModel(model_name)

# model.generate("Hello, how are you?")

# %%