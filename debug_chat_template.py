#!/usr/bin/env python3
"""Debug the chat template issue."""

from src.nnsight_models import NNsightChatModel
from src.data_loading import load_all_datasets

# Load model
print("Loading model...")
model = NNsightChatModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Load dataset
datasets = load_all_datasets()
dataset = datasets["anachronisms"]
item = dataset[0]

print(f"\nDataset item prompt type: {type(item['prompt'])}")
print(f"Dataset item prompt: {item['prompt']}")

# Apply chat template
try:
    formatted = model.apply_chat_template(item["prompt"])
    print(f"\nFormatted type: {type(formatted)}")
    print(f"Formatted length: {len(formatted) if hasattr(formatted, '__len__') else 'no len'}")
    print(f"Formatted[:100]: {repr(formatted[:100]) if hasattr(formatted, '__getitem__') else formatted}")
except Exception as e:
    print(f"Error in apply_chat_template: {e}")
    import traceback
    traceback.print_exc()

# Try tokenization  
try:
    from src.nnsight_utils.core.generation import generate_text
    print("\nTrying generate_text...")
    response = generate_text(model, formatted, max_new_tokens=5)
    print(f"Success! Response: {response}")
except Exception as e:
    print(f"Error in generate_text: {e}")
    import traceback
    traceback.print_exc()