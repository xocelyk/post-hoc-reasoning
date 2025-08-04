#!/usr/bin/env python3
"""Test chat template application and generation."""

import json
from src.nnsight_models import NNsightChatModel

# Create a test prompt similar to anachronisms dataset
test_item = {
    "prompt": [
        {"role": "user", "content": "Was Alexander the Great a contemporary of Napoleon Bonaparte?"},
        {"role": "assistant", "content": "Let me think about this:"}
    ],
    "correct_answer": "no",
    "correct_letter": "B"
}

print("Test item:", json.dumps(test_item, indent=2))

# Load model
print("\nLoading model...")
model = NNsightChatModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Test chat template
print("\n1. Testing apply_chat_template...")
try:
    formatted = model.apply_chat_template(test_item["prompt"])
    print(f"Success! Formatted length: {len(formatted)}")
    print(f"First 200 chars: {repr(formatted[:200])}...")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test tokenization
print("\n2. Testing tokenization...")
try:
    tokens = model.to_tokens(formatted)
    print(f"Success! Tokens shape: {tokens.shape}")
    print(f"Tokens device: {tokens.device}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test generation
print("\n3. Testing generation...")
try:
    from src.nnsight_utils.core.generation import generate_text
    response = generate_text(model, formatted, max_new_tokens=10, temperature=0.1)
    print(f"Success! Generated: {repr(response[:100])}...")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()