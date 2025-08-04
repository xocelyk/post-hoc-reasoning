#!/usr/bin/env python3
"""Debug script to isolate nnsight activation extraction issue."""

import sys
sys.path.append('/root/ws/post-hoc-reasoning/src')

from nnsight_models import NNsightChatModel
from nnsight_utils import batch_get_resid_activations
from data_loading import load_all_datasets
import traceback

# Load model
print("Loading model...")
model = NNsightChatModel('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')

# Get a real prompt from the dataset
print("Loading dataset...")
datasets = load_all_datasets()
dataset = datasets['anachronisms']
first_item = dataset[0]

# Convert chat messages to string
prompt_string = model.apply_chat_template(first_item['prompt'])
print(f"Prompt length: {len(prompt_string)} chars")

# Tokenize
tokens = model.to_tokens(prompt_string)
print(f"Token shape: {tokens.shape}")

# Try to extract activations
print("\nAttempting activation extraction...")
try:
    activations = batch_get_resid_activations(
        model=model,
        prompts=[prompt_string],
        layers=None,
        position="last"
    )
    print(f"Success! Activations shape: {activations.shape}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
    print("\nFull traceback:")
    traceback.print_exc()