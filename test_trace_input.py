#!/usr/bin/env python3
"""Test what input types model.trace expects."""

import torch
from nnsight import LanguageModel

# Load model
print("Loading model...")
model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)

# Test 1: String input
print("\n1. Testing model.trace with string input...")
try:
    with model.trace("Hello world"):
        if hasattr(model, 'lm_head'):
            logits = model.lm_head.output.save()
    print(f"Success with string! Logits shape: {logits.shape}")
except Exception as e:
    print(f"Error with string: {e}")

# Test 2: List of strings
print("\n2. Testing model.trace with list of strings...")
try:
    with model.trace(["Hello world", "Goodbye world"]):
        if hasattr(model, 'lm_head'):
            logits = model.lm_head.output.save()
    print(f"Success with list! Logits shape: {logits.shape}")
except Exception as e:
    print(f"Error with list: {e}")

# Test 3: Pre-tokenized tensor
print("\n3. Testing model.trace with pre-tokenized tensor...")
try:
    tokens = model.tokenizer("Hello world", return_tensors="pt")["input_ids"]
    print(f"Tokens shape: {tokens.shape}, device: {tokens.device}")
    with model.trace(tokens):
        if hasattr(model, 'lm_head'):
            logits = model.lm_head.output.save()
    print(f"Success with tensor! Logits shape: {logits.shape}")
except Exception as e:
    print(f"Error with tensor: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Dict input like HF
print("\n4. Testing model.trace with dict input...")
try:
    inputs = model.tokenizer("Hello world", return_tensors="pt")
    with model.trace(**inputs):
        if hasattr(model, 'lm_head'):
            logits = model.lm_head.output.save()
    print(f"Success with dict! Logits shape: {logits.shape}")
except Exception as e:
    print(f"Error with dict: {e}")