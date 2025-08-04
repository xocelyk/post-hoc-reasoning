#!/usr/bin/env python3
"""Test nnsight generation to debug the error."""

import torch
from nnsight import LanguageModel

# Load model
print("Loading model...")
model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)

# Test basic tokenization
test_prompt = "Hello, world!"
print(f"\nTest prompt: {test_prompt}")

# Tokenize
tokens = model.tokenizer(test_prompt, return_tensors="pt")
print(f"Tokenized: {tokens}")
print(f"Input IDs shape: {tokens['input_ids'].shape}")

# Try generation with model.generate
print("\n1. Testing model.generate()...")
try:
    output = model.generate(tokens['input_ids'], max_new_tokens=5)
    print(f"Success! Generated: {model.tokenizer.decode(output[0])}")
except Exception as e:
    print(f"Error with model.generate: {e}")
    import traceback
    traceback.print_exc()

# Try with model.trace
print("\n2. Testing model.trace()...")
try:
    input_ids = tokens['input_ids']
    with model.trace(input_ids):
        if hasattr(model, 'lm_head'):
            logits = model.lm_head.output.save()
        else:
            print("No lm_head found")
    print(f"Success! Got logits shape: {logits.shape}")
except Exception as e:
    print(f"Error with model.trace: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing full custom generation loop...")
try:
    toks = tokens['input_ids'].clone()
    if toks.dim() == 1:
        toks = toks.unsqueeze(0)
    
    for i in range(3):
        print(f"  Step {i}: tokens shape = {toks.shape}")
        with model.trace(toks):
            logits = model.lm_head.output.save()
        
        next_token_logits = logits[:, -1, :]
        next_tok = next_token_logits.argmax(dim=-1, keepdim=True)
        toks = torch.cat([toks, next_tok], dim=-1)
    
    print(f"Success! Final tokens shape: {toks.shape}")
    print(f"Generated text: {model.tokenizer.decode(toks[0])}")
except Exception as e:
    print(f"Error in custom generation: {e}")
    import traceback
    traceback.print_exc()