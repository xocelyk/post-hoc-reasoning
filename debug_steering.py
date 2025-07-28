#!/usr/bin/env python3
"""
Debug script to understand KV-cache + steering interaction.
This will help us definitively answer whether steering accumulates.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
from functools import partial
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

from transformer_lens import HookedTransformer

# Global variables to track what happens
steering_log = []
position_modifications = {}

def debug_steer_residual_stream(
    residual_component: torch.FloatTensor,
    hook: HookPoint,
    steering_vectors: torch.Tensor,
    alpha: float = 1.0,
    instruction_pos: int = 0,
    step_num: int = 0,
) -> torch.FloatTensor:
    """
    Debug version of steer_residual_stream that logs everything.
    """
    global steering_log, position_modifications
    
    layer = hook.layer()
    batch_size, seq_len, d_model = residual_component.shape
    
    # Log the call
    log_entry = {
        'step': step_num,
        'layer': layer,
        'seq_len': seq_len,
        'instruction_pos': instruction_pos,
        'will_steer': seq_len > instruction_pos,
        'positions_to_steer': list(range(instruction_pos, seq_len)) if seq_len > instruction_pos else []
    }
    
    # Store original values at positions we might modify
    if seq_len > instruction_pos:
        for pos in range(instruction_pos, seq_len):
            key = f"step_{step_num}_layer_{layer}_pos_{pos}"
            position_modifications[key] = {
                'original_norm': torch.norm(residual_component[0, pos, :]).item(),
                'steering_added': True
            }
    
    steering_log.append(log_entry)
    
    # Apply steering (same as original)
    if seq_len > instruction_pos:
        steering_vector = steering_vectors[layer]
        add_act = torch.tensor(alpha * steering_vector).to(residual_component.device)
        residual_component[:, instruction_pos:, :] += add_act
        
        # Log the modification
        print(f"STEERING APPLIED - Step {step_num}, Layer {layer}: "
              f"Modified positions {instruction_pos}:{seq_len} "
              f"(shape: {residual_component.shape})")
    else:
        print(f"NO STEERING - Step {step_num}, Layer {layer}: "
              f"seq_len={seq_len} <= instruction_pos={instruction_pos}")
    
    return residual_component

def debug_generate_with_hooks(
    model,
    tokens: torch.Tensor,
    steering_vectors: torch.Tensor,
    max_new_tokens: int = 3,  # Just a few tokens for debugging
    alpha: float = 1.0,
    layers: list = None,
) -> str:
    """
    Debug version of generate_with_hooks with extensive logging.
    """
    global steering_log, position_modifications
    steering_log = []
    position_modifications = {}
    
    print(f"\n=== STARTING DEBUG GENERATION ===")
    print(f"Initial tokens shape: {tokens.shape}")
    print(f"Initial prompt: {model.to_string(tokens)}")
    
    # Initialize KV cache
    kv_cache = HookedTransformerKeyValueCache.init_cache(
        cfg=model.cfg,
        device=tokens.device,
        batch_size=tokens.size(0),
    )
    
    if layers is None:
        layers = range(min(3, model.cfg.n_layers))  # Just first 3 layers for debugging
    
    instruction_pos = tokens.size(1)
    print(f"instruction_pos: {instruction_pos}")
    
    # CRITICAL TEST: Let's also try with instruction_pos = 0 to see if steering would work
    print(f"\n=== TESTING WITH instruction_pos=0 (should enable steering) ===")
    test_instruction_pos = 0
    
    # Step 0: Initial forward pass
    print(f"\n--- STEP 0: Initial forward pass ---")
    partial_steer_func = partial(
        debug_steer_residual_stream,
        steering_vectors=steering_vectors,
        alpha=alpha,
        instruction_pos=instruction_pos,
        step_num=0,
    )
    
    hooks = [
        (utils.get_act_name("resid_post", layer), partial_steer_func)
        for layer in layers
    ]
    
    with torch.no_grad():
        logits_full_prompt = model.run_with_hooks(
            tokens,
            fwd_hooks=hooks,
            return_type="logits",
            past_kv_cache=kv_cache,
        )
    
    model.reset_hooks()
    generated_tokens = []
    
    # Generation steps
    for step in range(1, max_new_tokens + 1):
        print(f"\n--- STEP {step}: Generating token {step} ---")
        print(f"Current tokens shape: {tokens.shape}")
        
        # Update hook for this step
        partial_steer_func = partial(
            debug_steer_residual_stream,
            steering_vectors=steering_vectors,
            alpha=alpha,
            instruction_pos=test_instruction_pos,  # Use 0 to test steering
            step_num=step,
        )
        
        hooks = [
            (utils.get_act_name("resid_post", layer), partial_steer_func)
            for layer in layers
        ]
        
        with torch.no_grad():
            logits_step = model.run_with_hooks(
                tokens[:, -1:],  # This is the key part to understand
                fwd_hooks=hooks,
                return_type="logits",
                past_kv_cache=kv_cache,
            )
        
        model.reset_hooks()
        
        # Sample next token
        next_logits = logits_step[:, -1, :]
        probs = torch.nn.functional.softmax(next_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        generated_tokens.append(next_token_id)
        
        # Append to tokens
        next_token_tensor = torch.tensor([[next_token_id]], device=tokens.device)
        tokens = torch.cat([tokens, next_token_tensor], dim=1)
        
        print(f"Generated token: {model.to_string(torch.tensor([[next_token_id]]))}")
        print(f"Full sequence so far: {model.to_string(tokens)}")
        
        # Stop early if we hit end token
        if next_token_id == model.tokenizer.eos_token_id:
            break
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    print(f"Total steering calls: {len(steering_log)}")
    
    # Check for accumulation
    positions_steered_multiple_times = {}
    for entry in steering_log:
        for pos in entry['positions_to_steer']:
            if pos not in positions_steered_multiple_times:
                positions_steered_multiple_times[pos] = []
            positions_steered_multiple_times[pos].append((entry['step'], entry['layer']))
    
    print("\nPositions steered multiple times:")
    for pos, steerings in positions_steered_multiple_times.items():
        if len(steerings) > len(layers):  # More than once per layer = accumulation
            print(f"  Position {pos}: steered {len(steerings)} times across steps/layers: {steerings}")
    
    # Print detailed log
    print(f"\nDetailed steering log:")
    for entry in steering_log:
        print(f"  Step {entry['step']}, Layer {entry['layer']}: "
              f"seq_len={entry['seq_len']}, will_steer={entry['will_steer']}, "
              f"positions={entry['positions_to_steer']}")
    
    return model.to_string(torch.tensor([generated_tokens]))

def main():
    print("Loading GPT-2 small model...")
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")  # GPT-2 small for M1 Mac
    
    # Create dummy steering vectors (GPT-2 has 12 layers, 768 dim)
    n_layers = min(3, model.cfg.n_layers)  # Just test first 3 layers
    steering_vectors = np.random.randn(n_layers, model.cfg.d_model) * 0.01  # Small random vectors
    
    # Simple test prompt
    prompt = "The cat"
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    print("Running debug generation...")
    result = debug_generate_with_hooks(
        model=model,
        tokens=tokens,
        steering_vectors=steering_vectors,
        max_new_tokens=3,
        alpha=0.05,  # Very small alpha for safety
    )
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Generated text: {result}")

if __name__ == "__main__":
    main()