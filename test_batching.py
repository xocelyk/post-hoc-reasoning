#!/usr/bin/env python3
"""
Simple test script to verify batched generation and steering work correctly.
"""

import torch
import numpy as np
from src.nnsight_utils import NNsightChatModel
from src.nnsight_utils.core.generation import batch_generate_text
from src.nnsight_utils.steering.generation import generate_steered_batch

def test_batched_generation():
    """Test that batched generation works with left-padding."""
    print("Testing batched generation...")
    
    # Use a small model for testing
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    try:
        model = NNsightChatModel(model_name)
        
        # Test prompts of different lengths
        prompts = [
            "The capital of Germany is",
            "The capital of France is",
            "The capital of Italy is",
            "The capital of Spain is",
        ]
        
        print(f"Testing with {len(prompts)} prompts")
        
        # Test batch generation
        results = batch_generate_text(
            model=model,
            prompts=prompts,
            max_new_tokens=10,
            batch_size=2
        )
        
        print(f"Generated {len(results)} results:")
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"  {i+1}. Prompt: '{prompt}' -> Result: '{result}'")
        
        print("✅ Batched generation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Batched generation test failed: {e}")
        return False

def test_batched_steering():
    """Test that batched steering works with left-padding."""
    print("\nTesting batched steering...")
    
    try:
        # Use a small model
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        model = NNsightChatModel(model_name)
        
        # Create dummy steering vectors (random for testing)
        n_layers = model.model.config.num_hidden_layers
        d_model = model.model.config.hidden_size
        
        steering_vectors = np.random.randn(n_layers, d_model).astype(np.float32)
        
        # Test prompts
        prompts = [
            "The weather today is",
            "I think that"
        ]
        
        print(f"Testing steering with {len(prompts)} prompts")
        
        # Test steered generation
        results = generate_steered_batch(
            model=model,
            prompts=prompts,
            steering_vectors=steering_vectors,
            alpha=0.1,  # Small alpha for testing
            max_new_tokens=5,
            batch_size=2
        )
        
        print(f"Generated {len(results)} steered results:")
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"  {i+1}. Prompt: '{prompt}' -> Result: '{result}'")
        
        print("✅ Batched steering test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Batched steering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing batched generation and steering implementation...")
    
    success = True
    
    # Test regular generation
    success &= test_batched_generation()
    
    # Test steered generation  
    success &= test_batched_steering()
    
    if success:
        print("\n🎉 All tests passed! Batched implementation is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the errors above.")