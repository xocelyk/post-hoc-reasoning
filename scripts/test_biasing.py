#!/usr/bin/env python3
"""
Simple test script to validate biasing functionality.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading import load_fewshot_examples, create_biased_cot_prompt

def test_fewshot_loading():
    """Test loading fewshot examples."""
    print("Testing fewshot loading...")
    
    # Test with sports_understanding
    pools = load_fewshot_examples("sports_understanding")
    print(f"Loaded {len(pools['yes'])} 'yes' examples and {len(pools['no'])} 'no' examples")
    
    if pools['yes']:
        print("Sample 'yes' example:")
        print(f"  Statement: {pools['yes'][0]['statement']}")
        print(f"  Answer: {pools['yes'][0]['answer']}")
    
    if pools['no']:
        print("Sample 'no' example:")
        print(f"  Statement: {pools['no'][0]['statement']}")
        print(f"  Answer: {pools['no'][0]['answer']}")
    
    return pools

def test_biased_prompts(pools):
    """Test creating biased prompts."""
    print("\nTesting biased prompt creation...")
    
    task_name = "sports_understanding"
    target_label = "yes"
    
    # Test positive bias
    print(f"\n--- Positive bias (target: {target_label}) ---")
    positive_prompt = create_biased_cot_prompt(task_name, target_label, "positive", pools, num_examples=2)
    print(f"Created prompt with {len(positive_prompt)} messages")
    for i, msg in enumerate(positive_prompt):
        print(f"Message {i} ({msg['role']}): {msg['content'][:100]}...")
    
    # Test negative bias
    print(f"\n--- Negative bias (target: {target_label}) ---")
    negative_prompt = create_biased_cot_prompt(task_name, target_label, "negative", pools, num_examples=2)
    print(f"Created prompt with {len(negative_prompt)} messages")
    for i, msg in enumerate(negative_prompt):
        print(f"Message {i} ({msg['role']}): {msg['content'][:100]}...")
    
    # Test neutral bias (None)
    print(f"\n--- Neutral bias (target: {target_label}) ---")
    neutral_prompt = create_biased_cot_prompt(task_name, target_label, None, pools, num_examples=2)
    print(f"Created prompt with {len(neutral_prompt)} messages")
    for i, msg in enumerate(neutral_prompt):
        print(f"Message {i} ({msg['role']}): {msg['content'][:100]}...")

def main():
    """Run all tests."""
    print("=== Biasing Implementation Test ===")
    
    try:
        pools = test_fewshot_loading()
        test_biased_prompts(pools)
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())