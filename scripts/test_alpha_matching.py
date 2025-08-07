#!/usr/bin/env python3
"""Test alpha matching logic"""

# Test the filename matching
test_files = [
    "steering_alpha_-10_yes.pkl",
    "steering_alpha_-2_yes.pkl", 
    "steering_alpha_0_yes.pkl",
    "steering_alpha_2_yes.pkl",
    "steering_alpha_10_yes.pkl",
    "steering_alpha_-2.0_yes.pkl",
    "steering_alpha_2.0_yes.pkl"
]

# Test different alpha values
test_alphas = [-10, -2, 0, 2, 10, -2.0, 2.0]

for alpha in test_alphas:
    print(f"\nTesting alpha = {alpha}")
    
    # Test exact match
    pattern = f'steering_alpha_{alpha}'
    print(f"  Pattern: '{pattern}'")
    
    matching = []
    for filename in test_files:
        if filename.startswith(pattern):
            print(f"    ✓ Matches: {filename}")
            matching.append(filename)
    
    if not matching:
        print(f"    ✗ No matches found")
        
    # Test with float conversion
    pattern2 = f'steering_alpha_{float(alpha)}'
    if pattern2 != pattern:
        print(f"  Alt pattern: '{pattern2}'")
        for filename in test_files:
            if filename.startswith(pattern2):
                print(f"    ✓ Matches: {filename}")