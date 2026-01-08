#!/usr/bin/env python3
"""
Test the biasing config system.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ConfigLoader

def main():
    """Test loading biasing config."""
    print("=== Testing Biasing Config ===")
    
    try:
        # Load the biasing experiments config
        config = ConfigLoader.load_experiment_config("configs/biasing_experiments.yaml")
        
        print(f"Loaded config with {len(config.models)} models and {len(config.datasets)} datasets")
        print(f"Bias evaluation: {config.bias_evaluation}")
        print(f"Cross-bias debiasing: {config.cross_bias_debiasing}")
        
        print("\nDatasets:")
        for i, dataset in enumerate(config.datasets):
            print(f"  {i+1}. {dataset.name}")
            print(f"     Train bias: {dataset.train_bias}")
            print(f"     Test bias: {dataset.test_bias}")
            if dataset.train_dataset:
                print(f"     Train dataset: {dataset.train_dataset}")
            if dataset.test_dataset:
                print(f"     Test dataset: {dataset.test_dataset}")
            print()
        
        print("✅ Config loading successful!")
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())