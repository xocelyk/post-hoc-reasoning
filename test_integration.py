#!/usr/bin/env python3
"""
Integration test script to validate the nnsight implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def test_basic_functionality():
    """Test basic functionality without loading large models."""
    print("🧪 Running integration tests...")
    
    # Test 1: Configuration loading
    print("\n1. Testing configuration loading...")
    try:
        from config import ConfigLoader, ModelConfig, DatasetConfig, SteeringConfig, ExperimentRunConfig
        
        # Create a minimal config
        config = ExperimentRunConfig(
            models=[ModelConfig(name="test-model", backend="auto")],
            datasets=[DatasetConfig(name="sports_understanding")],
            steering=SteeringConfig(alpha_range=[0, 1, 2]),
        )
        print("✓ Configuration creation works")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 2: Model factory
    print("\n2. Testing model factory...")
    try:
        from model_factory import get_recommended_backend, validate_model_config
        
        backend = get_recommended_backend("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        assert backend == "nnsight", f"Expected nnsight, got {backend}"
        
        validation = validate_model_config("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "transformer_lens")
        assert not validation["valid"], "Should flag DeepSeek + transformer_lens as invalid"
        
        print("✓ Model factory works")
        
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        return False
    
    # Test 3: Experiment runner creation
    print("\n3. Testing experiment runner creation...")
    try:
        from nnsight_experiment_runner import NNsightExperimentRunner
        
        runner = NNsightExperimentRunner(config)
        assert len(runner.experiment_configs) == 1, f"Expected 1 config, got {len(runner.experiment_configs)}"
        
        print("✓ Experiment runner creation works")
        
    except Exception as e:
        print(f"❌ Experiment runner test failed: {e}")
        return False
    
    # Test 4: Data loading
    print("\n4. Testing data loading...")
    try:
        from data_loading import load_all_datasets, list_available_datasets
        
        available = list_available_datasets()
        assert "sports_understanding" in available, "sports_understanding should be available"
        
        # Load a small sample 
        datasets = load_all_datasets()
        sports_data = datasets["sports_understanding"]
        assert len(sports_data) > 0, "Sports dataset should not be empty"
        
        print(f"✓ Data loading works (loaded {len(sports_data)} samples)")
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False
    
    # Test 5: Backend selection logic
    print("\n5. Testing backend selection...")
    try:
        # Import the updated run_experiments functions
        sys.path.append('.')
        from run_experiments import select_experiment_runner, analyze_backend_configuration
        
        # Test with nnsight models
        nnsight_config = ExperimentRunConfig(
            models=[ModelConfig(name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", backend="nnsight")],
            datasets=[DatasetConfig(name="sports_understanding")],
            steering=SteeringConfig(),
        )
        
        runner_type = select_experiment_runner(nnsight_config)
        assert runner_type == "nnsight", f"Expected nnsight runner, got {runner_type}"
        
        print("✓ Backend selection works")
        
    except Exception as e:
        print(f"❌ Backend selection test failed: {e}")
        return False
    
    print("\n🎉 All integration tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)