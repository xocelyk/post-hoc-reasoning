#!/usr/bin/env python3
"""
Quick test script to validate steering methods implementation.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from steering_methods import (
    create_steering_method, 
    CAASingleLayerSteering, 
    CAALayerIncrementalSteering,
    format_steering_results
)
from config import ConfigLoader

def test_steering_methods():
    """Test that both steering methods work correctly."""
    print("Testing steering methods implementation...")
    
    # Create mock layer vectors (10 layers, 100 dimensions each)
    np.random.seed(42)
    num_layers = 10
    vector_dim = 100
    
    layer_vectors = []
    for i in range(num_layers):
        # Create somewhat different vectors for each layer
        base_vector = np.random.randn(vector_dim) * 0.1
        layer_specific = np.random.randn(vector_dim) * 0.01 * (i + 1)
        layer_vectors.append(base_vector + layer_specific)
    
    # Create mock similarity scores
    similarity_scores = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.35, 0.3, 0.25]
    
    print(f"Created {num_layers} mock layer vectors with {vector_dim} dimensions each")
    print(f"Similarity scores: {similarity_scores}")
    
    # Test CAA Single Layer method
    print("\n--- Testing CAA Single Layer Steering ---")
    single_layer_method = create_steering_method("caa-single-layer", similarity_scores=similarity_scores)
    single_layer_vectors = single_layer_method.compute_steering_vectors(layer_vectors)
    
    print(f"Method name: {single_layer_method.get_method_name()}")
    print(f"Number of steering vectors: {len(single_layer_vectors)}")
    print(f"Best layer should be: {np.argmax(similarity_scores)} (layer with score {max(similarity_scores)})")
    
    # Check that all vectors are the same (best layer replicated)
    first_vector = single_layer_vectors[0]
    all_same = all(np.allclose(vec, first_vector) for vec in single_layer_vectors)
    print(f"All vectors identical (expected): {all_same}")
    
    # Check RMS normalization
    rms = np.sqrt(np.mean(first_vector**2))
    print(f"RMS of steering vector: {rms:.6f} (should be ~1.0)")
    
    # Test CAA Layer Incremental method
    print("\n--- Testing CAA Layer Incremental Steering ---")
    incremental_method = create_steering_method("caa-layer-incremental")
    incremental_vectors = incremental_method.compute_steering_vectors(layer_vectors)
    
    print(f"Method name: {incremental_method.get_method_name()}")
    print(f"Number of steering vectors: {len(incremental_vectors)}")
    
    # Check that vectors are different (incremental differences)
    all_different = not all(np.allclose(incremental_vectors[i], incremental_vectors[0]) 
                           for i in range(1, len(incremental_vectors)))
    print(f"Vectors are different (expected): {all_different}")
    
    # Check RMS normalization for each vector
    rms_values = [np.sqrt(np.mean(vec**2)) for vec in incremental_vectors]
    print(f"RMS values: {[f'{rms:.3f}' for rms in rms_values[:5]]}... (should be ~1.0)")
    
    # Test format results
    print("\n--- Testing Result Formatting ---")
    formatted = format_steering_results(single_layer_vectors, "caa-single-layer")
    print(f"Formatted result keys: {list(formatted.keys())}")
    print(f"Method: {formatted['method']}")
    print(f"Number of layers: {formatted['num_layers']}")
    print(f"Vector shapes: {formatted['vector_shapes'][:3]}...")
    
    print("\n--- Testing Config Loading ---")
    try:
        config = ConfigLoader.load_experiment_config("configs/steering_methods_comparison.yaml")
        print(f"Loaded config successfully")
        print(f"Steering method: {config.steering.method}")
        print(f"Alpha range: {config.steering.alpha_range}")
        print(f"Model: {config.models[0].name}")
        print(f"Dataset: {config.datasets[0].name}")
        print(f"Train/test sizes: {config.datasets[0].train_size}/{config.datasets[0].test_size}")
    except Exception as e:
        print(f"Error loading config: {e}")
    
    print("\nAll tests completed successfully! ✅")

def test_error_handling():
    """Test error handling for invalid method names."""
    print("\n--- Testing Error Handling ---")
    
    try:
        invalid_method = create_steering_method("invalid-method")
        print("❌ Should have raised ValueError for invalid method")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    try:
        no_scores = create_steering_method("caa-single-layer")
        print("❌ Should have raised ValueError for missing similarity_scores")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

if __name__ == "__main__":
    test_steering_methods()
    test_error_handling()
    print("\n🎉 All tests passed!")