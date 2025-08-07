#!/usr/bin/env python3
"""
Fix for steering issue in experiment_runner.py

The issue: CAA single-layer method is applying steering to ALL layers with the same vector,
when it should only apply steering to the selected best layer.

This script demonstrates the issue and provides a fix.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from steering_methods import CAASingleLayerSteering

def demonstrate_issue():
    """Show the current issue with CAA single-layer steering."""
    print("="*60)
    print("DEMONSTRATING THE ISSUE")
    print("="*60)
    
    # Example: 3 layers with similarity scores
    similarity_scores = [0.6, 0.8, 0.7]  # Layer 1 has best score
    layer_vectors = [
        np.array([1.0, 0.0, 0.0]),  # Layer 0 vector
        np.array([0.0, 1.0, 0.0]),  # Layer 1 vector (best)
        np.array([0.0, 0.0, 1.0]),  # Layer 2 vector
    ]
    
    # Create steering method
    method = CAASingleLayerSteering(similarity_scores)
    steering_vectors = method.compute_steering_vectors(layer_vectors)
    
    print(f"Layer vectors: {layer_vectors}")
    print(f"Similarity scores: {similarity_scores}")
    print(f"Best layer: {np.argmax(similarity_scores)}")
    print(f"\nSteering vectors returned by CAASingleLayerSteering:")
    for i, vec in enumerate(steering_vectors):
        print(f"  Layer {i}: {vec}")
    
    print("\nISSUE: All layers get the same vector! This means steering is applied at ALL layers.")
    print("This is different from the notebook which only steers at the selected layer.")
    
    return steering_vectors


def show_notebook_approach():
    """Show how the notebook handles single-layer steering."""
    print("\n" + "="*60)
    print("NOTEBOOK APPROACH (CORRECT)")
    print("="*60)
    
    # Same setup
    similarity_scores = [0.6, 0.8, 0.7]
    layer_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),  # Best layer
        np.array([0.0, 0.0, 1.0]),
    ]
    
    best_layer_idx = np.argmax(similarity_scores)
    best_layer = best_layer_idx
    
    # Notebook only stores the best layer's vector
    probe_coefficients = {best_layer: layer_vectors[best_layer_idx]}
    
    print(f"Notebook stores only: {probe_coefficients}")
    
    # During steering, notebook creates zero array and fills only the best layer
    num_layers = 3
    d_model = 3
    steering_vectors_array = np.zeros((num_layers, d_model))
    
    # Only set the selected layer's vector
    for layer, vector in probe_coefficients.items():
        steering_vectors_array[layer] = vector
    
    print(f"\nSteering vectors used in notebook:")
    for i, vec in enumerate(steering_vectors_array):
        print(f"  Layer {i}: {vec}")
    
    print("\nRESULT: Only layer 1 has non-zero steering vector!")
    
    return steering_vectors_array


def proposed_fix():
    """Proposed fix for experiment_runner.py"""
    print("\n" + "="*60)
    print("PROPOSED FIX")
    print("="*60)
    
    print("""
The fix requires modifying how experiment_runner.py handles CAA single-layer steering:

1. In train_probes_and_compute_steering_vectors():
   - After computing final_steering_vectors, check if method is caa-single-layer
   - If so, only save the best layer's vector instead of all replicated vectors
   
2. In generate_steered_examples():
   - When loading steering vectors for caa-single-layer, create zero array
   - Only fill in the selected layer's vector
   
Here's the specific code changes needed:

In experiment_runner.py, after line 455 (computing steering vectors):

```python
# Special handling for single-layer method
if steering_method_name == "caa-single-layer":
    # Find best layer
    best_layer_idx = np.argmax(similarity_scores)
    best_layer = layers[best_layer_idx]
    
    # Only save the best layer's vector
    single_layer_vectors = {best_layer: final_steering_vectors[best_layer_idx]}
    cache.save_pickle(single_layer_vectors, cache.get_probe_coefficients_path())
else:
    # For other methods, save all vectors
    cache.save_pickle(final_steering_vectors, cache.get_probe_coefficients_path())
```

And in generate_steered_examples(), replace the steering vector loading:

```python
# Load steering method metadata to check method type
metadata_path = os.path.join(cache.cache_dir, "steering_metadata.json")
if os.path.exists(metadata_path):
    metadata = cache.load_json(metadata_path)
    steering_method = metadata.get("method", "unknown")
else:
    steering_method = "unknown"

# Handle single-layer method specially
if steering_method == "caa-single-layer" and isinstance(all_contrastive_vectors, dict):
    # Create zero array for all layers
    num_layers = len(layers)
    d_model = list(all_contrastive_vectors.values())[0].shape[0]
    steering_vectors = np.zeros((num_layers, d_model))
    
    # Only fill in the selected layer
    for layer, vector in all_contrastive_vectors.items():
        if layer < num_layers:
            steering_vectors[layer] = vector
else:
    # For other methods, use vectors as-is
    steering_vectors = np.array(all_contrastive_vectors)
```
""")


def verify_fix_impact():
    """Show the impact of the fix."""
    print("\n" + "="*60)
    print("IMPACT OF THE FIX")
    print("="*60)
    
    print("""
With the current implementation:
- Steering is applied at ALL layers with the same vector
- This dilutes the steering effect and may cause unintended interactions
- The model's behavior is modified at every layer, not just the best one

With the fix:
- Steering is applied ONLY at the selected best layer
- This matches the notebook implementation
- More focused steering that should give better results
- Consistent with the CAA single-layer paper's approach

This explains why steering results look worse with run_transformer_lens_experiments.py
compared to the notebook - it's applying steering too broadly instead of precisely.
""")


if __name__ == "__main__":
    demonstrate_issue()
    show_notebook_approach()
    proposed_fix()
    verify_fix_impact()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The key difference is that:
- Notebook: Applies steering ONLY at the best layer (correct)
- experiment_runner.py: Applies steering at ALL layers with same vector (incorrect)

This is why steering performance appears worse when using run_transformer_lens_experiments.py.
The fix involves storing only the best layer's vector for CAA single-layer method and
creating a zero-filled array during steering with only that layer populated.
""")