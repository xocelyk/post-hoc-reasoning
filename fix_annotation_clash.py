#!/usr/bin/env python3
"""
Fix for annotation clash in Social Chemistry plot (and other high-density plots)
Replace the existing annotation logic in your visualization notebook with this code.
"""

import numpy as np
import matplotlib.pyplot as plt

def add_staggered_annotations(ax, annotation_data, min_distance=0.03):
    """
    Add annotations with collision avoidance for overlapping labels.
    
    Args:
        ax: matplotlib axis object
        annotation_data: list of tuples (x, y, text, model_name)
        min_distance: minimum distance between annotations to avoid overlap
    """
    # Sort annotations by y-value (highest first)
    sorted_annotations = sorted(annotation_data, key=lambda x: x[1], reverse=True)
    
    # Model-specific styling
    model_colors = {
        'google/gemma-2-2b-it': '#1f77b4',
        'google/gemma-2-9b-it': '#ff7f0e', 
        'Qwen/Qwen2.5-1.5B-Instruct': '#2ca02c',
        'Qwen/Qwen2.5-3B-Instruct': '#d62728',
        'Qwen/Qwen2.5-7B-Instruct': '#9467bd'
    }
    
    placed_annotations = []
    
    for x, y, text, model in sorted_annotations:
        # Find the best position for this annotation
        best_pos = find_best_annotation_position(ax, x, y, placed_annotations, min_distance)
        
        # Get model-specific color
        text_color = model_colors.get(model, 'black')
        
        # Add annotation with improved styling
        annotation = ax.annotate(
            text,
            xy=(x, y),
            xytext=best_pos,
            ha='center', va='bottom',
            fontsize=9, 
            fontweight='bold',
            color=text_color,
            bbox=dict(
                boxstyle='round,pad=0.3', 
                facecolor='white', 
                alpha=0.9, 
                edgecolor=text_color,
                linewidth=0.8
            ),
            arrowprops=dict(
                arrowstyle='-',
                color=text_color,
                alpha=0.6,
                linewidth=1
            )
        )
        
        # Track this annotation position
        placed_annotations.append(best_pos)
    
    return ax

def find_best_annotation_position(ax, x, y, placed_positions, min_distance):
    """
    Find the best position for an annotation that doesn't overlap with existing ones.
    """
    # Possible offset positions (in data coordinates)
    candidate_positions = [
        (x, y + 0.04),      # directly above
        (x + 0.02, y + 0.03), # upper right
        (x - 0.02, y + 0.03), # upper left
        (x, y + 0.06),      # higher above
        (x + 0.04, y + 0.02), # far right
        (x - 0.04, y + 0.02), # far left
        (x + 0.03, y + 0.05), # upper right far
        (x - 0.03, y + 0.05), # upper left far
    ]
    
    # Find the first position that doesn't conflict
    for candidate in candidate_positions:
        if not any(distance(candidate, placed) < min_distance for placed in placed_positions):
            return candidate
    
    # If all positions conflict, use the first one (fallback)
    return candidate_positions[0]

def distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Example usage - replace your existing annotation code with something like this:
"""
# In your visualization cell, replace the annotation section with:

# Collect all annotation data for this subplot
annotation_data = []
for model in MODELS:
    # ... your existing data loading code ...
    if max_auc > 0.6:  # Your existing threshold
        annotation_data.append((max_layer_norm, max_auc, f'{max_auc:.3f}', model))

# Apply staggered annotations to avoid collisions
add_staggered_annotations(ax, annotation_data)
"""

print("Annotation clash fix script created!")
print("To implement: Replace your annotation logic in the visualization notebook")
print("with the add_staggered_annotations() function above.") 