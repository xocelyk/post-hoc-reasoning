"""
Affine Concept Editing (ACE) for debiasing language models.

This module implements ACE debiasing which computes a unit direction vector
from the difference in mean activations between samples with different predicted
answers (yes vs no), and applies an affine transformation to neutralize bias.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class ACEVectors:
    """Container for ACE debiasing vectors and parameters."""
    unit_direction: np.ndarray  # Unit direction vector for debiasing
    bias: float  # Bias point for centering projections
    layer: int  # Layer where these vectors apply
    std_scale: float  # Standard deviation scaling factor used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "unit_direction": self.unit_direction.tolist(),
            "bias": float(self.bias),
            "layer": int(self.layer),
            "std_scale": float(self.std_scale)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACEVectors":
        """Create from dictionary."""
        return cls(
            unit_direction=np.array(data["unit_direction"]),
            bias=data["bias"],
            layer=data["layer"],
            std_scale=data["std_scale"]
        )


def compute_ace_vectors(
    activations_yes: List[np.ndarray],
    activations_no: List[np.ndarray],
    layer: int
) -> ACEVectors:
    """
    Compute ACE debiasing vectors from activations with different predicted answers.
    
    Args:
        activations_yes: List of activation vectors from samples where model predicted "yes"
        activations_no: List of activation vectors from samples where model predicted "no"  
        layer: Layer number these activations come from
        
    Returns:
        ACEVectors containing the unit direction and bias for debiasing
    """
    # Convert to numpy arrays and stack
    resid_yes = np.stack(activations_yes, axis=0)  # Shape: (n_yes, d_model)
    resid_no = np.stack(activations_no, axis=0)    # Shape: (n_no, d_model)
    
    # Compute group means
    mean_yes = np.mean(resid_yes, axis=0)  # Shape: (d_model,)
    mean_no = np.mean(resid_no, axis=0)    # Shape: (d_model,)
    
    # Compute difference in group means
    direction = mean_yes - mean_no  # Shape: (d_model,)
    
    # Scale by standard deviation of all activations
    all_activations = np.concatenate([resid_yes, resid_no], axis=0)  # Shape: (n_yes + n_no, d_model)
    std_all = np.std(all_activations, axis=0, ddof=1)  # Shape: (d_model,)
    
    # Avoid division by zero - use small epsilon for dimensions with zero std
    std_all = np.where(std_all == 0, 1e-8, std_all)
    
    # Scale direction by standard deviation
    scaled_direction = direction / std_all  # Shape: (d_model,)
    std_scale = np.mean(std_all)  # Store average std for reference
    
    # Unit normalize the direction
    norm = np.linalg.norm(scaled_direction)
    if norm == 0:
        raise ValueError("Direction vector has zero norm - cannot create unit direction")
    
    unit_direction = scaled_direction / norm  # Shape: (d_model,)
    
    # Compute bias as midpoint between group mean projections onto unit direction
    proj_mean_yes = np.dot(mean_yes, unit_direction)  # Scalar
    proj_mean_no = np.dot(mean_no, unit_direction)    # Scalar
    
    bias = (proj_mean_yes + proj_mean_no) / 2.0
    
    return ACEVectors(
        unit_direction=unit_direction,
        bias=bias,
        layer=layer,
        std_scale=std_scale
    )


def apply_ace_intervention(
    activations: np.ndarray,
    ace_vectors: ACEVectors
) -> np.ndarray:
    """
    Apply ACE debiasing intervention to activations.
    
    Args:
        activations: Input activations to debias, shape (batch_size, d_model) or (d_model,)
        ace_vectors: ACE vectors containing unit direction and bias
        
    Returns:
        Debiased activations with same shape as input
    """
    # Handle single vector case
    was_1d = False
    if activations.ndim == 1:
        activations = activations[None, :]  # Add batch dimension
        was_1d = True
    
    # Project onto unit direction
    projections = np.dot(activations, ace_vectors.unit_direction)  # Shape: (batch_size,)
    
    # Compute intervention: (projection - bias) * unit_direction
    intervention = (projections - ace_vectors.bias)[:, None] * ace_vectors.unit_direction[None, :]
    
    # Apply intervention: x_debiased = x - intervention
    debiased = activations - intervention
    
    # Remove batch dimension if input was 1D
    if was_1d:
        debiased = debiased[0]
    
    return debiased


def evaluate_ace_debiasing(
    original_activations: List[np.ndarray],
    debiased_activations: List[np.ndarray], 
    original_predictions: List[str],
    debiased_predictions: List[str]
) -> Dict[str, Any]:
    """
    Evaluate the effectiveness of ACE debiasing.
    
    Args:
        original_activations: Original activation vectors
        debiased_activations: Debiased activation vectors
        original_predictions: Original model predictions
        debiased_predictions: Predictions after debiasing
        
    Returns:
        Dictionary with evaluation metrics
    """
    n_samples = len(original_predictions)
    
    # Compute change in predictions
    prediction_changes = sum(
        1 for orig, debiased in zip(original_predictions, debiased_predictions)
        if orig != debiased
    )
    
    # Compute magnitude of intervention
    intervention_magnitudes = []
    for orig, debiased in zip(original_activations, debiased_activations):
        intervention_mag = np.linalg.norm(orig - debiased)
        intervention_magnitudes.append(intervention_mag)
    
    # Compute relative intervention magnitude
    original_magnitudes = [np.linalg.norm(act) for act in original_activations]
    relative_interventions = [
        interv / orig if orig > 0 else 0
        for interv, orig in zip(intervention_magnitudes, original_magnitudes)
    ]
    
    return {
        "total_samples": n_samples,
        "prediction_changes": prediction_changes,
        "prediction_change_rate": prediction_changes / n_samples if n_samples > 0 else 0,
        "mean_intervention_magnitude": float(np.mean(intervention_magnitudes)),
        "std_intervention_magnitude": float(np.std(intervention_magnitudes)),
        "mean_relative_intervention": float(np.mean(relative_interventions)),
        "std_relative_intervention": float(np.std(relative_interventions)),
        "original_yes_rate": sum(1 for p in original_predictions if p.lower() == "yes") / n_samples if n_samples > 0 else 0,
        "debiased_yes_rate": sum(1 for p in debiased_predictions if p.lower() == "yes") / n_samples if n_samples > 0 else 0
    }


class ACEDebiasingMethod:
    """
    Class for managing ACE debiasing experiments.
    
    This handles the full pipeline from computing debiasing vectors from training data
    to applying interventions during test-time generation.
    """
    
    def __init__(self, layer: int):
        """
        Initialize ACE debiasing for a specific layer.
        
        Args:
            layer: Layer number to apply debiasing at
        """
        self.layer = layer
        self.ace_vectors: Optional[ACEVectors] = None
    
    def fit(
        self,
        train_activations: List[np.ndarray],
        train_predictions: List[str]
    ) -> ACEVectors:
        """
        Fit ACE debiasing vectors from training data.
        
        Args:
            train_activations: Training activation vectors
            train_predictions: Training predictions ("yes" or "no")
            
        Returns:
            Computed ACE vectors
        """
        # Separate activations by predicted answer
        activations_yes = []
        activations_no = []
        
        for activation, prediction in zip(train_activations, train_predictions):
            pred_lower = prediction.lower().strip()
            if pred_lower == "yes":
                activations_yes.append(activation)
            elif pred_lower == "no":
                activations_no.append(activation)
        
        if len(activations_yes) == 0:
            raise ValueError("No samples with 'yes' predictions found in training data")
        if len(activations_no) == 0:
            raise ValueError("No samples with 'no' predictions found in training data")
        
        print(f"Computing ACE vectors from {len(activations_yes)} 'yes' and {len(activations_no)} 'no' samples")
        
        # Compute ACE vectors
        self.ace_vectors = compute_ace_vectors(activations_yes, activations_no, self.layer)
        
        return self.ace_vectors
    
    def transform(self, activations: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply ACE debiasing to test activations.
        
        Args:
            activations: Test activation vectors to debias
            
        Returns:
            Debiased activation vectors
        """
        if self.ace_vectors is None:
            raise ValueError("Must call fit() before transform()")
        
        debiased = []
        for activation in activations:
            debiased_activation = apply_ace_intervention(activation, self.ace_vectors)
            debiased.append(debiased_activation)
        
        return debiased
    
    def save(self, filepath: str):
        """Save ACE vectors to file."""
        if self.ace_vectors is None:
            raise ValueError("No ACE vectors to save. Call fit() first.")
        
        data = {
            "layer": self.layer,
            "ace_vectors": self.ace_vectors.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load ACE vectors from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.layer = data["layer"]
        self.ace_vectors = ACEVectors.from_dict(data["ace_vectors"])