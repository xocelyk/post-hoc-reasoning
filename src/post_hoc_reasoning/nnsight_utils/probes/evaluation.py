"""
Probe evaluation utilities.

This module provides functions for evaluating and comparing different
probe training methods.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .base import ProbeResult, convert_labels_to_binary


def evaluate_probe_performance(
    probe_result: ProbeResult,
    test_activations: np.ndarray,
    test_labels: List[str],
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Comprehensive evaluation of probe performance.
    
    Args:
        probe_result: Trained probe results
        test_activations: Test activation array
        test_labels: Test labels
        threshold: Decision threshold for classification
        
    Returns:
        Dictionary with various evaluation metrics
    """
    binary_labels = convert_labels_to_binary(test_labels)
    
    # Get the best performing layer (or use all layers for multi-layer methods)
    if probe_result.best_layer is not None:
        # Single-layer method
        eval_layers = [probe_result.best_layer]
    else:
        # Multi-layer method - evaluate average performance
        eval_layers = list(probe_result.vectors.keys())
    
    all_similarities = []
    layer_scores = {}
    
    for layer in eval_layers:
        layer_acts = test_activations[:, layer, :]
        vector = probe_result.vectors[layer]
        
        # Compute similarities
        similarities = layer_acts @ vector
        all_similarities.extend(similarities)
        
        # Layer-specific AUC
        try:
            layer_auc = roc_auc_score(binary_labels, similarities)
        except ValueError:
            layer_auc = 0.5
        layer_scores[layer] = layer_auc
    
    # Overall performance (average over layers for multi-layer)
    if len(eval_layers) == 1:
        similarities = test_activations[:, eval_layers[0], :] @ probe_result.vectors[eval_layers[0]]
    else:
        # For multi-layer, average the similarities
        total_similarities = np.zeros(len(test_labels))
        for layer in eval_layers:
            layer_acts = test_activations[:, layer, :]
            vector = probe_result.vectors[layer]
            total_similarities += layer_acts @ vector
        similarities = total_similarities / len(eval_layers)
    
    # Compute metrics
    try:
        auc = roc_auc_score(binary_labels, similarities)
    except ValueError:
        auc = 0.5
    
    # Classification metrics using threshold
    predictions = (similarities > threshold).astype(int)
    accuracy = accuracy_score(binary_labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels, predictions, average='binary', zero_division=0
    )
    
    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "layer_scores": layer_scores,
        "num_layers": len(eval_layers),
        "method": probe_result.method
    }


def compare_probe_methods(
    probe_results: List[ProbeResult],
    test_activations: np.ndarray,
    test_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple probe methods on the same test data.
    
    Args:
        probe_results: List of trained probe results
        test_activations: Test activation array
        test_labels: Test labels
        
    Returns:
        Dictionary mapping method names to evaluation metrics
    """
    comparison = {}
    
    for probe_result in probe_results:
        method_name = f"{probe_result.method}"
        if probe_result.best_layer is not None:
            method_name += f"_layer_{probe_result.best_layer}"
        
        metrics = evaluate_probe_performance(
            probe_result, test_activations, test_labels
        )
        comparison[method_name] = metrics
    
    return comparison


def find_optimal_threshold(
    probe_result: ProbeResult,
    test_activations: np.ndarray,
    test_labels: List[str],
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification threshold for a probe.
    
    Args:
        probe_result: Trained probe results
        test_activations: Test activation array
        test_labels: Test labels
        metric: Metric to optimize ("f1", "accuracy", "precision", "recall")
        
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    binary_labels = convert_labels_to_binary(test_labels)
    
    # Get similarities
    if probe_result.best_layer is not None:
        layer = probe_result.best_layer
        similarities = test_activations[:, layer, :] @ probe_result.vectors[layer]
    else:
        # Multi-layer: average similarities
        total_similarities = np.zeros(len(test_labels))
        for layer, vector in probe_result.vectors.items():
            layer_acts = test_activations[:, layer, :]
            total_similarities += layer_acts @ vector
        similarities = total_similarities / len(probe_result.vectors)
    
    # Try different thresholds
    thresholds = np.linspace(
        np.min(similarities), np.max(similarities), 100
    )
    
    best_score = -1
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        
        if metric == "accuracy":
            score = accuracy_score(binary_labels, predictions)
        elif metric == "f1":
            _, _, score, _ = precision_recall_fscore_support(
                binary_labels, predictions, average='binary', zero_division=0
            )
        elif metric == "precision":
            score, _, _, _ = precision_recall_fscore_support(
                binary_labels, predictions, average='binary', zero_division=0
            )
        elif metric == "recall":
            _, score, _, _ = precision_recall_fscore_support(
                binary_labels, predictions, average='binary', zero_division=0
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def get_layer_ranking(
    probe_result: ProbeResult,
    sort_by: str = "score"
) -> List[Tuple[int, float]]:
    """
    Get layers ranked by performance.
    
    Args:
        probe_result: Trained probe results
        sort_by: Ranking criterion ("score" or "norm")
        
    Returns:
        List of (layer_index, value) tuples sorted by ranking
    """
    if sort_by == "score":
        items = list(probe_result.scores.items())
        items.sort(key=lambda x: x[1], reverse=True)
    elif sort_by == "norm":
        # Sort by vector norm
        items = []
        for layer, vector in probe_result.vectors.items():
            norm = np.linalg.norm(vector)
            items.append((layer, norm))
        items.sort(key=lambda x: x[1], reverse=True)
    else:
        raise ValueError(f"Unknown sort criterion: {sort_by}")
    
    return items


def analyze_probe_vectors(
    probe_result: ProbeResult
) -> Dict[str, any]:
    """
    Analyze properties of probe vectors.
    
    Args:
        probe_result: Trained probe results
        
    Returns:
        Dictionary with vector analysis results
    """
    vectors = probe_result.vectors
    
    analysis = {
        "num_layers": len(vectors),
        "vector_dims": list(vectors.values())[0].shape[0] if vectors else 0,
        "method": probe_result.method
    }
    
    if vectors:
        # Compute vector norms
        norms = {layer: np.linalg.norm(vec) for layer, vec in vectors.items()}
        analysis["norms"] = norms
        analysis["mean_norm"] = np.mean(list(norms.values()))
        analysis["std_norm"] = np.std(list(norms.values()))
        
        # Find most/least active dimensions across all vectors
        all_vectors = np.array(list(vectors.values()))
        dim_activity = np.mean(np.abs(all_vectors), axis=0)
        
        analysis["most_active_dims"] = np.argsort(dim_activity)[-10:].tolist()
        analysis["least_active_dims"] = np.argsort(dim_activity)[:10].tolist()
        
        # Vector similarities between adjacent layers
        if len(vectors) > 1:
            layers = sorted(vectors.keys())
            similarities = []
            for i in range(len(layers) - 1):
                vec1 = vectors[layers[i]]
                vec2 = vectors[layers[i + 1]]
                sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                similarities.append(sim)
            
            analysis["adjacent_similarities"] = similarities
            analysis["mean_adjacent_similarity"] = np.mean(similarities)
    
    return analysis