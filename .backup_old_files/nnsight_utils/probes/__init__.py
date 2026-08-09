"""
Probe training and evaluation utilities.
"""

from .base import BaseProbe, ProbeResult, compute_similarity_score, convert_labels_to_binary
from .contrastive import (
    CAASingleLayerProbe,
    CAAMultiLayerProbe,
    compute_incremental_vectors,
    compute_mean_differences,
    evaluate_mean_differences,
    find_best_layer,
    train_caa_multi_layer,
    train_caa_single_layer,
)
from .evaluation import (
    analyze_probe_vectors,
    compare_probe_methods,
    evaluate_probe_performance,
    find_optimal_threshold,
    get_layer_ranking,
)
from .logistic import (
    LogisticRegressionProbe,
    evaluate_logistic_probes,
    extract_steering_vectors,
    train_logistic_probe,
    train_logistic_probes,
)

__all__ = [
    # Base classes
    "BaseProbe",
    "ProbeResult",
    "compute_similarity_score",
    "convert_labels_to_binary",
    # Contrastive methods
    "CAASingleLayerProbe",
    "CAAMultiLayerProbe",
    "compute_mean_differences",
    "evaluate_mean_differences",
    "find_best_layer",
    "compute_incremental_vectors",
    "train_caa_single_layer",
    "train_caa_multi_layer",
    # Logistic methods
    "LogisticRegressionProbe",
    "train_logistic_probes",
    "extract_steering_vectors",
    "evaluate_logistic_probes",
    "train_logistic_probe",
    # Evaluation
    "evaluate_probe_performance",
    "compare_probe_methods",
    "find_optimal_threshold",
    "get_layer_ranking",
    "analyze_probe_vectors",
]