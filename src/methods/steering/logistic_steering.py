"""
Logistic Regression Steering implementation.

This method trains logistic regression classifiers for all layers and uses coefficient vectors.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .base import SteeringMethod


class LogisticRegressionSteering(SteeringMethod):
    """Logistic Regression Steering: Train classifiers for all layers and use all coefficient vectors."""
    
    def __init__(self, train_activations: List[List[np.ndarray]], train_labels: List[str],
                 test_activations: List[List[np.ndarray]], test_labels: List[str]):
        """Initialize with training and test data for classifier training.
        
        Args:
            train_activations: List of activation lists (one per sample, one per layer)
            train_labels: Training labels ("yes" or "no")
            test_activations: List of activation lists for test data
            test_labels: Test labels ("yes" or "no")
        """
        self.train_activations = train_activations
        self.train_labels = train_labels
        self.test_activations = test_activations
        self.test_labels = test_labels
        
    def compute_steering_vectors(self, layer_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Train logistic regression classifiers and extract coefficient vectors."""
        num_layers = len(self.train_activations[0])
        steering_vectors = []
        
        for layer_idx in range(num_layers):
            # Prepare data for this layer
            train_data = self._prepare_data_for_layer(
                self.train_activations, self.train_labels, layer_idx
            )
            test_data = self._prepare_data_for_layer(
                self.test_activations, self.test_labels, layer_idx
            )
            
            if len(train_data) == 0 or len(test_data) == 0:
                # If no data, use zero vector
                vector_dim = len(self.train_activations[0][layer_idx])
                steering_vectors.append(np.zeros(vector_dim))
                continue
            
            # Train logistic regression classifier
            clf = self._train_classifier(train_data)
            
            # Extract coefficient vector (unnormalized)
            coef_vector = clf.coef_[0]
            steering_vectors.append(coef_vector)
            
            # Evaluate for logging
            auc_score = self._evaluate_classifier(clf, test_data)
            print(f"Layer {layer_idx} Logistic Regression AUC: {auc_score:.4f}")
        
        return steering_vectors
    
    def _prepare_data_for_layer(self, activations: List[List[np.ndarray]], 
                               labels: List[str], layer_idx: int) -> pd.DataFrame:
        """Prepare data for training classifier at specific layer."""
        data = []
        for sample_idx, sample_activations in enumerate(activations):
            # Only use samples where prediction matches ground truth
            if labels[sample_idx] in ["yes", "no"]:
                activation = sample_activations[layer_idx]
                data.append(activation.tolist() + [labels[sample_idx]])
        
        if not data:
            return pd.DataFrame()
            
        vector_dim = len(data[0]) - 1
        columns = [f"ac{i}" for i in range(vector_dim)] + ["pred"]
        df = pd.DataFrame(data, columns=columns)
        df = df[df["pred"].isin(["yes", "no"])]
        return df
    
    def _train_classifier(self, train_data: pd.DataFrame) -> LogisticRegression:
        """Train logistic regression classifier."""
        X = train_data[[col for col in train_data.columns if col.startswith("ac")]]
        y = train_data["pred"]
        return LogisticRegression(random_state=0, max_iter=1000).fit(X, y)
    
    def _evaluate_classifier(self, clf: LogisticRegression, test_data: pd.DataFrame) -> float:
        """Evaluate classifier performance."""
        if len(test_data) == 0:
            return 0.0
            
        X = test_data[[col for col in test_data.columns if col.startswith("ac")]]
        y = test_data["pred"]
        y_binary = y.apply(lambda x: 1 if x == "yes" else 0)
        
        try:
            y_pred_proba = clf.predict_proba(X)[:, 1]
            return roc_auc_score(y_binary, y_pred_proba)
        except ValueError:
            return 0.5
    
    def get_method_name(self) -> str:
        return "logistic-regression"