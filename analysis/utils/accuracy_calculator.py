"""
Accuracy calculation and confusion matrix utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


class AccuracyCalculator:
    """Calculate accuracy metrics and confusion matrices."""
    
    @staticmethod
    def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate simple accuracy score."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        if not predictions or not ground_truth:
            return 0.0
        
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)
    
    @staticmethod
    def calculate_per_class_metrics(predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
        """Calculate precision, recall, F1 for each class."""
        if not predictions or not ground_truth:
            return {}
        
        # Get unique labels
        labels = sorted(list(set(ground_truth + predictions)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        
        # Calculate metrics for each class
        metrics = {}
        for i, label in enumerate(labels):
            if cm.sum() == 0:
                continue
                
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': cm[i, :].sum()
            }
        
        return metrics
    
    @staticmethod
    def create_confusion_matrix(predictions: List[str], ground_truth: List[str], 
                              title: str = "Confusion Matrix") -> Tuple[np.ndarray, List[str]]:
        """Create confusion matrix."""
        if not predictions or not ground_truth:
            return np.array([]), []
        
        labels = sorted(list(set(ground_truth + predictions)))
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        
        return cm, labels
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix",
                            normalize: bool = False, figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix using seaborn."""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return plt.gcf()


class ExperimentAnalyzer:
    """Analyze experiment results across models and datasets."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, model: str, dataset: str, split_type: str,
                   predictions: List[str], ground_truth: List[str], 
                   experiment_hash: str = None):
        """Add a result to the analysis."""
        if len(predictions) != len(ground_truth):
            print(f"Warning: Length mismatch for {model}-{dataset}-{split_type}")
            return
        
        accuracy = AccuracyCalculator.calculate_accuracy(predictions, ground_truth)
        per_class_metrics = AccuracyCalculator.calculate_per_class_metrics(predictions, ground_truth)
        cm, labels = AccuracyCalculator.create_confusion_matrix(predictions, ground_truth)
        
        result = {
            'model': model,
            'dataset': dataset,
            'split_type': split_type,
            'accuracy': accuracy,
            'num_samples': len(predictions),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'labels': labels,
            'experiment_hash': experiment_hash,
            'predictions': predictions,
            'ground_truth': ground_truth
        }
        
        self.results.append(result)
    
    def get_accuracy_summary(self) -> pd.DataFrame:
        """Get summary of accuracies across all experiments."""
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'model': result['model'],
                'dataset': result['dataset'],
                'split_type': result['split_type'],
                'accuracy': result['accuracy'],
                'num_samples': result['num_samples'],
                'experiment_hash': result['experiment_hash']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_model_dataset_matrix(self, split_type: str = 'test') -> pd.DataFrame:
        """Get accuracy matrix with models as rows and datasets as columns."""
        # Filter results for specific split type
        filtered_results = [r for r in self.results if r['split_type'] == split_type]
        
        if not filtered_results:
            return pd.DataFrame()
        
        # Create matrix
        models = sorted(set(r['model'] for r in filtered_results))
        datasets = sorted(set(r['dataset'] for r in filtered_results))
        
        matrix_data = []
        for model in models:
            row = {'model': model}
            for dataset in datasets:
                # Find matching result
                matching_results = [r for r in filtered_results 
                                  if r['model'] == model and r['dataset'] == dataset]
                if matching_results:
                    row[dataset] = matching_results[0]['accuracy']
                else:
                    row[dataset] = np.nan
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data)
        df.set_index('model', inplace=True)
        return df
    
    def plot_accuracy_heatmap(self, split_type: str = 'test', figsize: Tuple[int, int] = (12, 8)):
        """Plot accuracy heatmap for models vs datasets."""
        matrix_df = self.get_model_dataset_matrix(split_type)
        
        if matrix_df.empty:
            print(f"No data available for split_type: {split_type}")
            return None
        
        plt.figure(figsize=figsize)
        sns.heatmap(matrix_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
        plt.title(f'Model vs Dataset Accuracy ({split_type.title()} Set)')
        plt.xlabel('Dataset')
        plt.ylabel('Model')
        plt.tight_layout()
        return plt.gcf()
    
    def get_per_class_summary(self) -> pd.DataFrame:
        """Get per-class performance summary."""
        summary_data = []
        
        for result in self.results:
            model = result['model']
            dataset = result['dataset']
            split_type = result['split_type']
            
            for class_name, metrics in result['per_class_metrics'].items():
                summary_data.append({
                    'model': model,
                    'dataset': dataset,
                    'split_type': split_type,
                    'class': class_name,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'support': metrics['support']
                })
        
        return pd.DataFrame(summary_data)
    
    def export_results_to_csv(self, output_dir: str):
        """Export all results to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export accuracy summary
        accuracy_df = self.get_accuracy_summary()
        accuracy_df.to_csv(f"{output_dir}/accuracy_summary.csv", index=False)
        
        # Export per-class metrics
        per_class_df = self.get_per_class_summary()
        per_class_df.to_csv(f"{output_dir}/per_class_metrics.csv", index=False)
        
        # Export train and test matrices
        for split_type in ['train', 'test']:
            matrix_df = self.get_model_dataset_matrix(split_type)
            if not matrix_df.empty:
                matrix_df.to_csv(f"{output_dir}/accuracy_matrix_{split_type}.csv")
        
        # Export individual experiment results
        detailed_data = []
        for result in self.results:
            for pred, gt in zip(result['predictions'], result['ground_truth']):
                detailed_data.append({
                    'model': result['model'],
                    'dataset': result['dataset'],
                    'split_type': result['split_type'],
                    'experiment_hash': result['experiment_hash'],
                    'prediction': pred,
                    'ground_truth': gt,
                    'correct': pred == gt
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(f"{output_dir}/detailed_predictions.csv", index=False)
        
        print(f"Results exported to {output_dir}")