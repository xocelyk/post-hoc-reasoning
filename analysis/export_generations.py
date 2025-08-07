"""
Export all train/test generations to CSV files for analysis.
"""

import pandas as pd
import os
from pathlib import Path
import sys

# Add parent directory to path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

from analysis.utils.data_loader import CacheDataLoader, DatasetProcessor, ResponseParser


def export_generation_csvs(cache_dir: str = "cache", output_dir: str = "results/train_test_generations"):
    """Export all train/test generations to CSV files organized by model and dataset."""
    
    # Initialize loader
    loader = CacheDataLoader(cache_dir)
    experiments = loader.get_all_experiments()
    
    print(f"Found {len(experiments)} experiments")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful_exports = 0
    failed_exports = 0
    
    for exp in experiments:
        try:
            print(f"Processing {exp['model']} - {exp['dataset']}...")
            
            # Load experiment data
            train_gen, test_gen = loader.load_train_test_generations(exp['path'])
            dataset, split_info = loader.load_dataset_and_split(exp['path'])
            
            if train_gen is None or test_gen is None or dataset is None:
                print(f"  - Missing data files, skipping")
                failed_exports += 1
                continue
            
            # Extract ground truth labels
            questions, labels = DatasetProcessor.extract_labels_from_dataset(
                exp['dataset'], dataset, split_info
            )
            
            if not questions or not labels:
                print(f"  - Could not extract labels, skipping")
                failed_exports += 1
                continue
            
            # Split into train/test
            if split_info and isinstance(split_info, dict) and 'train_indices' in split_info:
                train_indices = split_info['train_indices']
                test_indices = split_info['test_indices']
                
                train_questions = [questions[i] for i in train_indices]
                train_labels = [labels[i] for i in train_indices]
                test_questions = [questions[i] for i in test_indices]
                test_labels = [labels[i] for i in test_indices]
            else:
                # Fallback approach
                train_size = len(train_gen) if isinstance(train_gen, list) else 0
                test_size = len(test_gen) if isinstance(test_gen, list) else 0
                
                if train_size + test_size > len(labels):
                    print(f"  - Size mismatch, skipping")
                    failed_exports += 1
                    continue
                
                train_questions = questions[:train_size]
                train_labels = labels[:train_size]
                test_questions = questions[train_size:train_size + test_size]
                test_labels = labels[train_size:train_size + test_size]
            
            # Extract predictions
            train_predictions = [
                ResponseParser.extract_predicted_label(gen, exp['dataset']) 
                for gen in train_gen
            ]
            
            test_predictions = [
                ResponseParser.extract_predicted_label(gen, exp['dataset']) 
                for gen in test_gen
            ]
            
            # Create output directory for this model/dataset
            model_dataset_dir = Path(output_dir) / exp['model'] / exp['dataset']
            model_dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Create train CSV
            if len(train_predictions) == len(train_labels) == len(train_questions) == len(train_gen):
                train_df = pd.DataFrame({
                    'question': train_questions,
                    'generation': train_gen,
                    'ground_truth': train_labels,
                    'predicted_label': train_predictions,
                    'correct': [p == gt for p, gt in zip(train_predictions, train_labels)],
                    'experiment_hash': exp['experiment_hash'],
                    'split': exp['split']
                })
                
                train_csv_path = model_dataset_dir / "train_generations.csv"
                train_df.to_csv(train_csv_path, index=False)
                print(f"  - Exported train: {len(train_df)} samples to {train_csv_path}")
            else:
                print(f"  - Train size mismatch: {len(train_predictions)} vs {len(train_labels)} vs {len(train_questions)} vs {len(train_gen)}")
            
            # Create test CSV
            if len(test_predictions) == len(test_labels) == len(test_questions) == len(test_gen):
                test_df = pd.DataFrame({
                    'question': test_questions,
                    'generation': test_gen,
                    'ground_truth': test_labels,
                    'predicted_label': test_predictions,
                    'correct': [p == gt for p, gt in zip(test_predictions, test_labels)],
                    'experiment_hash': exp['experiment_hash'],
                    'split': exp['split']
                })
                
                test_csv_path = model_dataset_dir / "test_generations.csv"
                test_df.to_csv(test_csv_path, index=False)
                print(f"  - Exported test: {len(test_df)} samples to {test_csv_path}")
            else:
                print(f"  - Test size mismatch: {len(test_predictions)} vs {len(test_labels)} vs {len(test_questions)} vs {len(test_gen)}")
            
            successful_exports += 1
            print(f"  - Success")
            
        except Exception as e:
            print(f"  - Error: {e}")
            failed_exports += 1
            continue
    
    print(f"\nExport complete: {successful_exports} successful, {failed_exports} failed")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    export_generation_csvs()