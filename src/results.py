"""
Results wrapper for post-hoc reasoning experiments.

This module provides a unified interface to load and analyze steering experiment results
from cache directories. The main class ResultsData loads all results into a pandas DataFrame
for easy filtering, aggregation, and analysis.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from collections import defaultdict


class ExperimentMetadata:
    """Store metadata for an experiment (probe scores, accuracies, etc.)"""
    
    def __init__(self, experiment_path: Path):
        self.path = experiment_path
        self.probe_auc = None
        self.best_layer = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.config = {}
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from experiment directory."""
        # Load probe AUC scores
        auc_file = self.path / "probes" / "auc_scores.json"
        if auc_file.exists():
            with open(auc_file, 'r') as f:
                auc_scores = json.load(f)
                if isinstance(auc_scores, list):
                    self.probe_auc = max(auc_scores) if auc_scores else None
                    self.best_layer = auc_scores.index(max(auc_scores)) if auc_scores else None
        
        # Load generation accuracies
        for split in ['train', 'test']:
            gen_file = self.path / "data" / f"{split}_generations.pkl"
            if gen_file.exists():
                with open(gen_file, 'rb') as f:
                    generations = pickle.load(f)
                    if isinstance(generations, list) and generations:
                        correct = sum(1 for g in generations 
                                    if g.get('pred_answer') == g.get('correct_answer'))
                        accuracy = correct / len(generations) if generations else 0
                        setattr(self, f"{split}_accuracy", accuracy)


class ResultsData:
    """
    Main class for loading and analyzing steering experiment results.
    
    Loads all steering results into a pandas DataFrame where each row
    represents a single generation/sample.
    """
    
    def __init__(self, cache_path: Union[str, Path], verbose: bool = True):
        """
        Initialize ResultsData from a cache directory.
        
        Args:
            cache_path: Path to cache directory containing experiments
            verbose: Whether to print loading progress
        """
        self.cache_path = Path(cache_path)
        self.verbose = verbose
        self.df = pd.DataFrame()
        self.metadata = {}
        
        if not self.cache_path.exists():
            raise ValueError(f"Cache path does not exist: {cache_path}")
        
        self._load_experiments()
    
    def _load_experiments(self):
        """Load all experiments from cache directory into DataFrame."""
        all_records = []
        experiments_dir = self.cache_path / "experiments"
        
        if not experiments_dir.exists():
            experiments_dir = self.cache_path
        
        if self.verbose:
            print(f"Loading experiments from {experiments_dir}")
        
        # Iterate through model directories
        for model_dir in experiments_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            # Iterate through dataset directories
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                
                # Find experiment directories (split_*/hash)
                for split_dir in dataset_dir.iterdir():
                    if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
                        continue
                    
                    split_info = split_dir.name  # e.g., "split_42_500_500"
                    
                    for hash_dir in split_dir.iterdir():
                        if not hash_dir.is_dir() or len(hash_dir.name) < 10:
                            continue
                        
                        experiment_hash = hash_dir.name
                        steering_dir = hash_dir / "steering"
                        
                        if not steering_dir.exists():
                            continue
                        
                        # Load metadata for this experiment
                        exp_key = f"{model_name}/{dataset_name}/{experiment_hash}"
                        self.metadata[exp_key] = ExperimentMetadata(hash_dir)
                        
                        # Load all steering result files
                        for pkl_file in steering_dir.glob("*.pkl"):
                            records = self._load_steering_file(
                                pkl_file, model_name, dataset_name, 
                                experiment_hash, split_info
                            )
                            all_records.extend(records)
        
        # Create DataFrame from all records
        if all_records:
            self.df = pd.DataFrame(all_records)
            if self.verbose:
                print(f"Loaded {len(self.df)} samples from {len(self.metadata)} experiments")
                print(f"Models: {self.df['model'].nunique()}")
                print(f"Datasets: {self.df['dataset'].nunique()}")
        else:
            if self.verbose:
                print("No steering results found in cache directory")
    
    def _load_steering_file(self, pkl_file: Path, model: str, dataset: str, 
                           experiment_hash: str, split_info: str) -> List[Dict]:
        """Load a single steering result pickle file."""
        records = []
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list):
                return records
            
            # Parse filename to get alpha and direction info
            filename = pkl_file.stem
            parts = filename.split("_")
            
            if "alpha" not in filename:
                return records
            
            # Extract alpha value
            alpha_idx = parts.index("alpha") + 1
            alpha = float(parts[alpha_idx])
            
            # Determine direction based on filename and alpha
            # For alpha=0, suffix indicates ORIGINAL answer
            if alpha == 0:
                if "yes" in filename:
                    direction = "no"  # yes→no steering
                elif "no" in filename:
                    direction = "yes"  # no→yes steering
                else:
                    return records
            else:
                # For non-zero alpha, positive = no→yes, negative = yes→no
                direction = "yes" if alpha > 0 else "no"
            
            # Create a record for each sample in the file
            for idx, sample in enumerate(data):
                record = {
                    'model': model,
                    'dataset': dataset,
                    'direction': direction,
                    'alpha': abs(alpha),
                    'sample_idx': idx,
                    'experiment_hash': experiment_hash,
                    'split_info': split_info,
                    'filename': filename,
                }
                
                # Add all fields from the sample
                record.update(sample)
                
                records.append(record)
        
        except Exception as e:
            if self.verbose:
                print(f"Error loading {pkl_file}: {e}")
        
        return records
    
    def filter(self, **kwargs) -> pd.DataFrame:
        """
        Filter the DataFrame by any column values.
        
        Args:
            **kwargs: Column name and value pairs to filter by
            
        Returns:
            Filtered DataFrame
        
        Examples:
            results.filter(model='gemma-2b', alpha=2)
            results.filter(dataset='sports', direction='yes')
        """
        df = self.df.copy()
        
        for col, value in kwargs.items():
            if col not in df.columns:
                print(f"Warning: column '{col}' not found in DataFrame")
                continue
            
            if isinstance(value, (list, tuple, range)):
                df = df[df[col].isin(value)]
            else:
                df = df[df[col] == value]
        
        return df
    
    def get_success_rate(self, **filters) -> float:
        """
        Calculate success rate for filtered data.
        
        Success rate is calculated as: successful steerings / valid parses
        
        Args:
            **filters: Filtering criteria
            
        Returns:
            Success rate as percentage (0-100)
        """
        df = self.filter(**filters)
        
        if df.empty:
            return 0.0
        
        valid_df = df[df['is_valid_parse'] == True]
        if valid_df.empty:
            return 0.0
        
        success_count = valid_df['success'].sum()
        return (success_count / len(valid_df)) * 100
    
    def get_parse_rate(self, **filters) -> float:
        """
        Calculate parse rate for filtered data.
        
        Args:
            **filters: Filtering criteria
            
        Returns:
            Parse rate as percentage (0-100)
        """
        df = self.filter(**filters)
        
        if df.empty:
            return 0.0
        
        valid_count = df['is_valid_parse'].sum()
        return (valid_count / len(df)) * 100
    
    def get_generations(self, **filters) -> List[str]:
        """
        Get raw generation texts for filtered data.
        
        Args:
            **filters: Filtering criteria
            
        Returns:
            List of generation strings
        """
        df = self.filter(**filters)
        return df['steered_generation'].tolist() if 'steered_generation' in df.columns else []
    
    def get_summary_stats(self, groupby: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary statistics for the data.
        
        Args:
            groupby: Columns to group by (default: ['model', 'dataset', 'direction', 'alpha'])
            
        Returns:
            DataFrame with summary statistics
        """
        if groupby is None:
            groupby = ['model', 'dataset', 'direction', 'alpha']
        
        if self.df.empty:
            return pd.DataFrame()
        
        # Group and calculate statistics
        grouped = self.df.groupby(groupby)
        
        stats = grouped.agg({
            'sample_idx': 'count',  # Number of samples
            'is_valid_parse': lambda x: x.sum() / len(x) * 100,  # Parse rate
            'success': lambda x: x[self.df.loc[x.index, 'is_valid_parse']].sum() / 
                                 self.df.loc[x.index, 'is_valid_parse'].sum() * 100
                                 if self.df.loc[x.index, 'is_valid_parse'].sum() > 0 else 0  # Success rate
        }).rename(columns={
            'sample_idx': 'n_samples',
            'is_valid_parse': 'parse_rate',
            'success': 'success_rate'
        })
        
        return stats.round(2)
    
    def to_csv(self, path: Union[str, Path], **filters):
        """
        Export filtered data to CSV.
        
        Args:
            path: Output file path
            **filters: Filtering criteria
        """
        df = self.filter(**filters) if filters else self.df
        df.to_csv(path, index=False)
        if self.verbose:
            print(f"Saved {len(df)} rows to {path}")
    
    def to_template_format(self, path: Union[str, Path], min_samples: int = 50):
        """
        Export data in template format matching Excel structure.
        
        Args:
            path: Output file path
            min_samples: Minimum samples required to include results
        """
        template_rows = []
        
        # Group by model and dataset
        for (model, dataset), group in self.df.groupby(['model', 'dataset']):
            # Get metadata
            exp_key = f"{model}/{dataset}"
            metadata = None
            for key in self.metadata:
                if key.startswith(exp_key):
                    metadata = self.metadata[key]
                    break
            
            # Process each direction
            for direction in ['yes', 'no']:
                dir_data = group[group['direction'] == direction]
                if dir_data.empty:
                    continue
                
                # Find max sample count
                alpha_counts = dir_data.groupby('alpha').size()
                max_n = alpha_counts.max() if not alpha_counts.empty else 0
                
                if max_n <= min_samples:
                    continue
                
                # Build row
                row = {
                    'Model': model,
                    'Dataset': dataset,
                    'Direction': direction.upper(),
                    'N': max_n,
                    'Train_Acc': metadata.train_accuracy * 100 if metadata and metadata.train_accuracy else '',
                    'Test_Acc': metadata.test_accuracy * 100 if metadata and metadata.test_accuracy else '',
                    'Probe_AUC': metadata.probe_auc if metadata else '',
                    'Best_Layer': metadata.best_layer if metadata else ''
                }
                
                # Add results for each alpha
                for alpha in range(21):
                    alpha_data = dir_data[dir_data['alpha'] == alpha]
                    
                    if len(alpha_data) == max_n:  # Only include complete data
                        parse_rate = self.get_parse_rate(
                            model=model, dataset=dataset, direction=direction, alpha=alpha
                        )
                        success_rate = self.get_success_rate(
                            model=model, dataset=dataset, direction=direction, alpha=alpha
                        )
                        
                        row[f'Alpha_{alpha}_Success'] = round(success_rate, 1)
                        row[f'Alpha_{alpha}_Unparsed'] = round(100 - parse_rate, 1)
                    else:
                        row[f'Alpha_{alpha}_Success'] = ''
                        row[f'Alpha_{alpha}_Unparsed'] = ''
                
                template_rows.append(row)
        
        # Create DataFrame and save
        template_df = pd.DataFrame(template_rows)
        template_df.to_csv(path, index=False)
        
        if self.verbose:
            print(f"Saved template format with {len(template_df)} rows to {path}")
    
    def __repr__(self):
        return f"ResultsData(samples={len(self.df)}, models={self.df['model'].nunique() if not self.df.empty else 0}, datasets={self.df['dataset'].nunique() if not self.df.empty else 0})"
    
    def __len__(self):
        return len(self.df)


class ResultsAggregator:
    """Helper class for complex aggregations across experiments."""
    
    @staticmethod
    def compare_models(results: ResultsData, dataset: str, metric: str = 'success_rate') -> pd.DataFrame:
        """Compare models on a specific dataset."""
        stats = []
        
        for model in results.df['model'].unique():
            for direction in ['yes', 'no']:
                for alpha in results.df['alpha'].unique():
                    filtered = results.filter(model=model, dataset=dataset, direction=direction, alpha=alpha)
                    
                    if not filtered.empty:
                        if metric == 'success_rate':
                            value = results.get_success_rate(model=model, dataset=dataset, 
                                                            direction=direction, alpha=alpha)
                        elif metric == 'parse_rate':
                            value = results.get_parse_rate(model=model, dataset=dataset,
                                                          direction=direction, alpha=alpha)
                        else:
                            value = None
                        
                        stats.append({
                            'model': model,
                            'direction': direction,
                            'alpha': alpha,
                            metric: value,
                            'n_samples': len(filtered)
                        })
        
        return pd.DataFrame(stats)
    
    @staticmethod
    def find_best_alpha(results: ResultsData, model: str, dataset: str, direction: str) -> Dict:
        """Find the alpha value with best success rate."""
        best_alpha = None
        best_rate = 0
        
        for alpha in results.df['alpha'].unique():
            rate = results.get_success_rate(model=model, dataset=dataset, 
                                           direction=direction, alpha=alpha)
            if rate > best_rate:
                best_rate = rate
                best_alpha = alpha
        
        return {'alpha': best_alpha, 'success_rate': best_rate}


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        cache_path = sys.argv[1]
    else:
        cache_path = "cache/experiments"
    
    # Load results
    print(f"Loading results from {cache_path}...")
    results = ResultsData(cache_path)
    
    # Show summary
    print(f"\n{results}")
    
    # Show summary statistics
    print("\nSummary statistics:")
    print(results.get_summary_stats().head(20))
    
    # Example: Get success rate for specific combination
    if not results.df.empty:
        sample_model = results.df['model'].iloc[0]
        sample_dataset = results.df['dataset'].iloc[0]
        
        rate = results.get_success_rate(model=sample_model, dataset=sample_dataset, 
                                       direction='yes', alpha=2)
        print(f"\nSuccess rate for {sample_model} on {sample_dataset} (yes, α=2): {rate:.1f}%")