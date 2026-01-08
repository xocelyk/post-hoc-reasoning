"""
Results wrapper for post-hoc reasoning experiments.

This module provides a unified interface to load and analyze all experiment results:
- Steering results (generation with different alpha values)
- Generation results (train/test set performance)
- Probe results (AUC scores and coefficients)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class SteeringResults:
    """
    Handles steering experiment results (generation with different alpha values).
    
    Loads all steering results into a pandas DataFrame where each row
    represents a single generation/sample.
    """
    
    def __init__(self, cache_path: Union[str, Path] = None, verbose: bool = True):
        """
        Initialize ResultsData from a cache directory.
        
        Args:
            cache_path: Path to cache directory containing experiments
            verbose: Whether to print loading progress
        """
        self.verbose = verbose
        self.df = pd.DataFrame()
        
        if cache_path is not None:
            self.cache_path = Path(cache_path)
            if not self.cache_path.exists():
                raise ValueError(f"Cache path does not exist: {cache_path}")
            self._load_experiments()
        else:
            self.cache_path = None
    
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
                n_experiments = self.df.groupby(['model', 'dataset', 'experiment_hash']).ngroups if not self.df.empty else 0
                print(f"Loaded {len(self.df)} samples from {n_experiments} experiments")
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
                    'Train_Acc': '',  # Can be filled from generation results if needed
                    'Test_Acc': '',   # Can be filled from generation results if needed
                    'Probe_AUC': '',  # Can be filled from probe results if needed
                    'Best_Layer': ''  # Can be filled from probe results if needed
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


class GenerationResults:
    """
    Handles train/test generation results.
    
    Loads generation data into a DataFrame where each row represents
    a single generation from the train or test set.
    """
    
    def __init__(self, cache_path: Union[str, Path] = None, verbose: bool = True):
        """Initialize GenerationResults from a cache directory."""
        self.verbose = verbose
        self.df = pd.DataFrame()
        
        if cache_path is not None:
            self.cache_path = Path(cache_path)
            if not self.cache_path.exists():
                raise ValueError(f"Cache path does not exist: {cache_path}")
            self._load_generations()
        else:
            self.cache_path = None
    
    def _load_generations(self):
        """Load all generation data from cache directory."""
        all_records = []
        experiments_dir = self.cache_path / "experiments"
        
        if not experiments_dir.exists():
            experiments_dir = self.cache_path
        
        if self.verbose:
            print(f"Loading generation data from {experiments_dir}")
        
        # Iterate through experiments
        for model_dir in experiments_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                
                for split_dir in dataset_dir.iterdir():
                    if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
                        continue
                    
                    for hash_dir in split_dir.iterdir():
                        if not hash_dir.is_dir() or len(hash_dir.name) < 10:
                            continue
                        
                        experiment_hash = hash_dir.name
                        data_dir = hash_dir / "data"
                        
                        if not data_dir.exists():
                            continue
                        
                        # Load train and test generations
                        for split in ['train', 'test']:
                            gen_file = data_dir / f"{split}_generations.pkl"
                            if gen_file.exists():
                                try:
                                    with open(gen_file, 'rb') as f:
                                        generations = pickle.load(f)
                                    
                                    for idx, gen in enumerate(generations):
                                        record = {
                                            'model': model_name,
                                            'dataset': dataset_name,
                                            'split': split,
                                            'sample_idx': idx,
                                            'experiment_hash': experiment_hash,
                                            'prompt': gen.get('prompt', ''),
                                            'response': str(gen.get('response', '')),
                                            'correct_letter': gen.get('correct_letter', ''),
                                            'correct_answer': gen.get('correct_answer', ''),
                                            'pred_letter': gen.get('pred_letter', ''),
                                            'pred_answer': gen.get('pred_answer', ''),
                                            'is_correct': gen.get('pred_answer') == gen.get('correct_answer')
                                        }
                                        all_records.append(record)
                                except Exception as e:
                                    if self.verbose:
                                        print(f"Error loading {gen_file}: {e}")
        
        if all_records:
            self.df = pd.DataFrame(all_records)
            if self.verbose:
                print(f"Loaded {len(self.df)} generation samples")
    
    def get_accuracy(self, **filters) -> float:
        """Calculate accuracy for filtered data."""
        df = self.filter(**filters)
        if df.empty:
            return 0.0
        return (df['is_correct'].sum() / len(df)) * 100
    
    def filter(self, **kwargs) -> pd.DataFrame:
        """Filter the DataFrame by any column values."""
        df = self.df.copy()
        
        for col, value in kwargs.items():
            if col not in df.columns:
                continue
            
            if isinstance(value, (list, tuple, range)):
                df = df[df[col].isin(value)]
            else:
                df = df[df[col] == value]
        
        return df
    
    def get_errors(self, **filters) -> pd.DataFrame:
        """Get incorrect predictions for analysis."""
        df = self.filter(**filters)
        return df[df['is_correct'] == False]


class ProbeResults:
    """
    Handles probe training results.
    
    Stores AUC scores and coefficients for each layer of each model/dataset.
    """
    
    def __init__(self, cache_path: Union[str, Path] = None, verbose: bool = True):
        """Initialize ProbeResults from a cache directory."""
        self.verbose = verbose
        self.df = pd.DataFrame()
        self.coefficients = {}  # Store coefficients separately due to size
        
        if cache_path is not None:
            self.cache_path = Path(cache_path)
            if not self.cache_path.exists():
                raise ValueError(f"Cache path does not exist: {cache_path}")
            self._load_probes()
        else:
            self.cache_path = None
    
    def _load_probes(self):
        """Load all probe data from cache directory."""
        all_records = []
        experiments_dir = self.cache_path / "experiments"
        
        if not experiments_dir.exists():
            experiments_dir = self.cache_path
        
        if self.verbose:
            print(f"Loading probe data from {experiments_dir}")
        
        # Iterate through experiments
        for model_dir in experiments_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                
                for split_dir in dataset_dir.iterdir():
                    if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
                        continue
                    
                    for hash_dir in split_dir.iterdir():
                        if not hash_dir.is_dir() or len(hash_dir.name) < 10:
                            continue
                        
                        experiment_hash = hash_dir.name
                        probe_dir = hash_dir / "probes"
                        
                        if not probe_dir.exists():
                            continue
                        
                        # Load AUC scores
                        auc_file = probe_dir / "auc_scores.json"
                        if auc_file.exists():
                            try:
                                with open(auc_file, 'r') as f:
                                    auc_scores = json.load(f)
                                
                                if isinstance(auc_scores, list):
                                    for layer_idx, auc in enumerate(auc_scores):
                                        record = {
                                            'model': model_name,
                                            'dataset': dataset_name,
                                            'layer': layer_idx,
                                            'auc_score': auc,
                                            'experiment_hash': experiment_hash
                                        }
                                        all_records.append(record)
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error loading {auc_file}: {e}")
                        
                        # Load coefficients
                        coef_file = probe_dir / "coefficients.pkl"
                        if coef_file.exists():
                            try:
                                with open(coef_file, 'rb') as f:
                                    coefficients = pickle.load(f)
                                
                                coef_key = f"{model_name}/{dataset_name}/{experiment_hash}"
                                self.coefficients[coef_key] = coefficients
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error loading {coef_file}: {e}")
        
        if all_records:
            self.df = pd.DataFrame(all_records)
            if self.verbose:
                print(f"Loaded {len(self.df)} probe layer results")
    
    def get_best_layer(self, model: str, dataset: str) -> Tuple[int, float]:
        """Get the best performing layer for a model/dataset."""
        df = self.df[(self.df['model'] == model) & (self.df['dataset'] == dataset)]
        if df.empty:
            return None, None
        
        best_row = df.loc[df['auc_score'].idxmax()]
        return int(best_row['layer']), float(best_row['auc_score'])
    
    def get_auc_curve(self, model: str, dataset: str) -> pd.DataFrame:
        """Get AUC scores for all layers."""
        df = self.df[(self.df['model'] == model) & (self.df['dataset'] == dataset)]
        return df[['layer', 'auc_score']].sort_values('layer')
    
    def get_coefficients(self, model: str, dataset: str, layer: Optional[int] = None) -> Optional[np.ndarray]:
        """Get probe coefficients for a specific layer."""
        # Find the right experiment
        for key in self.coefficients:
            if key.startswith(f"{model}/{dataset}"):
                coef_dict = self.coefficients[key]
                if layer is not None and layer in coef_dict:
                    return coef_dict[layer]
                elif layer is None:
                    # Return best layer coefficients
                    best_layer, _ = self.get_best_layer(model, dataset)
                    if best_layer is not None and best_layer in coef_dict:
                        return coef_dict[best_layer]
        return None


class DebiasingResults:
    """
    Handles ACE debiasing experiment results.
    
    Loads all debiasing results into a pandas DataFrame where each row
    represents a single experiment.
    """
    
    def __init__(self, cache_path: Union[str, Path] = None, verbose: bool = True):
        """
        Initialize DebiasingResults from a cache directory.
        
        Args:
            cache_path: Path to cache directory containing experiments
            verbose: Whether to print loading progress
        """
        self.verbose = verbose
        self.df = pd.DataFrame()
        
        if cache_path is not None:
            self.cache_path = Path(cache_path)
            if not self.cache_path.exists():
                raise ValueError(f"Cache path does not exist: {cache_path}")
            self._load_experiments()
        else:
            self.cache_path = None
    
    def _load_experiments(self):
        """Load all debiasing experiments from cache directory into DataFrame."""
        all_records = []
        experiments_dir = self.cache_path / "experiments"
        
        if not experiments_dir.exists():
            experiments_dir = self.cache_path
        
        if self.verbose:
            print(f"Loading debiasing experiments from {experiments_dir}")
        
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
                        debiasing_dir = hash_dir / "debiasing"
                        
                        if not debiasing_dir.exists():
                            continue
                        
                        # Load debiasing results
                        results_file = debiasing_dir / "debiasing_alpha_0.0.pkl"
                        summary_file = debiasing_dir / "summary.json"
                        
                        if results_file.exists() and summary_file.exists():
                            try:
                                # Load main results
                                with open(results_file, 'rb') as f:
                                    results_data = pickle.load(f)
                                
                                # Load summary
                                with open(summary_file, 'r') as f:
                                    summary_data = json.load(f)
                                
                                record = {
                                    'model': model_name,
                                    'dataset': dataset_name,
                                    'experiment_hash': experiment_hash,
                                    'split_info': split_info,
                                    'layer': summary_data.get('layer'),
                                    'ace_bias': summary_data.get('ace_bias'),
                                    'ace_direction_norm': summary_data.get('ace_direction_norm'),
                                    'original_accuracy': summary_data.get('original_accuracy'),
                                    'debiased_accuracy': summary_data.get('debiased_accuracy'),
                                    'accuracy_change': summary_data.get('accuracy_change'),
                                    'prediction_change_rate': summary_data.get('prediction_change_rate'),
                                    'n_train_samples': results_data.get('n_train_samples'),
                                    'n_test_samples': results_data.get('n_test_samples'),
                                }
                                all_records.append(record)
                                
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error loading debiasing results from {results_file}: {e}")
        
        # Create DataFrame from all records
        if all_records:
            self.df = pd.DataFrame(all_records)
            if self.verbose:
                n_experiments = len(self.df)
                print(f"Loaded {n_experiments} debiasing experiments")
        else:
            if self.verbose:
                print("No debiasing results found")
    
    def get_intervention_magnitude(self, model: str = None, dataset: str = None) -> Optional[float]:
        """Get mean intervention magnitude for specified model/dataset."""
        df_filtered = self.df
        
        if model:
            df_filtered = df_filtered[df_filtered['model'] == model]
        if dataset:
            df_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        if df_filtered.empty:
            return None
        
        return df_filtered['mean_intervention_magnitude'].iloc[0]
    
    def get_debiasing_layer(self, model: str = None, dataset: str = None) -> Optional[int]:
        """Get the layer used for debiasing for specified model/dataset."""
        df_filtered = self.df
        
        if model:
            df_filtered = df_filtered[df_filtered['model'] == model]
        if dataset:
            df_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        if df_filtered.empty:
            return None
        
        return df_filtered['layer'].iloc[0]
    
    def get_ace_bias(self, model: str = None, dataset: str = None) -> Optional[float]:
        """Get ACE bias value for specified model/dataset."""
        df_filtered = self.df
        
        if model:
            df_filtered = df_filtered[df_filtered['model'] == model]
        if dataset:
            df_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        if df_filtered.empty:
            return None
        
        return df_filtered['ace_bias'].iloc[0]
    
    def get_accuracy_metrics(self, model: str = None, dataset: str = None) -> Optional[Dict[str, float]]:
        """Get accuracy metrics for specified model/dataset."""
        df_filtered = self.df
        
        if model:
            df_filtered = df_filtered[df_filtered['model'] == model]
        if dataset:
            df_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        if df_filtered.empty:
            return None
        
        row = df_filtered.iloc[0]
        return {
            "original_accuracy": row.get('original_accuracy'),
            "debiased_accuracy": row.get('debiased_accuracy'),
            "accuracy_change": row.get('accuracy_change'),
        }
    
    def get_debiased_accuracy(self, model: str = None, dataset: str = None) -> Optional[float]:
        """Get debiased accuracy for specified model/dataset."""
        df_filtered = self.df
        
        if model:
            df_filtered = df_filtered[df_filtered['model'] == model]
        if dataset:
            df_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        if df_filtered.empty:
            return None
        
        return df_filtered['debiased_accuracy'].iloc[0]
    
    def get_accuracy_change(self, model: str = None, dataset: str = None) -> Optional[float]:
        """Get accuracy change for specified model/dataset."""
        df_filtered = self.df
        
        if model:
            df_filtered = df_filtered[df_filtered['model'] == model]
        if dataset:
            df_filtered = df_filtered[df_filtered['dataset'] == dataset]
        
        if df_filtered.empty:
            return None
        
        return df_filtered['accuracy_change'].iloc[0]


class Results:
    """
    Main container class for all experiment results.
    
    Provides unified access to steering, generation, and probe results.
    """
    
    def __init__(self, cache_path: Union[str, Path], verbose: bool = True):
        """
        Initialize Results from a cache directory.
        
        Args:
            cache_path: Path to cache directory containing experiments
            verbose: Whether to print loading progress
        """
        self.cache_path = Path(cache_path)
        self.verbose = verbose
        
        if not self.cache_path.exists():
            raise ValueError(f"Cache path does not exist: {cache_path}")
        
        if verbose:
            print(f"Loading results from {cache_path}")
            print("=" * 60)
        
        # Load each type of results
        self.steering = SteeringResults(cache_path, verbose)
        self.generation = GenerationResults(cache_path, verbose)
        self.probe = ProbeResults(cache_path, verbose)
        self.debiasing = DebiasingResults(cache_path, verbose)
        
        if verbose:
            print("=" * 60)
            print(f"Loaded:")
            print(f"  - {len(self.steering.df)} steering samples")
            print(f"  - {len(self.generation.df)} generation samples")
            print(f"  - {len(self.probe.df)} probe layer results")
            print(f"  - {len(self.debiasing.df)} debiasing experiments")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all results."""
        summary = {
            'models': sorted(set(
                list(self.steering.df['model'].unique() if not self.steering.df.empty else []) +
                list(self.generation.df['model'].unique() if not self.generation.df.empty else [])
            )),
            'datasets': sorted(set(
                list(self.steering.df['dataset'].unique() if not self.steering.df.empty else []) +
                list(self.generation.df['dataset'].unique() if not self.generation.df.empty else [])
            )),
            'steering_samples': len(self.steering.df),
            'generation_samples': len(self.generation.df),
            'probe_results': len(self.probe.df),
            'debiasing_experiments': len(self.debiasing.df)
        }
        
        # Add accuracy summary
        if not self.generation.df.empty:
            accuracy_summary = []
            for model in summary['models']:
                for dataset in summary['datasets']:
                    train_acc = self.generation.get_accuracy(model=model, dataset=dataset, split='train')
                    test_acc = self.generation.get_accuracy(model=model, dataset=dataset, split='test')
                    if train_acc > 0 or test_acc > 0:
                        accuracy_summary.append({
                            'model': model,
                            'dataset': dataset,
                            'train_accuracy': train_acc,
                            'test_accuracy': test_acc
                        })
            summary['accuracy_summary'] = pd.DataFrame(accuracy_summary)
        
        # Add probe summary
        if not self.probe.df.empty:
            probe_summary = []
            for model in summary['models']:
                for dataset in summary['datasets']:
                    best_layer, best_auc = self.probe.get_best_layer(model, dataset)
                    if best_layer is not None:
                        probe_summary.append({
                            'model': model,
                            'dataset': dataset,
                            'best_layer': best_layer,
                            'best_auc': best_auc
                        })
            summary['probe_summary'] = pd.DataFrame(probe_summary)
        
        return summary
    
    def export_all(self, output_dir: Union[str, Path]):
        """Export all results to separate CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export steering results
        if not self.steering.df.empty:
            self.steering.to_csv(output_dir / "steering_results.csv")
        
        # Export generation results
        if not self.generation.df.empty:
            self.generation.df.to_csv(output_dir / "generation_results.csv", index=False)
        
        # Export probe results
        if not self.probe.df.empty:
            self.probe.df.to_csv(output_dir / "probe_results.csv", index=False)
        
        # Export debiasing results
        if not self.debiasing.df.empty:
            self.debiasing.df.to_csv(output_dir / "debiasing_results.csv", index=False)
        
        if self.verbose:
            print(f"Exported all results to {output_dir}")
    
    def __repr__(self):
        return (f"Results(steering={len(self.steering.df)} samples, "
                f"generation={len(self.generation.df)} samples, "
                f"probe={len(self.probe.df)} results, "
                f"debiasing={len(self.debiasing.df)} experiments)")


class ResultsAggregator:
    """Helper class for complex aggregations across experiments."""
    
    @staticmethod
    def compare_models(results: Results, dataset: str, metric: str = 'success_rate') -> pd.DataFrame:
        """Compare models on a specific dataset."""
        stats = []
        
        for model in results.steering.df['model'].unique():
            for direction in ['yes', 'no']:
                for alpha in results.steering.df['alpha'].unique():
                    filtered = results.steering.filter(model=model, dataset=dataset, direction=direction, alpha=alpha)
                    
                    if not filtered.empty:
                        if metric == 'success_rate':
                            value = results.steering.get_success_rate(model=model, dataset=dataset, 
                                                            direction=direction, alpha=alpha)
                        elif metric == 'parse_rate':
                            value = results.steering.get_parse_rate(model=model, dataset=dataset,
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
    def find_best_alpha(results: Results, model: str, dataset: str, direction: str) -> Dict:
        """Find the alpha value with best success rate."""
        best_alpha = None
        best_rate = 0
        
        for alpha in results.steering.df['alpha'].unique():
            rate = results.steering.get_success_rate(model=model, dataset=dataset, 
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
    
    # Load all results
    print(f"Loading all results from {cache_path}...")
    results = Results(cache_path)
    
    # Show summary
    print(f"\n{results}")
    
    # Get comprehensive summary
    summary = results.get_summary()
    print("\nModels:", summary['models'])
    print("Datasets:", summary['datasets'])
    
    # Show accuracy summary if available
    if 'accuracy_summary' in summary and not summary['accuracy_summary'].empty:
        print("\nAccuracy Summary:")
        print(summary['accuracy_summary'].head())
    
    # Show probe summary if available
    if 'probe_summary' in summary and not summary['probe_summary'].empty:
        print("\nProbe Summary:")
        print(summary['probe_summary'].head())
    
    # Show steering summary statistics
    if not results.steering.df.empty:
        print("\nSteering Summary Statistics:")
        print(results.steering.get_summary_stats().head(10))
    
    # Example: Get success rate for specific combination
    if not results.steering.df.empty:
        sample_model = results.steering.df['model'].iloc[0]
        sample_dataset = results.steering.df['dataset'].iloc[0]
        
        rate = results.steering.get_success_rate(model=sample_model, dataset=sample_dataset, 
                                                 direction='yes', alpha=2)
        print(f"\nSteering success rate for {sample_model} on {sample_dataset} (yes, α=2): {rate:.1f}%")
    
    # Example: Get generation accuracy
    if not results.generation.df.empty:
        sample_model = results.generation.df['model'].iloc[0]
        sample_dataset = results.generation.df['dataset'].iloc[0]
        
        train_acc = results.generation.get_accuracy(model=sample_model, dataset=sample_dataset, split='train')
        test_acc = results.generation.get_accuracy(model=sample_model, dataset=sample_dataset, split='test')
        print(f"\nGeneration accuracy for {sample_model} on {sample_dataset}:")
        print(f"  Train: {train_acc:.1f}%")
        print(f"  Test: {test_acc:.1f}%")
    
    # Example: Get best probe layer
    if not results.probe.df.empty:
        sample_model = results.probe.df['model'].iloc[0]
        sample_dataset = results.probe.df['dataset'].iloc[0]
        
        best_layer, best_auc = results.probe.get_best_layer(sample_model, sample_dataset)
        print(f"\nBest probe layer for {sample_model} on {sample_dataset}:")
        print(f"  Layer {best_layer} with AUC={best_auc:.3f}")