"""
Utilities for loading and processing cached experiment data.
"""

import pickle
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re


class CacheDataLoader:
    """Loader for cached experiment data from the post-hoc reasoning experiments."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.experiments_dir = self.cache_dir / "experiments"
    
    def get_all_experiments(self) -> List[Dict[str, str]]:
        """Get list of all experiments with model, dataset, split info."""
        experiments = []
        
        if not self.experiments_dir.exists():
            return experiments
            
        for model_dir in self.experiments_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                    
                for split_dir in dataset_dir.iterdir():
                    if not split_dir.is_dir() or not split_dir.name.startswith("split_"):
                        continue
                        
                    for experiment_dir in split_dir.iterdir():
                        if not experiment_dir.is_dir():
                            continue
                            
                        # Check if this experiment has the required files
                        data_dir = experiment_dir / "data"
                        metadata_dir = experiment_dir / "metadata"
                        
                        if data_dir.exists() and metadata_dir.exists():
                            experiments.append({
                                "model": model_dir.name,
                                "dataset": dataset_dir.name,
                                "split": split_dir.name,
                                "experiment_hash": experiment_dir.name,
                                "path": str(experiment_dir)
                            })
        
        return experiments
    
    def load_experiment_metadata(self, experiment_path: str) -> Dict[str, Any]:
        """Load experiment metadata from config.json."""
        config_path = Path(experiment_path) / "metadata" / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_steering_metadata(self, experiment_path: str) -> Optional[Dict[str, Any]]:
        """Load steering metadata if available."""
        metadata_path = Path(experiment_path) / "steering_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def load_pickle_file(self, file_path: str) -> Any:
        """Load a pickle file safely."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_train_test_generations(self, experiment_path: str) -> Tuple[Any, Any]:
        """Load train and test generations for an experiment."""
        data_dir = Path(experiment_path) / "data"
        
        train_gen_path = data_dir / "train_generations.pkl"
        test_gen_path = data_dir / "test_generations.pkl"
        
        train_gen = self.load_pickle_file(train_gen_path) if train_gen_path.exists() else None
        test_gen = self.load_pickle_file(test_gen_path) if test_gen_path.exists() else None
        
        return train_gen, test_gen
    
    def load_dataset_and_split(self, experiment_path: str) -> Tuple[Any, Any]:
        """Load dataset and train/test split information."""
        data_dir = Path(experiment_path) / "data"
        
        dataset_path = data_dir / "dataset.pkl"
        split_path = data_dir / "train_test_split.pkl"
        
        dataset = self.load_pickle_file(dataset_path) if dataset_path.exists() else None
        split_info = self.load_pickle_file(split_path) if split_path.exists() else None
        
        return dataset, split_info
    
    def load_steering_results(self, experiment_path: str) -> Dict[str, Any]:
        """Load all steering results for different alpha values."""
        steering_dir = Path(experiment_path) / "steering"
        steering_results = {}
        
        if not steering_dir.exists():
            return steering_results
            
        for steering_file in steering_dir.glob("steering_alpha_*.pkl"):
            # Parse filename: steering_alpha_{value}_{direction}.pkl
            match = re.match(r'steering_alpha_([^_]+)_([^.]+)\.pkl', steering_file.name)
            if match:
                alpha_value, direction = match.groups()
                key = f"alpha_{alpha_value}_{direction}"
                steering_results[key] = self.load_pickle_file(steering_file)
        
        return steering_results


class DatasetProcessor:
    """Processor for different dataset formats to extract questions and labels."""
    
    @staticmethod
    def extract_labels_from_dataset(dataset_name: str, dataset_data: Any, split_info: Any = None) -> Tuple[List[str], List[str]]:
        """Extract questions and ground truth labels from dataset."""
        if dataset_name == "sports_understanding":
            return DatasetProcessor._process_sports_understanding(dataset_data, split_info)
        elif dataset_name == "anachronisms":
            return DatasetProcessor._process_anachronisms(dataset_data, split_info)
        elif dataset_name == "logical_deduction":
            return DatasetProcessor._process_logical_deduction(dataset_data, split_info)
        elif dataset_name == "social_chemistry":
            return DatasetProcessor._process_social_chemistry(dataset_data, split_info)
        else:
            print(f"Unknown dataset: {dataset_name}")
            return [], []
    
    @staticmethod
    def _process_sports_understanding(dataset_data: Any, split_info: Any) -> Tuple[List[str], List[str]]:
        """Process sports understanding dataset."""
        questions = []
        labels = []
        
        if isinstance(dataset_data, list):
            for example in dataset_data:
                if isinstance(example, list) and len(example) >= 2:
                    questions.append(example[0])  # Question
                    labels.append(example[1])     # Label (yes/no)
        
        return questions, labels
    
    @staticmethod
    def _process_anachronisms(dataset_data: Any, split_info: Any) -> Tuple[List[str], List[str]]:
        """Process anachronisms dataset."""
        questions = []
        labels = []
        
        if isinstance(dataset_data, list):
            for example in dataset_data:
                if isinstance(example, list) and len(example) >= 2:
                    questions.append(example[0])  # Question
                    labels.append(example[1])     # Label (yes/no)
        
        return questions, labels
    
    @staticmethod
    def _process_logical_deduction(dataset_data: Any, split_info: Any) -> Tuple[List[str], List[str]]:
        """Process logical deduction dataset."""
        questions = []
        labels = []
        
        if isinstance(dataset_data, list):
            for example in dataset_data:
                if isinstance(example, list) and len(example) >= 3:
                    # Combine object description and statement for question
                    question = f"{example[0]}\n\nStatement: {example[1]}"
                    questions.append(question)
                    labels.append(example[2])     # Label (yes/no)
        
        return questions, labels
    
    @staticmethod
    def _process_social_chemistry(dataset_data: Any, split_info: Any) -> Tuple[List[str], List[str]]:
        """Process social chemistry dataset."""
        questions = []
        labels = []
        
        if isinstance(dataset_data, list):
            for example in dataset_data:
                if isinstance(example, list) and len(example) >= 2:
                    questions.append(example[0])  # Action/question
                    labels.append(example[1])     # Label (yes/no)
        
        return questions, labels


class ResponseParser:
    """Parser for extracting predicted labels from model responses."""
    
    @staticmethod
    def extract_predicted_label(response: str, dataset_name: str) -> str:
        """Extract predicted label from model response."""
        if not response or not isinstance(response, str):
            return "unknown"
        
        response = response.strip().lower()
        
        # Common patterns for yes/no answers
        if "yes" in response and "no" not in response:
            return "yes"
        elif "no" in response and "yes" not in response:
            return "no"
        elif response.startswith("yes"):
            return "yes"
        elif response.startswith("no"):
            return "no"
        
        # For logical deduction, might have different patterns
        if dataset_name == "logical_deduction":
            if any(word in response for word in ["plausible", "correct", "true", "valid"]):
                return "yes"
            elif any(word in response for word in ["implausible", "incorrect", "false", "invalid"]):
                return "no"
        
        # For social chemistry
        if dataset_name == "social_chemistry":
            if any(word in response for word in ["appropriate", "acceptable", "ok", "fine"]):
                return "yes"
            elif any(word in response for word in ["inappropriate", "unacceptable", "wrong", "bad"]):
                return "no"
        
        # Fallback: look for any yes/no patterns
        if re.search(r'\byes\b', response):
            return "yes"
        elif re.search(r'\bno\b', response):
            return "no"
        
        return "unknown"