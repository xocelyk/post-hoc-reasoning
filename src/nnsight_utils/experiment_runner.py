"""
Unified NNsight experiment runner using nnsight_utils.

This module provides a drop-in replacement for both EnhancedExperimentRunner
and NNsightExperimentRunner, using the unified nnsight_utils API.
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Import from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cache_manager import ExperimentCache, ExperimentConfig, ExperimentManager
from config import ExperimentRunConfig, create_experiment_configs
from data_loading import load_all_datasets
from parsing_utils import parse_response
from visualizer import create_visualizer

# Import from nnsight_utils
from . import (
    NNsightChatModel,
    extract_activations,
    train_probes,
    generate_with_steering,
    ProbeResult,
    smart_empty_cache,
    memory_cleanup_context
)


class UnifiedExperimentRunner:
    """Unified experiment runner using nnsight_utils for all probe methods."""

    def __init__(self, run_config: ExperimentRunConfig):
        self.run_config = run_config
        self.exp_manager = ExperimentManager(run_config.cache_dir)
        self.visualizer = create_visualizer(run_config.interactive)

        # Setup logging
        self.setup_logging()

        # Create experiment configurations
        self.experiment_configs = create_experiment_configs(run_config)
        self.logger.info(
            f"Created {len(self.experiment_configs)} experiment configurations"
        )

        # Track experiment status
        self.experiments_status = {}

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = os.path.join(self.run_config.cache_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"unified_experiment_run_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("UnifiedExperimentRunner")

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model response to extract answer."""
        return parse_response(response, thinking=True)

    def batch_get_generations(
        self, 
        prompts: List[str], 
        model: NNsightChatModel, 
        temperature: float = 0.7, 
        max_new_tokens: int = 100
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        self.logger.info(f"Generating for {len(prompts)} prompts")
        
        generations = []
        for prompt in prompts:
            # Use nnsight_utils generation
            response = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
            generations.append(response)
        
        return generations

    def generate_and_cache_data(
        self, model: NNsightChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Generate and cache model data (generations and activations)."""
        if cache.has_generations() and cache.has_activations():
            self.logger.info(
                f"Data already cached for {config.model_name} on {config.dataset_name}"
            )
            return True

        self.logger.info(
            f"Generating data for {config.model_name} on {config.dataset_name}"
        )

        # Load or use cached dataset
        if cache.has_dataset():
            dataset = cache.load_pickle(cache.get_dataset_path())
        else:
            datasets = load_all_datasets()
            dataset = datasets[config.dataset_name]
            cache.save_pickle(dataset, cache.get_dataset_path())

        # Load or create train/test split
        if cache.has_train_test_split():
            train_dataset, test_dataset = cache.load_pickle(
                cache.get_train_test_split_path()
            )
        else:
            train_dataset, test_dataset = train_test_split(
                dataset,
                train_size=config.train_size,
                test_size=config.test_size,
                random_state=config.split_seed,
            )
            cache.save_pickle(
                (train_dataset, test_dataset), cache.get_train_test_split_path()
            )

        # Log data label distributions
        train_labels = [item["correct_answer"] for item in train_dataset]
        test_labels = [item["correct_answer"] for item in test_dataset]
        train_distribution = {label: train_labels.count(label) for label in set(train_labels)}
        test_distribution = {label: test_labels.count(label) for label in set(test_labels)}
        
        self.logger.info(f"Train data labels: {train_distribution} (total: {len(train_dataset)})")
        self.logger.info(f"Test data labels: {test_distribution} (total: {len(test_dataset)})")

        # Generate data if not cached
        if not cache.has_generations():
            # Process training data
            train_prompts = [model.apply_chat_template(item["prompt"]) for item in train_dataset]
            train_generations = self.batch_get_generations(
                train_prompts, model, config.temperature, config.max_new_tokens
            )
            
            # Parse training responses
            train_results = []
            for i, (item, generation) in enumerate(zip(train_dataset, train_generations)):
                pred_letter, pred_answer = self.parse_response(generation)
                train_results.append({
                    "prompt": item["prompt"],
                    "response": generation,
                    "pred_answer": pred_answer,
                    "pred_letter": pred_letter,
                    "correct_answer": item["correct_answer"],
                    "correct_letter": item["correct_letter"]
                })

            # Process test data
            test_prompts = [model.apply_chat_template(item["prompt"]) for item in test_dataset]
            test_generations = self.batch_get_generations(
                test_prompts, model, config.temperature, config.max_new_tokens
            )
            
            # Parse test responses
            test_results = []
            for i, (item, generation) in enumerate(zip(test_dataset, test_generations)):
                pred_letter, pred_answer = self.parse_response(generation)
                test_results.append({
                    "prompt": item["prompt"],
                    "response": generation,
                    "pred_answer": pred_answer,
                    "pred_letter": pred_letter,
                    "correct_answer": item["correct_answer"],
                    "correct_letter": item["correct_letter"]
                })

            # Cache generations
            cache.save_pickle(train_results, cache.get_train_generations_path())
            cache.save_pickle(test_results, cache.get_test_generations_path())
            
            self.logger.info("Generations cached successfully")

        # Extract activations if not cached
        if not cache.has_activations():
            # Load cached generations
            train_results = cache.load_pickle(cache.get_train_generations_path())
            test_results = cache.load_pickle(cache.get_test_generations_path())
            
            # Prepare prompts for activation extraction
            train_prompts = [model.apply_chat_template(r["prompt"]) for r in train_results]
            test_prompts = [model.apply_chat_template(r["prompt"]) for r in test_results]
            
            self.logger.info("Extracting activations...")
            
            # Extract activations using nnsight_utils
            with memory_cleanup_context():
                train_activations = extract_activations(model, train_prompts)
                test_activations = extract_activations(model, test_prompts)
            
            # Cache activations
            cache.save_pickle(train_activations, cache.get_train_activations_path())
            cache.save_pickle(test_activations, cache.get_test_activations_path())
            
            self.logger.info("Activations cached successfully")

        return True

    def train_and_cache_probes(
        self, model: NNsightChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Train and cache probes using method-aware approach."""
        # Get method from config
        method = getattr(self.run_config.steering, 'method', 'caa-single-layer')
        
        if cache.has_probes(method):
            self.logger.info(
                f"Probes already cached for {method} on {config.model_name}/{config.dataset_name}"
            )
            return True

        self.logger.info(
            f"Training {method} probes for {config.model_name} on {config.dataset_name}"
        )

        # Load cached data
        train_results = cache.load_pickle(cache.get_train_generations_path())
        test_results = cache.load_pickle(cache.get_test_generations_path())
        train_activations = cache.load_pickle(cache.get_train_activations_path())
        test_activations = cache.load_pickle(cache.get_test_activations_path())

        # Filter to only correct predictions for training
        correct_train_indices = [
            i for i, r in enumerate(train_results) 
            if r["pred_answer"] == r["correct_answer"]
        ]
        
        if len(correct_train_indices) == 0:
            self.logger.warning("No correct training predictions found!")
            return False
        
        # Extract correct predictions and corresponding activations
        correct_train_results = [train_results[i] for i in correct_train_indices]
        correct_train_activations = train_activations[correct_train_indices]
        
        # Extract labels
        train_labels = [r["correct_answer"] for r in correct_train_results]
        test_labels = [r["correct_answer"] for r in test_results]
        
        self.logger.info(f"Training on {len(train_labels)} correct predictions")

        # Train probes using unified interface
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # For test data in evaluation, use test activations and test labels
            if method == "caa-single-layer":
                # CAA methods need test data for layer selection
                probe_result = train_probes(
                    method=method,
                    activations=correct_train_activations,
                    labels=train_labels,
                    test_activations=test_activations,
                    test_labels=test_labels
                )
            else:
                # Other methods
                probe_result = train_probes(
                    method=method,
                    activations=correct_train_activations,
                    labels=train_labels
                )

        # Cache results using method-specific paths
        cache.save_pickle(probe_result.vectors, cache.get_probe_coefficients_path(method))
        cache.save_json(probe_result.scores, cache.get_auc_scores_path(method))
        
        # Log results
        if probe_result.best_layer is not None:
            best_score = probe_result.scores[probe_result.best_layer]
            self.logger.info(
                f"Best layer: {probe_result.best_layer} (score: {best_score:.4f})"
            )
        else:
            avg_score = np.mean(list(probe_result.scores.values()))
            self.logger.info(f"Average score across layers: {avg_score:.4f}")

        # Update visualizer
        if hasattr(self.visualizer, "update_auc_scores"):
            scores_list = [probe_result.scores.get(i, 0) for i in range(model.cfg.n_layers)]
            self.visualizer.update_auc_scores(
                config.model_name, config.dataset_name, scores_list
            )

        return True

    def run_steering_experiments(
        self, model: NNsightChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Run steering experiments using method-aware approach."""
        method = getattr(self.run_config.steering, 'method', 'caa-single-layer')
        
        self.logger.info(
            f"Running {method} steering for {config.model_name} on {config.dataset_name}"
        )

        # Load cached data
        test_results = cache.load_pickle(cache.get_test_generations_path())
        
        # Load probe results
        probe_vectors = cache.load_pickle(cache.get_probe_coefficients_path(method))
        probe_scores = cache.load_json(cache.get_auc_scores_path(method))
        
        # Convert scores back to int keys
        probe_scores = {int(k): v for k, v in probe_scores.items()}
        
        # Create ProbeResult object
        best_layer = None
        if method == "caa-single-layer":
            best_layer = max(probe_scores.keys(), key=lambda k: (probe_scores[k], -k))
        
        probe_result = ProbeResult(
            method=method,
            vectors=probe_vectors,
            scores=probe_scores,
            best_layer=best_layer
        )

        # Create test subsets
        yes_test_data = [
            result for result in test_results
            if result["pred_answer"] == "yes" and result["correct_answer"] == "yes"
        ]
        no_test_data = [
            result for result in test_results
            if result["pred_answer"] == "no" and result["correct_answer"] == "no"
        ]

        self.logger.info(f"Test data: {len(yes_test_data)} yes, {len(no_test_data)} no")

        # Run steering for each alpha
        for alpha in config.alpha_range:
            # Yes to No steering (negative alpha)
            alpha_yes = -abs(alpha)
            if not cache.has_steering_results(alpha_yes, "yes"):
                results_yes = self.generate_steered_examples(
                    model, yes_test_data, probe_result, alpha_yes, config
                )
                cache.save_pickle(
                    results_yes, cache.get_steering_results_path(alpha_yes, "yes")
                )

                success_rate = (
                    sum(r["success"] for r in results_yes) / len(results_yes)
                    if results_yes else 0
                )
                self.logger.info(
                    f"Alpha {alpha_yes:+.1f} (yes→no): {success_rate:.2f} success rate"
                )

            # No to Yes steering (positive alpha)
            alpha_no = abs(alpha)
            if not cache.has_steering_results(alpha_no, "no"):
                results_no = self.generate_steered_examples(
                    model, no_test_data, probe_result, alpha_no, config
                )
                cache.save_pickle(
                    results_no, cache.get_steering_results_path(alpha_no, "no")
                )

                success_rate = (
                    sum(r["success"] for r in results_no) / len(results_no)
                    if results_no else 0
                )
                self.logger.info(
                    f"Alpha {alpha_no:+.1f} (no→yes): {success_rate:.2f} success rate"
                )

        return True

    def generate_steered_examples(
        self,
        model: NNsightChatModel,
        test_data: List[Dict],
        probe_result: ProbeResult,
        alpha: float,
        config: ExperimentConfig,
    ) -> List[Dict]:
        """Generate steered examples using unified steering interface."""
        steered_results = []

        with memory_cleanup_context():
            for example in test_data:
                # Convert to tokens
                prompt_string = model.apply_chat_template(example["prompt"])
                tokens = model.to_tokens(prompt_string, prepend_bos=False)
                
                # Use unified steering interface (method-aware)
                steered_response = generate_with_steering(
                    model=model,
                    tokens=tokens,
                    probe_result=probe_result,
                    alpha=alpha,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature
                )
                
                # Parse response
                pred_letter, pred_answer = self.parse_response(steered_response)
                
                # Determine success
                target_answer = "no" if alpha < 0 else "yes"
                success = pred_answer == target_answer
                
                steered_results.append({
                    "original_answer": example["pred_answer"],
                    "steered_answer": pred_answer,
                    "target_answer": target_answer,
                    "success": success,
                    "response": steered_response
                })

        return steered_results

    def run_single_experiment(
        self, config: ExperimentConfig, model: NNsightChatModel
    ) -> Dict[str, Any]:
        """Run a single experiment with the given configuration."""
        cache = self.exp_manager.add_experiment(config)
        
        self.logger.info(f"Starting experiment: {config.model_name} on {config.dataset_name}")
        
        # Phase 1: Generate and cache data
        if not self.generate_and_cache_data(model, config, cache):
            self.logger.error("Failed to generate and cache data")
            return {"success": False, "error": "Data generation failed"}
        
        # Phase 2: Train and cache probes
        if not self.train_and_cache_probes(model, config, cache):
            self.logger.error("Failed to train and cache probes")
            return {"success": False, "error": "Probe training failed"}
        
        # Phase 3: Run steering experiments
        if not self.run_steering_experiments(model, config, cache):
            self.logger.error("Failed to run steering experiments")
            return {"success": False, "error": "Steering experiments failed"}
        
        self.logger.info(f"Completed experiment: {config.model_name} on {config.dataset_name}")
        
        return {
            "success": True,
            "model_name": config.model_name,
            "dataset_name": config.dataset_name,
            "cache_dir": cache.cache_dir
        }

    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all configured experiments."""
        results = {}
        
        for config in self.experiment_configs:
            # Load model
            self.logger.info(f"Loading model: {config.model_name}")
            model = NNsightChatModel(config.model_name)
            
            try:
                result = self.run_single_experiment(config, model)
                results[f"{config.model_name}_{config.dataset_name}"] = result
            except Exception as e:
                self.logger.error(f"Experiment failed: {e}")
                results[f"{config.model_name}_{config.dataset_name}"] = {
                    "success": False,
                    "error": str(e)
                }
            finally:
                # Clean up model memory
                del model
                smart_empty_cache()
        
        return results

    def resume_experiments(self, experiment_ids: Optional[List[str]] = None):
        """Resume incomplete experiments."""
        # For now, just run all experiments (could be enhanced to check completion status)
        self.logger.info("Resuming experiments...")
        return self.run_all_experiments()

    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary of experiment results."""
        return self.exp_manager.get_experiments_summary()