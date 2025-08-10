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
from parsing_utils import parse_response, filter_think_tags
from visualizer import create_visualizer
from wandb_integration import WandbExperimentLogger

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
from .core.generation import generate_text
from .steering.kv_cached_generation import generate_with_kv_cached_steering, estimate_kv_cache_savings


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
        
        # W&B logger will be created per experiment
        self.wandb_logger = None

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

    def parse_response(self, response: str, model_name: Optional[str] = None) -> Tuple[str, str]:
        """Parse model response to extract answer."""
        # Use the same parser for all models now, including DeepSeek
        # DeepSeek will use thinking=True prompts and the standard parser
        return parse_response(response, thinking=True)

    def batch_get_generations(
        self, 
        prompts: List[str], 
        model: NNsightChatModel, 
        temperature: float = 0.7, 
        max_new_tokens: int = 100
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        # Override max_new_tokens for DeepSeek models
        if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
            max_new_tokens = 2000
            self.logger.info(f"Using DeepSeek model, overriding max_new_tokens to {max_new_tokens}")
        
        self.logger.info(f"Generating for {len(prompts)} prompts")
        
        generations = []
        for i, prompt in enumerate(prompts):
            if i % 5 == 0:
                self.logger.info(f"  Progress: {i}/{len(prompts)} prompts generated")
            
            # Use the basic generation function without steering
            response = generate_text(
                model=model,
                prompt=prompt,  # generate_text handles tokenization
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
            generations.append(response)
            
            # Log the first prompt and response for debugging
            if i == 0:
                # Filter think tags for DeepSeek models when displaying
                display_response = response
                if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
                    display_response = filter_think_tags(response)
                    
                self.logger.info("=" * 80)
                self.logger.info("FIRST PROMPT AND RESPONSE:")
                self.logger.info("=" * 80)
                self.logger.info(f"Full Prompt:\n{prompt}")
                self.logger.info("-" * 80)
                self.logger.info(f"Response (filtered for DeepSeek):\n{display_response}")
                self.logger.info("=" * 80)
            
            # Memory cleanup every few generations
            if i % 5 == 0:
                smart_empty_cache()
        
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
            datasets = load_all_datasets(model_name=config.model_name)
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
            train_prompts = []
            for i, item in enumerate(train_dataset):
                try:
                    # For prompts with incomplete assistant messages, don't use add_generation_prompt
                    # as it conflicts with continue_final_message
                    formatted = model.apply_chat_template(item["prompt"])
                    train_prompts.append(formatted)
                except Exception as e:
                    self.logger.error(f"Error formatting prompt {i}: {e}")
                    self.logger.error(f"Prompt type: {type(item['prompt'])}")
                    self.logger.error(f"Prompt: {item['prompt']}")
                    raise
            
            train_generations = self.batch_get_generations(
                train_prompts, model, config.temperature, config.max_new_tokens
            )
            
            # Parse training responses
            train_results = []
            for i, (item, generation) in enumerate(zip(train_dataset, train_generations)):
                # Filter think tags for DeepSeek models before parsing
                response_to_parse = generation
                if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
                    response_to_parse = filter_think_tags(generation)
                
                pred_letter, pred_answer = self.parse_response(response_to_parse, model.model_name)
                train_results.append({
                    "prompt": item["prompt"],
                    "response": generation,
                    "pred_answer": pred_answer,
                    "pred_letter": pred_letter,
                    "correct_answer": item["correct_answer"],
                    "correct_letter": item["correct_letter"]
                })
                
                # Log first 5 training examples to W&B and console
                if i < 5:
                    is_correct = pred_answer == item["correct_answer"]
                    
                    # Log to W&B
                    if self.wandb_logger:
                        self.wandb_logger.log_training_example(
                            example_idx=i,
                            prompt=item["prompt"],
                            response=generation,
                            predicted_answer=pred_answer,
                            correct_answer=item["correct_answer"],
                            is_correct=is_correct
                        )
                    
                    # Also print to console
                    self.logger.info("=" * 80)
                    self.logger.info(f"TRAINING EXAMPLE {i + 1}/5")
                    self.logger.info("=" * 80)
                    
                    # Format prompt for display
                    if isinstance(item["prompt"], list):
                        prompt_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in item["prompt"]])
                    else:
                        prompt_str = str(item["prompt"])
                    
                    self.logger.info(f"FULL PROMPT:\n{prompt_str}")
                    self.logger.info("-" * 80)
                    
                    # Filter think tags for DeepSeek models when displaying
                    display_response = generation
                    if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
                        display_response = filter_think_tags(generation)
                    
                    self.logger.info(f"RESPONSE (filtered for DeepSeek):\n{display_response}")
                    self.logger.info("-" * 80)
                    self.logger.info(f"PARSED ANSWER: {pred_answer}")
                    self.logger.info(f"CORRECT ANSWER: {item['correct_answer']}")
                    self.logger.info(f"RESULT: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    self.logger.info("=" * 80)

            # Process test data
            test_prompts = [model.apply_chat_template(item["prompt"]) for item in test_dataset]
            test_generations = self.batch_get_generations(
                test_prompts, model, config.temperature, config.max_new_tokens
            )
            
            # Parse test responses
            test_results = []
            for i, (item, generation) in enumerate(zip(test_dataset, test_generations)):
                # Filter think tags for DeepSeek models before parsing
                response_to_parse = generation
                if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
                    response_to_parse = filter_think_tags(generation)
                
                pred_letter, pred_answer = self.parse_response(response_to_parse, model.model_name)
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
            
            # Log first activation prompt to verify formatting
            if train_prompts:
                self.logger.info("First activation extraction prompt:")
                self.logger.info(f"{train_prompts[0][:500]}...")  # First 500 chars
                
            self.logger.info("Extracting activations...")
            
            # Extract activations using nnsight_utils with batching
            with memory_cleanup_context():
                # Use smaller batch size for activation extraction to prevent OOM
                batch_size = getattr(config, 'activation_batch_size', 1)
                self.logger.info(f"Extracting train activations (batch_size={batch_size})...")
                train_activations = extract_activations(model, train_prompts, batch_size=batch_size)
                
                # Clear memory between train and test
                smart_empty_cache()
                
                self.logger.info(f"Extracting test activations (batch_size={batch_size})...")
                test_activations = extract_activations(model, test_prompts, batch_size=batch_size)
            
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
        
        # Handle both list and array formats for activations
        if isinstance(train_activations, list):
            # If activations are a list of arrays, stack them
            correct_train_activations = np.stack([train_activations[i] for i in correct_train_indices])
        else:
            # If already an array, use fancy indexing
            correct_train_activations = train_activations[correct_train_indices]
        
        # Extract labels
        train_labels = [r["correct_answer"] for r in correct_train_results]
        test_labels = [r["correct_answer"] for r in test_results]
        
        # Convert test activations to numpy array if needed
        if isinstance(test_activations, list):
            test_activations = np.stack(test_activations)
        
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
            
        # Log to W&B
        if self.wandb_logger:
            # Log probe training results for each layer
            for layer, score in probe_result.scores.items():
                self.wandb_logger.log_probe_training(
                    layer=int(layer),
                    train_auc=score,  # Using score as train AUC
                    test_auc=score,   # For now, using same score
                    method=method
                )
            
            # Log best layer selection
            if probe_result.best_layer is not None:
                all_scores = [probe_result.scores.get(i, 0) for i in range(model.cfg.n_layers)]
                self.wandb_logger.log_best_layer_selection(
                    best_layer=probe_result.best_layer,
                    best_score=probe_result.scores[probe_result.best_layer],
                    method=method,
                    all_scores=all_scores
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

        # Apply max_gen limit if specified
        max_gen = self.run_config.steering.max_gen
        if max_gen is not None:
            original_yes_count = len(yes_test_data)
            original_no_count = len(no_test_data)
            yes_test_data = yes_test_data[:max_gen]
            no_test_data = no_test_data[:max_gen]
            
            self.logger.info(
                f"Applied max_gen limit: Using {len(yes_test_data)}/{original_yes_count} yes examples, "
                f"{len(no_test_data)}/{original_no_count} no examples (max_gen={max_gen})"
            )
        else:
            self.logger.info(
                f"Using all available examples: {len(yes_test_data)} yes examples, {len(no_test_data)} no examples"
            )

        # Early stopping state - track per direction
        should_stop_yes_to_no = False
        should_stop_no_to_yes = False

        # Run steering for each alpha in sorted order for proper early stopping
        for alpha in sorted(config.alpha_range):
            # Yes to No steering (negative alpha)
            alpha_yes = -abs(alpha)
            if should_stop_yes_to_no and self.run_config.steering.stop_alpha_early and alpha > 0:
                self.logger.info(f"Skipping alpha {alpha_yes} (yes→no): Early stopping due to 100% unparsed rate")
            elif not cache.has_steering_results(alpha_yes, "yes"):
                results_yes = self.generate_steered_examples(
                    model, yes_test_data, probe_result, alpha_yes, config
                )
                cache.save_pickle(
                    results_yes, cache.get_steering_results_path(alpha_yes, "yes")
                )

                if results_yes:
                    # Calculate category breakdown
                    total = len(results_yes)
                    success_count = sum(1 for r in results_yes if r["category"] == "success")
                    failure_count = sum(1 for r in results_yes if r["category"] == "failure")
                    unparsed_count = sum(1 for r in results_yes if r["category"] == "unparsed")
                    
                    success_rate = success_count / total
                    failure_rate = failure_count / total
                    unparsed_rate = unparsed_count / total
                    
                    self.logger.info(
                        f"Alpha {alpha_yes:+.1f} (yes→no): {success_rate:.2f} success, "
                        f"{failure_rate:.2f} failure, {unparsed_rate:.2f} unparsed"
                    )
                    
                    # Check for early stopping - 100% unparsed rate
                    if self.run_config.steering.stop_alpha_early and alpha > 0 and unparsed_rate >= 1.0:
                        should_stop_yes_to_no = True
                        self.logger.info(f"Early stopping yes→no direction: 100% unparsed rate at alpha {alpha_yes}")
                    
                    # Log to W&B
                    if self.wandb_logger:
                        self.wandb_logger.log_steering_summary(
                            alpha=alpha_yes,
                            direction="yes_to_no",
                            total_examples=total,
                            success_count=success_count,
                            failure_count=failure_count,
                            unparsed_count=unparsed_count
                        )
                else:
                    self.logger.info(f"Alpha {alpha_yes:+.1f} (yes→no): No results")

            # No to Yes steering (positive alpha)
            alpha_no = abs(alpha)
            if should_stop_no_to_yes and self.run_config.steering.stop_alpha_early and alpha > 0:
                self.logger.info(f"Skipping alpha {alpha_no} (no→yes): Early stopping due to 100% unparsed rate")
            elif not cache.has_steering_results(alpha_no, "no"):
                results_no = self.generate_steered_examples(
                    model, no_test_data, probe_result, alpha_no, config
                )
                cache.save_pickle(
                    results_no, cache.get_steering_results_path(alpha_no, "no")
                )

                if results_no:
                    # Calculate category breakdown
                    total = len(results_no)
                    success_count = sum(1 for r in results_no if r["category"] == "success")
                    failure_count = sum(1 for r in results_no if r["category"] == "failure")
                    unparsed_count = sum(1 for r in results_no if r["category"] == "unparsed")
                    
                    success_rate = success_count / total
                    failure_rate = failure_count / total
                    unparsed_rate = unparsed_count / total
                    
                    self.logger.info(
                        f"Alpha {alpha_no:+.1f} (no→yes): {success_rate:.2f} success, "
                        f"{failure_rate:.2f} failure, {unparsed_rate:.2f} unparsed"
                    )
                    
                    # Check for early stopping - 100% unparsed rate
                    if self.run_config.steering.stop_alpha_early and alpha > 0 and unparsed_rate >= 1.0:
                        should_stop_no_to_yes = True
                        self.logger.info(f"Early stopping no→yes direction: 100% unparsed rate at alpha {alpha_no}")
                    
                    # Log to W&B
                    if self.wandb_logger:
                        self.wandb_logger.log_steering_summary(
                            alpha=alpha_no,
                            direction="no_to_yes",
                            total_examples=total,
                            success_count=success_count,
                            failure_count=failure_count,
                            unparsed_count=unparsed_count
                        )
                else:
                    self.logger.info(f"Alpha {alpha_no:+.1f} (no→yes): No results")

        return True

    def generate_steered_examples(
        self,
        model: NNsightChatModel,
        test_data: List[Dict],
        probe_result: ProbeResult,
        alpha: float,
        config: ExperimentConfig,
    ) -> List[Dict]:
        """Generate steered examples using KV-cached generation for optimal performance."""
        # Override max_new_tokens for DeepSeek models
        max_new_tokens = config.max_new_tokens
        if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
            max_new_tokens = 2000
            self.logger.info(f"Using DeepSeek model for steering, overriding max_new_tokens to {max_new_tokens}")
        
        # Prepare prompts for batch processing
        # Check if prompts are already formatted strings or need formatting
        if test_data and isinstance(test_data[0]["prompt"], str):
            # Prompts are already formatted strings
            prompt_strings = [example["prompt"] for example in test_data]
        else:
            # Prompts need formatting (they are message lists)
            prompt_strings = [model.apply_chat_template(example["prompt"]) for example in test_data]
        
        # Estimate potential KV cache savings
        cache_stats = estimate_kv_cache_savings(prompt_strings, model.cfg.n_layers)
        if cache_stats['shared_prefix_length'] > 50:
            self.logger.info(
                f"Using KV caching - estimated speedup: {cache_stats['estimated_speedup']} "
                f"(shared prefix: {cache_stats['shared_prefix_length']} chars)"
            )
        
        with memory_cleanup_context():
            # Use KV-cached generation for better performance
            steered_responses = generate_with_kv_cached_steering(
                model=model,
                prompts=prompt_strings,
                probe_result=probe_result,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                temperature=config.temperature
            )
        
        # Process results
        steered_results = []
        for i, (example, steered_response) in enumerate(zip(test_data, steered_responses)):
            # Filter think tags for DeepSeek models before parsing
            response_to_parse = steered_response
            if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
                response_to_parse = filter_think_tags(steered_response)
            
            # Parse response
            pred_letter, pred_answer = self.parse_response(response_to_parse)
            
            # Determine success and category
            target_answer = "no" if alpha < 0 else "yes"
            is_valid_parse = pred_answer in ["yes", "no"]
            
            if not is_valid_parse:
                category = "unparsed"
                success = False
            elif pred_answer == target_answer:
                category = "success"
                success = True
            else:
                category = "failure"
                success = False
            
            # Log detailed steering result
            # Filter think tags for DeepSeek models when displaying
            display_response = steered_response
            if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
                display_response = filter_think_tags(steered_response)
                
            self.logger.info(
                f"Steering result {i+1}/{len(test_data)} (α={alpha}):\n"
                f"  Original: {example['pred_answer']} → Target: {target_answer}\n"
                f"  Response (filtered):\n{display_response}\n"
                f"  Parsed letter: '{pred_letter}', Parsed answer: '{pred_answer}'\n"
                f"  Result: {category} (valid_parse={is_valid_parse}, success={success})"
            )
            
            steered_results.append({
                "original_answer": example["pred_answer"],
                "steered_answer": pred_answer,
                "target_answer": target_answer,
                "success": success,
                "category": category,
                "is_valid_parse": is_valid_parse,
                "response": steered_response
            })
            
            # Log to W&B
            if self.wandb_logger:
                direction = "yes_to_no" if alpha < 0 else "no_to_yes"
                self.wandb_logger.log_steering_example(
                    alpha=abs(alpha),
                    direction=direction,
                    prompt=example.get("prompt", ""),
                    original_answer=example["pred_answer"],
                    steered_response=steered_response,
                    steered_answer=pred_answer,
                    target_answer=target_answer,
                    category=category,
                    example_idx=i
                )

        return steered_results

    def run_single_experiment(
        self, config: ExperimentConfig, model: NNsightChatModel
    ) -> Dict[str, Any]:
        """Run a single experiment with the given configuration."""
        cache = self.exp_manager.add_experiment(config)
        
        # Initialize W&B for this experiment
        steering_method = getattr(self.run_config.steering, 'method', 'caa-single-layer')
        wandb_config = {
            "model_name": config.model_name,
            "dataset_name": config.dataset_name,
            "steering_method": steering_method,
            "train_size": config.train_size,
            "test_size": config.test_size,
            "split_seed": config.split_seed,
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
            "alpha_range": config.alpha_range,
            "runner": "nnsight"
        }
        self.wandb_logger = WandbExperimentLogger(experiment_config=wandb_config)
        
        self.logger.info(f"Starting experiment: {config.model_name} on {config.dataset_name}")
        
        try:
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
            
            # Log experiment summary to W&B
            if self.wandb_logger:
                self.wandb_logger.log_experiment_summary({
                    "experiment_completed": True,
                    "model_name": config.model_name,
                    "dataset_name": config.dataset_name,
                    "steering_method": steering_method,
                    "cache_dir": cache.cache_dir
                })
            
            return {
                "success": True,
                "model_name": config.model_name,
                "dataset_name": config.dataset_name,
                "cache_dir": cache.cache_dir
            }
            
        finally:
            # Finish W&B run
            if self.wandb_logger:
                self.wandb_logger.finish()
                self.wandb_logger = None

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
                import traceback
                self.logger.error(f"Experiment failed: {e}")
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                results[f"{config.model_name}_{config.dataset_name}"] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
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