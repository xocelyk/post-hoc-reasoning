import gc
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rich.live import Live
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cache_manager import ExperimentCache, ExperimentConfig, ExperimentManager
from config import ExperimentRunConfig, create_experiment_configs
from data_loading import load_all_datasets
from models import ChatModel
from parsing_utils import parse_response
from utils import generate_with_hooks
from visualizer import create_visualizer
from steering_methods import create_steering_method, format_steering_results


class PromptDataset(Dataset):
    """Dataset wrapper for prompts."""

    def __init__(self, data: List[Dict[str, Any]], model: ChatModel):
        self.data = data
        self.model = model

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, Tuple[str, str]]:
        prompt_data = self.data[idx]["prompt"]
        
        # Convert chat messages to string using model's chat template
        if not isinstance(prompt_data, list):
            raise TypeError(f"Expected prompt to be a list of chat messages, got {type(prompt_data)}")
        
        prompt_string = self.model.apply_chat_template(prompt_data)
        
        return prompt_string, (
            self.data[idx]["correct_answer"],
            self.data[idx]["correct_letter"],
        )


class EnhancedExperimentRunner:
    """Enhanced experiment runner with caching, visualization, and resume functionality."""

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
        log_file = os.path.join(log_dir, f"experiment_run_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("ExperimentRunner")

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model response to extract answer."""
        return parse_response(response, thinking=True)

    def batch_get_resid_activations(self, prompts: List[str], model: ChatModel):
        """Get residual stream activations for a batch of prompts with memory optimization."""
        layers = list(range(model.cfg.n_layers))
        
        with torch.no_grad():  # Ensure no gradients are computed
            tokens = model.to_tokens(prompts, prepend_bos=True)
            _, cache = model.run_with_cache(tokens, pos_slice=-1)

            # Pre-allocate with float32 to save memory
            activations = np.zeros((len(prompts), model.cfg.n_layers, model.cfg.d_model), dtype=np.float32)

            for layer in layers:
                layer_activations = cache["resid_post", layer]
                # Convert to float32 before converting to numpy to avoid BFloat16 issues on MPS
                layer_activations = layer_activations.squeeze().detach().float().cpu().numpy().astype(np.float32)
                activations[:, layer, :] = layer_activations
                
                # Immediate cleanup
                del layer_activations
                
                # More aggressive memory cleanup
                if layer % 5 == 0:  # Every 5 layers
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
            
            # Final cleanup
            del cache, tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        return activations

    def batch_get_generations(
        self, prompts: List[str], model: ChatModel, temperature=0.7, max_new_tokens=100
    ):
        """Get generations for a batch of prompts with memory optimization."""
        with torch.no_grad():  # Ensure no gradients are computed
            tokens = model.to_tokens(prompts, prepend_bos=True)
            token_generations = model.generate(
                tokens, max_new_tokens=max_new_tokens, temperature=temperature, verbose=True
            )
            generations = model.to_string(token_generations)
            
            # Clean up immediately
            del tokens, token_generations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
        return generations

    def process_batch(
        self,
        prompts: List[str],
        correct_tups: Tuple[List[str], List[str]],
        model: ChatModel,
        get_activations=True,
        temperature=0.7,
        max_new_tokens=100,
        log_first_batch=False,
    ):
        """Process a batch of prompts."""
        correct_answers, correct_letters = correct_tups

        activations = (
            self.batch_get_resid_activations(prompts, model)
            if get_activations
            else None
        )
        
        generations = self.batch_get_generations(
            prompts, model, temperature=temperature, max_new_tokens=max_new_tokens
        )
        generations = [gen[len(prompt) :] for gen, prompt in zip(generations, prompts)]

        responses = [self.parse_response(response) for response in generations]
        pred_letters, pred_answers = zip(*responses)

        corrects = [
            pred == correct for pred, correct in zip(pred_letters, correct_letters)
        ]

        # Log details for first batch
        if log_first_batch:
            self.logger.info("FIRST BATCH DETAILED LOGGING")
            for i, prompt in enumerate(prompts):
                self.logger.info(f"--- SAMPLE {i+1} ---")
                self.logger.info(f"GENERATED RESPONSE: {repr(generations[i])}")
                self.logger.info(f"PARSED LETTER: {pred_letters[i]}")
                self.logger.info(f"PARSED ANSWER: {pred_answers[i]}")
                self.logger.info(f"CORRECT LETTER: {correct_letters[i]}")
                self.logger.info(f"CORRECT ANSWER: {correct_answers[i]}")
                self.logger.info(f"CORRECT: {corrects[i]}")

        return activations, generations, pred_letters, pred_answers, corrects

    def generate_and_cache_data(
        self, model: ChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Generate and cache model data (generations and activations)."""
        if self.run_config.use_cache and cache.has_generations() and cache.has_activations():
            self.logger.info(
                f"Data already cached for {config.model_name} on {config.dataset_name}"
            )
            return True

        self.logger.info(
            f"Generating data for {config.model_name} on {config.dataset_name}"
        )

        # Load or use cached dataset
        if self.run_config.use_cache and cache.has_dataset():
            dataset = cache.load_pickle(cache.get_dataset_path())
        else:
            datasets = load_all_datasets()
            dataset = datasets[config.dataset_name]
            if self.run_config.use_cache:
                cache.save_pickle(dataset, cache.get_dataset_path())

        # Load or create train/test split
        if self.run_config.use_cache and cache.has_train_test_split():
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
            if self.run_config.use_cache:
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

        
        batch_size = next(
            (
                m.batch_size
                for m in self.run_config.models
                if m.name == config.model_name
            ),
            2,
        )

        # Process training data
        if not (self.run_config.use_cache and cache.has_generations()):
            train_dataloader = DataLoader(
                PromptDataset(train_dataset, model), batch_size=batch_size, shuffle=False
            )
            test_dataloader = DataLoader(
                PromptDataset(test_dataset, model), batch_size=batch_size, shuffle=False
            )

            self.logger.info("Processing TRAINING data...")
            train_results, train_activations = self.process_dataset(
                train_dataloader, model, config.train_size, config, phase="train"
            )
            self.logger.info("Processing TEST data...")
            test_results, test_activations = self.process_dataset(
                test_dataloader, model, len(test_dataset), config, phase="test"
            )

            # Always save results for downstream use
            cache.save_pickle(train_results, cache.get_train_generations_path())
            cache.save_pickle(test_results, cache.get_test_generations_path())
            cache.save_pickle(train_activations, cache.get_train_activations_path())
            cache.save_pickle(test_activations, cache.get_test_activations_path())

        return True

    def process_dataset(
        self,
        dataloader: DataLoader,
        model: ChatModel,
        max_samples: int,
        config: ExperimentConfig,
        phase: str = "unknown",
    ):
        """Process entire dataset."""
        results = []
        activations_list = []
        sample_count = 0

        all_corrects = []
        
        for batch_idx, (prompts, correct_tups) in enumerate(dataloader):
            # Enable detailed logging for first batch
            log_first_batch = (batch_idx == 0)
            
            activations, generations, pred_letters, pred_answers, corrects = (
                self.process_batch(
                    prompts,
                    correct_tups,
                    model,
                    get_activations=True,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                    log_first_batch=log_first_batch,
                )
            )

            sample_count += len(prompts)
            all_corrects.extend(corrects)

            for i, prompt in enumerate(prompts):
                result = {
                    "prompt": prompt,
                    "response": (pred_letters[i], pred_answers[i]),
                    "correct_letter": correct_tups[1][i],
                    "correct_answer": correct_tups[0][i],
                    "pred_letter": pred_letters[i],
                    "pred_answer": pred_answers[i],
                }
                results.append(result)
                if activations is not None:
                    activations_list.append(activations[i])
                
                # Log individual sample details
                correct_mark = "✓" if corrects[i] else "✗"
                self.logger.info(
                    f"{phase.upper()} sample {len(results)}/{max_samples}: "
                    f"pred={pred_letters[i]}/{pred_answers[i]}, "
                    f"correct={correct_tups[1][i]}/{correct_tups[0][i]} {correct_mark}"
                )

            if sample_count >= max_samples:
                break

        # Summary logging
        overall_accuracy = np.mean(all_corrects) if all_corrects else 0.0
        pred_answers_list = [r["pred_answer"] for r in results]
        pred_distribution = {answer: pred_answers_list.count(answer) for answer in set(pred_answers_list)}
        
        self.logger.info(
            f"{phase.upper()} phase complete: {len(results)} samples, "
            f"overall accuracy: {overall_accuracy:.2f}, "
            f"predictions: {pred_distribution}"
        )

        return results, activations_list

    def train_and_cache_probes(
        self, model: ChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Compute and cache contrastive activation vectors using configured steering method."""
        if self.run_config.use_cache and cache.has_probes():
            self.logger.info(
                f"Contrastive vectors already cached for {config.model_name} on {config.dataset_name}"
            )
            return True

        # Get steering method from config
        steering_method_name = getattr(self.run_config.steering, 'method', 'caa-single-layer')
        
        self.logger.info(
            f"Computing contrastive activation vectors using {steering_method_name} for {config.model_name} on {config.dataset_name}"
        )

        # Load cached data
        train_results = cache.load_pickle(cache.get_train_generations_path())
        test_results = cache.load_pickle(cache.get_test_generations_path())
        train_activations = cache.load_pickle(cache.get_train_activations_path())
        test_activations = cache.load_pickle(cache.get_test_activations_path())

        layers = list(range(model.cfg.n_layers))
        all_contrastive_vectors = []
        similarity_scores = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for layer in layers:
                # Prepare training data for contrastive vector computation
                train_layer_activations = []
                train_labels = []
                
                for idx, result in enumerate(train_results):
                    if result["pred_answer"] == result["correct_answer"]:
                        activation = train_activations[idx][layer]
                        train_layer_activations.append(activation)
                        train_labels.append(result["pred_answer"])

                # Compute contrastive vector for this layer (raw difference vector)
                contrastive_vector = self.extract_contrastive_vector(
                    train_layer_activations, train_labels, normalize=False
                )
                all_contrastive_vectors.append(contrastive_vector)

                # Evaluate using test data
                test_layer_activations = []
                test_labels = []
                
                for idx, result in enumerate(test_results):
                    if result["pred_answer"] == result["correct_answer"]:
                        activation = test_activations[idx][layer]
                        test_layer_activations.append(activation)
                        test_labels.append(result["pred_answer"])

                # Compute similarity score as evaluation metric
                similarity_score = self.evaluate_contrastive_vector(
                    contrastive_vector, test_layer_activations, test_labels
                )
                similarity_scores.append(similarity_score)

                self.logger.info(f"Layer {layer} Similarity Score: {similarity_score:.4f}")

        # Create and apply steering method
        try:
            if steering_method_name == "logistic-regression":
                # Prepare activation data for logistic regression
                train_activations_list = []
                train_labels_list = []
                test_activations_list = []
                test_labels_list = []
                
                # Collect training data
                for idx, result in enumerate(train_results):
                    if result["pred_answer"] == result["correct_answer"]:
                        train_activations_list.append(train_activations[idx])
                        train_labels_list.append(result["pred_answer"])
                
                # Collect test data
                for idx, result in enumerate(test_results):
                    if result["pred_answer"] == result["correct_answer"]:
                        test_activations_list.append(test_activations[idx])
                        test_labels_list.append(result["pred_answer"])
                
                steering_method = create_steering_method(
                    steering_method_name,
                    train_activations=train_activations_list,
                    train_labels=train_labels_list,
                    test_activations=test_activations_list,
                    test_labels=test_labels_list
                )
                # For logistic regression, we pass empty list since it doesn't use contrastive vectors
                final_steering_vectors = steering_method.compute_steering_vectors([])
            else:
                steering_method = create_steering_method(
                    steering_method_name, 
                    similarity_scores=similarity_scores
                )
                final_steering_vectors = steering_method.compute_steering_vectors(all_contrastive_vectors)
            
            # Log method-specific information
            if steering_method_name == "caa-single-layer":
                best_score = max(similarity_scores)
                best_layers = [i for i, score in enumerate(similarity_scores) if score == best_score]
                best_layer = layers[max(best_layers)]
                
                self.logger.info(
                    f"Best Similarity Score: {best_score:.4f} at layer {best_layer} for {config.model_name} on {config.dataset_name}"
                )
                
                if len(best_layers) > 1:
                    self.logger.info(
                        f"Tie between layers {[layers[i] for i in best_layers]} - selecting latest layer {best_layer}"
                    )
            elif steering_method_name == "caa-layer-incremental":
                self.logger.info(
                    f"Computed incremental steering vectors for all {len(layers)} layers"
                )
            elif steering_method_name == "logistic-regression":
                self.logger.info(
                    f"Trained logistic regression classifiers for all {len(layers)} layers"
                )
                
        except Exception as e:
            self.logger.error(f"Error creating steering method '{steering_method_name}': {e}")
            # Fallback to single-layer method
            steering_method = create_steering_method("caa-single-layer", similarity_scores=similarity_scores)
            final_steering_vectors = steering_method.compute_steering_vectors(all_contrastive_vectors)
        
        # Save results for downstream use
        cache.save_pickle(final_steering_vectors, cache.get_probe_coefficients_path())
        cache.save_json(similarity_scores, cache.get_auc_scores_path())
        
        # Save steering method metadata
        steering_metadata = format_steering_results(final_steering_vectors, steering_method_name)
        steering_metadata.update({
            "all_layer_scores": [float(score) for score in similarity_scores],
            "best_score": float(max(similarity_scores)),
            "best_layer": int(layers[np.argmax(similarity_scores)])
        })
        cache.save_json(steering_metadata, os.path.join(cache.cache_dir, "steering_metadata.json"))

        # Update visualizer
        if hasattr(self.visualizer, "update_auc_scores"):
            self.visualizer.update_auc_scores(
                config.model_name, config.dataset_name, similarity_scores
            )

        return True


    def extract_contrastive_vector(self, activations, labels, normalize=True):
        """Extract contrastive activation vector from activations and labels.
        
        Args:
            activations: numpy array of activations for this layer
            labels: list of labels ("yes" or "no")
            normalize: whether to normalize the vector (default: True)
            
        Returns:
            numpy array: normalized difference vector (mean_yes - mean_no)
        """
        activations_array = np.array(activations)
        labels_array = np.array(labels)
        
        # Separate activations by class
        yes_mask = labels_array == "yes"
        no_mask = labels_array == "no"
        
        if not np.any(yes_mask) or not np.any(no_mask):
            # If we don't have both classes, return zero vector
            return np.zeros(activations_array.shape[1])
        
        # Compute mean activations for each class
        mean_yes = np.mean(activations_array[yes_mask], axis=0)
        mean_no = np.mean(activations_array[no_mask], axis=0)
        
        # Compute difference vector: mean(yes) - mean(no)
        difference_vector = mean_yes - mean_no
        
        # Normalize the vector if requested
        if normalize:
            vector_norm = np.linalg.norm(difference_vector)
            if vector_norm > 0:  # Avoid division by zero
                difference_vector = difference_vector / vector_norm
            # If norm is 0, vector remains as zero vector
                
        return difference_vector

    def evaluate_contrastive_vector(self, contrastive_vector, activations, labels):
        """Evaluate contrastive vector using similarity scores.
        
        Args:
            contrastive_vector: The computed contrastive vector
            activations: Test activations for this layer
            labels: Test labels ("yes" or "no")
            
        Returns:
            float: Similarity score (higher is better)
        """
        if len(activations) == 0:
            return 0.0
            
        activations_array = np.array(activations)
        labels_array = np.array(labels)
        
        # Compute dot product of each activation with contrastive vector
        similarities = np.dot(activations_array, contrastive_vector)
        
        # Convert labels to binary (1 for "yes", 0 for "no")
        binary_labels = (labels_array == "yes").astype(int)
        
        # Compute correlation between similarities and labels
        # Higher correlation means the contrastive vector better separates the classes
        if len(set(binary_labels)) < 2:
            # If all labels are the same, return 0
            return 0.0
            
        correlation = np.corrcoef(similarities, binary_labels)[0, 1]
        
        # Return absolute correlation (we care about separation, not direction)
        return abs(correlation) if not np.isnan(correlation) else 0.0

    def run_steering_experiments(
        self, model: ChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Run steering experiments."""
        self.logger.info(
            f"Running steering for {config.model_name} on {config.dataset_name}"
        )

        # Load cached data
        test_results = cache.load_pickle(cache.get_test_generations_path())
        all_contrastive_vectors = cache.load_pickle(cache.get_probe_coefficients_path())

        # Create test subsets
        yes_test_data = [
            result
            for result in test_results
            if result["pred_answer"] == "yes" and result["correct_answer"] == "yes"
        ]
        no_test_data = [
            result
            for result in test_results
            if result["pred_answer"] == "no" and result["correct_answer"] == "no"
        ]

        layers = list(range(model.cfg.n_layers))

        for alpha in config.alpha_range:
            # Yes to No steering
            alpha_yes = -abs(alpha)  # Negative alpha steers "yes" to "no"
            if not (self.run_config.use_cache and cache.has_steering_results(alpha_yes, "yes")):
                results_yes = self.generate_steered_examples(
                    model, yes_test_data, all_contrastive_vectors, layers, alpha_yes, config
                )
                if self.run_config.use_cache:
                    cache.save_pickle(
                        results_yes, cache.get_steering_results_path(alpha_yes, "yes")
                    )

                success_rate = (
                    sum(r["success"] for r in results_yes) / len(results_yes)
                    if results_yes
                    else 0
                )
                self.logger.info(
                    f"Alpha {alpha_yes:+.1f} (yes): {success_rate:.2f} success rate"
                )

                # Update visualizer
                if hasattr(self.visualizer, "update_steering_results"):
                    self.visualizer.update_steering_results(
                        config.model_name,
                        config.dataset_name,
                        alpha_yes,
                        "yes",
                        success_rate,
                    )

            # No to Yes steering
            alpha_no = abs(alpha)  # Positive alpha steers "no" to "yes"
            if not (self.run_config.use_cache and cache.has_steering_results(alpha_no, "no")):
                results_no = self.generate_steered_examples(
                    model, no_test_data, all_contrastive_vectors, layers, alpha_no, config
                )
                if self.run_config.use_cache:
                    cache.save_pickle(
                        results_no, cache.get_steering_results_path(alpha_no, "no")
                    )

                success_rate = (
                    sum(r["success"] for r in results_no) / len(results_no)
                    if results_no
                    else 0
                )
                self.logger.info(
                    f"Alpha {alpha_no:+.1f} (no): {success_rate:.2f} success rate"
                )

                # Update visualizer
                if hasattr(self.visualizer, "update_steering_results"):
                    self.visualizer.update_steering_results(
                        config.model_name,
                        config.dataset_name,
                        alpha_no,
                        "no",
                        success_rate,
                    )

        return True

    def generate_steered_examples(
        self,
        model: ChatModel,
        test_data: List[Dict],
        all_contrastive_vectors: List,
        layers: List[int],
        alpha: float,
        config: ExperimentConfig,
    ):
        """Generate steered examples with memory optimization."""
        steered_results = []
        
        # Convert to tensor once and reuse
        steering_vectors = np.array(all_contrastive_vectors)
        
        # Process in smaller batches to reduce memory pressure
        batch_size = min(10, len(test_data))  # Process max 10 examples at once
        
        for batch_start in range(0, len(test_data), batch_size):
            batch_end = min(batch_start + batch_size, len(test_data))
            batch_data = test_data[batch_start:batch_end]
            
            self.logger.info(f"Processing steering batch {batch_start//batch_size + 1}/{(len(test_data) + batch_size - 1)//batch_size}")
            
            for i, example in enumerate(batch_data):
                global_idx = batch_start + i
                example_prompt = example["prompt"]
                
                # Memory-efficient tokenization
                with torch.no_grad():
                    example_tokens = model.to_tokens(example_prompt, prepend_bos=False)

                    generation = generate_with_hooks(
                        model,
                        example_tokens,
                        temperature=config.temperature,
                        max_new_tokens=config.max_new_tokens,
                        alpha=alpha,
                        steering_vectors=steering_vectors,
                        layers=layers,
                        verbose=True,
                    )
                    
                    # Clean up tokens immediately
                    del example_tokens

                new_letter, new_answer = self.parse_response(generation)
                orig = example["pred_answer"]
                success = (orig == "yes" and new_answer == "no") or (
                    orig == "no" and new_answer == "yes"
                )

                steered_results.append(
                    {
                        "original_prompt": example_prompt,
                        "steered_generation": generation,
                        "original_answer": orig,
                        "new_answer": new_answer,
                        "original_letter": example["pred_letter"],
                        "new_letter": new_letter,
                        "alpha": alpha,
                        "success": success,
                    }
                )
                
                # Clear variables and cache after each generation
                del generation, new_letter, new_answer
                
                # Aggressive memory cleanup every few examples
                if (global_idx + 1) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
                self.logger.info(f"Completed steering example {global_idx + 1}/{len(test_data)}")
            
            # Clean up between batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return steered_results

    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        cache = self.exp_manager.add_experiment(config)
        exp_key = f"{config.model_name}_{config.dataset_name}"

        try:
            # Load model
            self.logger.info(f"Loading model: {config.model_name}")
            model = ChatModel(config.model_name)

            # Step 1: Generate and cache data
            if not self.generate_and_cache_data(model, config, cache):
                return {"success": False, "error": "Failed to generate data"}

            # Step 2: Train and cache probes
            if not self.train_and_cache_probes(model, config, cache):
                return {"success": False, "error": "Failed to train probes"}

            # Step 3: Run steering experiments
            if not self.run_steering_experiments(model, config, cache):
                return {"success": False, "error": "Failed to run steering"}

            # Update status
            status = cache.get_experiment_status()
            self.experiments_status[exp_key] = status

            return {"success": True, "status": status}

        except Exception as e:
            self.logger.error(f"Error in experiment {exp_key}: {str(e)}")
            return {"success": False, "error": str(e)}

        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def run_all_experiments(self):
        """Run all experiments."""
        self.logger.info(f"Starting {len(self.experiment_configs)} experiments")

        # Initialize experiment status
        for config in self.experiment_configs:
            exp_key = f"{config.model_name}_{config.dataset_name}"
            cache = self.exp_manager.add_experiment(config)
            self.experiments_status[exp_key] = cache.get_experiment_status()

        if self.run_config.interactive and hasattr(self.visualizer, "layout"):
            # Run with live visualization
            with Live(self.visualizer.layout, refresh_per_second=2) as live:
                self._run_experiments_with_visualization(live)
        else:
            # Run without live visualization
            self._run_experiments_simple()

        # Print final summary
        if hasattr(self.visualizer, "print_summary"):
            self.visualizer.print_summary(self.experiments_status)

    def _run_experiments_with_visualization(self, live):
        """Run experiments with live visualization."""
        for config in self.experiment_configs:
            exp_key = f"{config.model_name}_{config.dataset_name}"

            # Update display
            self.visualizer.update_display(self.experiments_status)

            # Run experiment
            result = self.run_single_experiment(config)

            # Update status
            if result["success"]:
                self.experiments_status[exp_key] = result["status"]

            # Final display update
            self.visualizer.update_display(self.experiments_status)

    def _run_experiments_simple(self):
        """Run experiments without live visualization."""
        for config in self.experiment_configs:
            exp_key = f"{config.model_name}_{config.dataset_name}"

            if hasattr(self.visualizer, "start_experiment"):
                self.visualizer.start_experiment(
                    config.model_name, config.dataset_name, 3
                )

            result = self.run_single_experiment(config)

            if result["success"]:
                self.experiments_status[exp_key] = result["status"]
                if hasattr(self.visualizer, "complete_experiment"):
                    self.visualizer.complete_experiment(
                        config.model_name, config.dataset_name
                    )

    def resume_experiments(self, experiment_ids: Optional[List[str]] = None):
        """Resume incomplete experiments."""
        self.logger.info("Resuming incomplete experiments")

        # Get list of all experiments
        all_experiments = self.exp_manager.list_experiments()

        # Filter by provided IDs if specified
        if experiment_ids:
            all_experiments = [
                (exp_id, config)
                for exp_id, config in all_experiments
                if exp_id in experiment_ids
            ]

        # Find incomplete experiments
        incomplete_experiments = []
        for exp_id, config in all_experiments:
            cache = ExperimentCache(config, self.run_config.cache_dir)
            status = cache.get_experiment_status()

            # Check if experiment is incomplete
            if not (
                status["generations"]
                and status["probes"]
                and status["steering_complete"]
            ):
                incomplete_experiments.append(config)

        if not incomplete_experiments:
            self.logger.info("No incomplete experiments found")
            return

        self.logger.info(f"Found {len(incomplete_experiments)} incomplete experiments")

        # Update experiment configs and run
        self.experiment_configs = incomplete_experiments
        self.run_all_experiments()

    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary of all experiment results."""
        return self.exp_manager.get_experiments_summary()
