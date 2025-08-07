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
from utils import generate_with_steering
from visualizer import create_visualizer
from steering_methods import create_steering_method, format_steering_results

# W&B integration
try:
    from wandb_integration import WandbExperimentLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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

        # W&B logger will be initialized per experiment
        self.wandb_logger = None
        self.wandb_available = WANDB_AVAILABLE and not os.environ.get("WANDB_DISABLED", "false").lower() == "true"

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
        # Override max_new_tokens for DeepSeek models (though they typically use nnsight)
        if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
            max_new_tokens = 2000
            self.logger.info(f"Using DeepSeek model, overriding max_new_tokens to {max_new_tokens}")
        
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
            datasets = load_all_datasets(model_name=config.model_name)
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
                
                # Log first 5 training examples to W&B and console
                if phase == "train" and len(results) <= 5:
                    is_correct = pred_answers[i] == correct_tups[0][i]
                    # Need to get the actual generation text for logging
                    generation_text = generations[i] if i < len(generations) else ""
                    
                    # Log to W&B
                    if self.wandb_logger:
                        self.wandb_logger.log_training_example(
                            example_idx=len(results) - 1,
                            prompt=prompt,
                            response=generation_text,
                            predicted_answer=pred_answers[i],
                            correct_answer=correct_tups[0][i],
                            is_correct=is_correct
                        )
                    
                    # Also print to console
                    self.logger.info("=" * 80)
                    self.logger.info(f"TRAINING EXAMPLE {len(results)}/5")
                    self.logger.info("=" * 80)
                    
                    # Format prompt for display
                    if isinstance(prompt, list):
                        prompt_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in prompt])
                    else:
                        prompt_str = str(prompt)
                    
                    self.logger.info(f"FULL PROMPT:\n{prompt_str}")
                    self.logger.info("-" * 80)
                    self.logger.info(f"RESPONSE:\n{generation_text}")
                    self.logger.info("-" * 80)
                    self.logger.info(f"PARSED ANSWER: {pred_answers[i]}")
                    self.logger.info(f"CORRECT ANSWER: {correct_tups[0][i]}")
                    self.logger.info(f"RESULT: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    self.logger.info("=" * 80)
                
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
        auc_scores = []

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

                # Compute AUC score as evaluation metric
                auc_score = self.evaluate_contrastive_vector(
                    contrastive_vector, test_layer_activations, test_labels
                )
                auc_scores.append(auc_score)

                self.logger.info(f"Layer {layer} AUC Score: {auc_score:.4f}")

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
                    similarity_scores=auc_scores
                )
                final_steering_vectors = steering_method.compute_steering_vectors(all_contrastive_vectors)
            
            # Log method-specific information
            if steering_method_name == "caa-single-layer":
                best_score = max(auc_scores)
                best_layers = [i for i, score in enumerate(auc_scores) if score == best_score]
                best_layer = layers[min(best_layers)]  # EARLIEST layer wins ties
                
                self.logger.info(
                    f"Best Similarity Score: {best_score:.4f} at layer {best_layer} for {config.model_name} on {config.dataset_name}"
                )
                
                if len(best_layers) > 1:
                    self.logger.info(
                        f"Tie between layers {[layers[i] for i in best_layers]} - selecting earliest layer {best_layer}"
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
            steering_method = create_steering_method("caa-single-layer", similarity_scores=auc_scores)
            final_steering_vectors = steering_method.compute_steering_vectors(all_contrastive_vectors)
        
        # Save results for downstream use
        # Special handling for single-layer method
        if steering_method_name == "caa-single-layer":
            # Find best layer (earliest wins ties)
            best_score = max(auc_scores)
            best_indices = [i for i, score in enumerate(auc_scores) if score == best_score]
            best_layer_idx = min(best_indices)  # EARLIEST layer wins ties
            best_layer = layers[best_layer_idx]
            
            # Only save the best layer's vector as a dict
            single_layer_vectors = {best_layer: final_steering_vectors[best_layer_idx]}
            cache.save_pickle(single_layer_vectors, cache.get_probe_coefficients_path())
            
            self.logger.info(
                f"CAA Single-Layer: Saved only layer {best_layer} steering vector"
            )
        else:
            # For other methods, save all vectors
            cache.save_pickle(final_steering_vectors, cache.get_probe_coefficients_path())
        
        cache.save_json(auc_scores, cache.get_auc_scores_path())
        
        # Save steering method metadata
        steering_metadata = format_steering_results(final_steering_vectors, steering_method_name)
        steering_metadata.update({
            "method": steering_method_name,
            "all_layer_scores": [float(score) for score in auc_scores],
            "best_score": float(max(auc_scores)),
            "best_layer": int(layers[min([i for i, s in enumerate(auc_scores) if s == max(auc_scores)])])
        })
        cache.save_json(steering_metadata, os.path.join(cache.cache_dir, "steering_metadata.json"))

        # Log to W&B
        if self.wandb_logger:
            # Log probe results for each layer
            for i, (layer, score) in enumerate(zip(layers, auc_scores)):
                self.wandb_logger.log_probe_training(
                    layer=layer,
                    train_auc=score,  # Using similarity score as proxy
                    test_auc=score,
                    similarity_score=score,
                    method=steering_method_name
                )
            
            # Log best layer selection
            best_score = max(auc_scores)
            best_indices = [i for i, score in enumerate(auc_scores) if score == best_score]
            best_idx = min(best_indices)  # EARLIEST layer wins ties
            self.wandb_logger.log_best_layer_selection(
                best_layer=layers[best_idx],
                best_score=auc_scores[best_idx],
                method=steering_method_name,
                all_scores=auc_scores
            )

        # Update visualizer
        if hasattr(self.visualizer, "update_auc_scores"):
            self.visualizer.update_auc_scores(
                config.model_name, config.dataset_name, auc_scores
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
        """Evaluate contrastive vector using AUC-ROC score.
        
        Args:
            contrastive_vector: The computed contrastive vector
            activations: Test activations for this layer
            labels: Test labels ("yes" or "no")
            
        Returns:
            float: AUC-ROC score (higher is better, 0.5 is random)
        """
        from sklearn.metrics import roc_auc_score
        
        if len(activations) == 0:
            return 0.5  # Return random baseline for empty data
            
        activations_array = np.array(activations)
        labels_array = np.array(labels)
        
        # Compute dot product of each activation with contrastive vector
        similarities = np.dot(activations_array, contrastive_vector)
        
        # Convert labels to binary (1 for "yes", 0 for "no")
        binary_labels = (labels_array == "yes").astype(int)
        
        # Check if we have both classes
        if len(set(binary_labels)) < 2:
            # If all labels are the same, return random baseline
            return 0.5
            
        try:
            # Compute AUC-ROC score
            auc_score = roc_auc_score(binary_labels, similarities)
            return auc_score
        except ValueError:
            # Handle edge cases
            return 0.5

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
                        f"Alpha {alpha_yes:+.1f} (yes): {success_rate:.2f} success, "
                        f"{failure_rate:.2f} failure, {unparsed_rate:.2f} unparsed"
                    )
                    
                    # Log summary to W&B
                    if self.wandb_logger:
                        self.wandb_logger.log_steering_summary(
                            alpha=abs(alpha_yes),
                            direction="yes_to_no",
                            total_examples=total,
                            success_count=success_count,
                            failure_count=failure_count,
                            unparsed_count=unparsed_count
                        )
                        
                        # Log results table
                        self.wandb_logger.log_steering_results_table(
                            results=results_yes[:20],
                            alpha=alpha_yes,
                            direction="yes_to_no"
                        )
                else:
                    success_rate = 0
                    self.logger.info(f"Alpha {alpha_yes:+.1f} (yes): No results")

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
                        f"Alpha {alpha_no:+.1f} (no): {success_rate:.2f} success, "
                        f"{failure_rate:.2f} failure, {unparsed_rate:.2f} unparsed"
                    )
                    
                    # Log summary to W&B
                    if self.wandb_logger:
                        self.wandb_logger.log_steering_summary(
                            alpha=alpha_no,
                            direction="no_to_yes",
                            total_examples=total,
                            success_count=success_count,
                            failure_count=failure_count,
                            unparsed_count=unparsed_count
                        )
                        
                        # Log results table
                        self.wandb_logger.log_steering_results_table(
                            results=results_no[:20],
                            alpha=alpha_no,
                            direction="no_to_yes"
                        )
                else:
                    success_rate = 0
                    self.logger.info(f"Alpha {alpha_no:+.1f} (no): No results")

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
        # Override max_new_tokens for DeepSeek models (though they typically use nnsight)
        max_new_tokens = config.max_new_tokens
        if hasattr(model, 'model_name') and model.model_name.lower().startswith('deepseek'):
            max_new_tokens = 2000
            self.logger.info(f"Using DeepSeek model for steering, overriding max_new_tokens to {max_new_tokens}")
        
        steered_results = []
        
        # Load steering method metadata to check method type
        cache_dir = config.get_cache_dir(self.run_config.cache_dir)
        metadata_path = os.path.join(cache_dir, "steering_metadata.json")
        
        steering_method = "unknown"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                steering_method = metadata.get("method", "unknown")
        
        # Handle single-layer method specially
        if steering_method == "caa-single-layer" and isinstance(all_contrastive_vectors, dict):
            # For single-layer, all_contrastive_vectors is a dict with only the best layer
            # Create zero array for all layers
            num_layers = len(layers)
            best_layer = list(all_contrastive_vectors.keys())[0]
            d_model = all_contrastive_vectors[best_layer].shape[0]
            steering_vectors = np.zeros((num_layers, d_model))
            
            # Only fill in the selected layer
            if best_layer < num_layers:
                steering_vectors[best_layer] = all_contrastive_vectors[best_layer]
            
            # Only steer at the best layer
            layers_to_steer = [best_layer]
            
            self.logger.info(
                f"CAA Single-Layer: Steering only at layer {best_layer} (alpha={alpha})"
            )
        else:
            # For other methods, use vectors as-is
            steering_vectors = np.array(all_contrastive_vectors)
            layers_to_steer = layers
        
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

                    generation = generate_with_steering(
                        model,
                        example_tokens,
                        temperature=config.temperature,
                        max_new_tokens=max_new_tokens,
                        alpha=alpha,
                        steering_vectors=steering_vectors,
                        layers=layers_to_steer,
                    )
                    
                    # Clean up tokens immediately
                    del example_tokens

                new_letter, new_answer = self.parse_response(generation)
                orig = example["pred_answer"]
                
                # Log the generation and parsing results
                self.logger.info(f"Generated response: {generation}")
                self.logger.info(f"Parsed: letter='{new_letter}', answer='{new_answer}'")
                
                # Determine target answer based on original answer
                target_answer = "no" if orig == "yes" else "yes"
                is_valid_parse = new_answer in ["yes", "no"]
                
                if not is_valid_parse:
                    category = "unparsed"
                    success = False
                elif new_answer == target_answer:
                    category = "success"
                    success = True
                else:
                    category = "failure"
                    success = False

                result = {
                    "original_prompt": example_prompt,
                    "steered_generation": generation,
                    "original_answer": orig,
                    "new_answer": new_answer,
                    "target_answer": target_answer,
                    "original_letter": example["pred_letter"],
                    "new_letter": new_letter,
                    "alpha": alpha,
                    "success": success,
                    "category": category,
                    "is_valid_parse": is_valid_parse,
                }
                steered_results.append(result)
                
                # Log to W&B
                if self.wandb_logger:
                    direction = "yes_to_no" if orig == "yes" else "no_to_yes"
                    self.wandb_logger.log_steering_example(
                        alpha=alpha,
                        direction=direction,
                        prompt=str(example_prompt) if not isinstance(example_prompt, str) else example_prompt,
                        original_answer=orig,
                        steered_response=generation,
                        steered_answer=new_answer,
                        target_answer=target_answer,
                        category=category,
                        example_idx=global_idx,
                        model_name=config.model_name,
                        dataset_name=config.dataset_name
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

    def print_experiment_summary(self, config: ExperimentConfig, cache: ExperimentCache):
        """Print a nice summary of the completed experiment."""
        print(f"\n{'='*60}")
        print(f"📊 EXPERIMENT SUMMARY: {config.model_name}")
        print(f"Dataset: {config.dataset_name}")
        print(f"{'='*60}")
        
        # Load and summarize probe results
        try:
            auc_scores = cache.load_json(cache.get_auc_scores_path())
            best_score = float(max(auc_scores))
            best_indices = [i for i, s in enumerate(auc_scores) if s == best_score]
            best_layer = int(min(best_indices))  # EARLIEST layer wins ties
            avg_score = float(np.mean(auc_scores))
            
            print(f"🎯 PROBE PERFORMANCE:")
            print(f"   Best layer: {best_layer} (similarity score: {best_score:.4f})")
            print(f"   Average similarity score: {avg_score:.4f}")
            print(f"   Total layers analyzed: {len(auc_scores)}")
        except:
            print(f"🎯 PROBE PERFORMANCE: Data not available")
        
        # Load and summarize steering results
        completed_steering = cache.get_completed_steering()
        if completed_steering:
            print(f"\n🎮 STEERING RESULTS:")
            
            # Group by alpha value
            steering_by_alpha = {}
            for alpha, direction in completed_steering:
                if alpha not in steering_by_alpha:
                    steering_by_alpha[alpha] = {}
                steering_by_alpha[alpha][direction] = True
            
            for alpha in sorted(steering_by_alpha.keys()):
                directions = steering_by_alpha[alpha]
                alpha_display = f"{alpha:+.1f}" if alpha != 0 else "0.0"
                
                direction_summary = []
                for direction in ['yes', 'no']:
                    if direction in directions:
                        try:
                            # Load steering results to get success rates
                            results = cache.load_pickle(cache.get_steering_results_path(alpha, direction))
                            if results:
                                total = len(results)
                                success_count = sum(1 for r in results if r.get("success", False))
                                parsed_count = sum(1 for r in results if r.get("category") != "unparsed")
                                
                                if parsed_count > 0:
                                    success_rate_parsed = success_count / parsed_count
                                    direction_summary.append(f"{direction}: {success_rate_parsed:.0%} ({parsed_count} parsed)")
                                else:
                                    direction_summary.append(f"{direction}: 0% (0 parsed)")
                            else:
                                direction_summary.append(f"{direction}: no data")
                        except:
                            direction_summary.append(f"{direction}: ✓")
                    else:
                        direction_summary.append(f"{direction}: ✗")
                
                print(f"   α={alpha_display}: {', '.join(direction_summary)}")
        else:
            print(f"\n🎮 STEERING RESULTS: No completed steering experiments")
        
        print(f"{'='*60}\n")

    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        cache = self.exp_manager.add_experiment(config)
        exp_key = f"{config.model_name}_{config.dataset_name}"

        # Initialize W&B for this specific experiment
        if self.wandb_available:
            try:
                steering_method = getattr(self.run_config.steering, 'method', 'caa-single-layer')
                self.wandb_logger = WandbExperimentLogger(
                    experiment_config={
                        "model_name": config.model_name,
                        "dataset_name": config.dataset_name,
                        "steering_method": steering_method,
                        "train_size": config.train_size,
                        "test_size": config.test_size,
                        "split_seed": config.split_seed,
                        "temperature": config.temperature,
                        "max_new_tokens": config.max_new_tokens,
                        "alpha_range": config.alpha_range,
                        "runner": "transformer_lens"
                    }
                )
                self.logger.info(f"W&B logging initialized for {exp_key}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B for {exp_key}: {e}")
                self.wandb_logger = None

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

            # Print individual experiment summary
            self.print_experiment_summary(config, cache)

            return {"success": True, "status": status}

        except Exception as e:
            self.logger.error(f"Error in experiment {exp_key}: {str(e)}")
            return {"success": False, "error": str(e)}

        finally:
            # Finish W&B run for this experiment
            if self.wandb_logger:
                self.wandb_logger.finish()
                self.wandb_logger = None
                
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
        
        # Print comprehensive final summary
        self.print_final_summary()

    def print_final_summary(self):
        """Print comprehensive summary statistics for all experiments."""
        print(f"\n{'#'*80}")
        print(f"🏆 FINAL EXPERIMENT SUMMARY")
        print(f"{'#'*80}")
        
        successful_experiments = []
        failed_experiments = []
        
        for config in self.experiment_configs:
            exp_key = f"{config.model_name}_{config.dataset_name}"
            if exp_key in self.experiments_status:
                status = self.experiments_status[exp_key]
                if status.get("generations", False) and status.get("probes", False):
                    successful_experiments.append((config, exp_key))
                else:
                    failed_experiments.append((config, exp_key))
            else:
                failed_experiments.append((config, exp_key))
        
        print(f"📈 OVERALL STATISTICS:")
        print(f"   Total experiments: {len(self.experiment_configs)}")
        print(f"   Successful: {len(successful_experiments)}")
        print(f"   Failed: {len(failed_experiments)}")
        print(f"   Success rate: {len(successful_experiments) / len(self.experiment_configs):.1%}")
        
        if successful_experiments:
            print(f"\n🎯 PROBE PERFORMANCE COMPARISON:")
            
            all_best_scores = []
            all_avg_scores = []
            model_scores = []
            
            for config, exp_key in successful_experiments:
                cache = self.exp_manager.add_experiment(config)
                try:
                    auc_scores = cache.load_json(cache.get_auc_scores_path())
                    best_score = float(max(auc_scores))
                    avg_score = float(np.mean(auc_scores))
                    best_indices = [i for i, s in enumerate(auc_scores) if s == best_score]
                    best_layer = int(min(best_indices))  # EARLIEST layer wins ties
                    
                    all_best_scores.append(best_score)
                    all_avg_scores.append(avg_score)
                    model_scores.append((config.model_name, best_score, avg_score, best_layer))
                except:
                    pass
            
            if model_scores:
                # Sort by best score descending
                model_scores.sort(key=lambda x: x[1], reverse=True)
                
                print(f"   📊 Model Rankings (by best layer performance):")
                for i, (model_name, best_score, avg_score, best_layer) in enumerate(model_scores, 1):
                    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                    print(f"      {i}. {model_short[:30]:<30} | Best: {best_score:.4f} (L{best_layer}) | Avg: {avg_score:.4f}")
                
                print(f"\n   🔬 Cross-Model Statistics:")
                print(f"      Best score across all models: {max(all_best_scores):.4f}")
                print(f"      Average best score: {np.mean(all_best_scores):.4f}")
                print(f"      Standard deviation: {np.std(all_best_scores):.4f}")
                print(f"      Average layer performance: {np.mean(all_avg_scores):.4f}")
        
        if successful_experiments:
            print(f"\n🎮 STEERING PERFORMANCE SUMMARY:")
            
            steering_success_by_alpha = {}
            total_steering_experiments = 0
            successful_steering_experiments = 0
            total_parsed = 0
            total_unparsed = 0
            total_failures = 0
            
            for config, exp_key in successful_experiments:
                cache = self.exp_manager.add_experiment(config)
                completed_steering = cache.get_completed_steering()
                
                for alpha, direction in completed_steering:
                    if alpha not in steering_success_by_alpha:
                        steering_success_by_alpha[alpha] = {
                            'total': 0, 
                            'successful': 0, 
                            'parsed': 0, 
                            'unparsed': 0,
                            'failures': 0
                        }
                    
                    try:
                        results = cache.load_pickle(cache.get_steering_results_path(alpha, direction))
                        if results:
                            total_steering_experiments += len(results)
                            successes = sum(1 for r in results if r.get("success", False))
                            parsed = sum(1 for r in results if r.get("category") != "unparsed")
                            unparsed = sum(1 for r in results if r.get("category") == "unparsed")
                            failures = sum(1 for r in results if r.get("category") == "failure")
                            
                            successful_steering_experiments += successes
                            total_parsed += parsed
                            total_unparsed += unparsed
                            total_failures += failures
                            
                            steering_success_by_alpha[alpha]['total'] += len(results)
                            steering_success_by_alpha[alpha]['successful'] += successes
                            steering_success_by_alpha[alpha]['parsed'] += parsed
                            steering_success_by_alpha[alpha]['unparsed'] += unparsed
                            steering_success_by_alpha[alpha]['failures'] += failures
                    except:
                        pass
            
            if steering_success_by_alpha:
                print(f"   📊 Steering Results by Alpha:")
                for alpha in sorted(steering_success_by_alpha.keys()):
                    data = steering_success_by_alpha[alpha]
                    if data['total'] > 0:
                        alpha_display = f"{alpha:+.1f}" if alpha != 0 else "0.0"
                        parsed_rate = data['parsed'] / data['total']
                        unparsed_rate = data['unparsed'] / data['total']
                        
                        # Success rate among all attempts
                        success_rate_all = data['successful'] / data['total']
                        
                        # Success rate among parsed responses only
                        if data['parsed'] > 0:
                            success_rate_parsed = data['successful'] / data['parsed']
                            failure_rate_parsed = data['failures'] / data['parsed']
                            print(f"      α={alpha_display}: {success_rate_all:.1%} success overall | "
                                  f"Parsed: {parsed_rate:.1%} ({data['parsed']}/{data['total']}) | "
                                  f"Among parsed: {success_rate_parsed:.1%} success, {failure_rate_parsed:.1%} failure")
                        else:
                            print(f"      α={alpha_display}: {success_rate_all:.1%} success overall | "
                                  f"Parsed: {parsed_rate:.1%} ({data['parsed']}/{data['total']}) | "
                                  f"No parsed responses")
                
                print(f"\n   📈 Overall Steering Statistics:")
                if total_steering_experiments > 0:
                    overall_steering_rate = successful_steering_experiments / total_steering_experiments
                    overall_parsed_rate = total_parsed / total_steering_experiments
                    overall_unparsed_rate = total_unparsed / total_steering_experiments
                    
                    print(f"      Total steering attempts: {total_steering_experiments}")
                    print(f"      Overall success rate: {overall_steering_rate:.1%} ({successful_steering_experiments}/{total_steering_experiments})")
                    print(f"      Parsing rate: {overall_parsed_rate:.1%} parsed, {overall_unparsed_rate:.1%} unparsed")
                    
                    if total_parsed > 0:
                        success_rate_among_parsed = successful_steering_experiments / total_parsed
                        failure_rate_among_parsed = total_failures / total_parsed
                        print(f"      Among parsed responses: {success_rate_among_parsed:.1%} success, {failure_rate_among_parsed:.1%} failure")
                    else:
                        print(f"      Among parsed responses: No parsed responses available")
        
        if failed_experiments:
            print(f"\n❌ FAILED EXPERIMENTS:")
            for config, exp_key in failed_experiments:
                model_short = config.model_name.split('/')[-1] if '/' in config.model_name else config.model_name
                print(f"   • {model_short} on {config.dataset_name}")
        
        print(f"\n{'#'*80}")
        print(f"🎉 Experiment run completed!")
        print(f"{'#'*80}\n")

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
