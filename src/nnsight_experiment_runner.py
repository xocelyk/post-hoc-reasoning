"""
NNsight-based experiment runner providing drop-in replacement for EnhancedExperimentRunner.

This module maintains the exact same interface and functionality as the original
experiment runner but uses nnsight for broader model compatibility.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cache_manager import ExperimentCache, ExperimentConfig, ExperimentManager
from config import ExperimentRunConfig, create_experiment_configs
from data_loading import load_all_datasets
from nnsight_models import NNsightChatModel
from parsing_utils import parse_response
from nnsight_utils import batch_get_resid_activations
from nnsight_steering import generate_with_nnsight_steering


class NNsightPromptDataset(Dataset):
    """Dataset wrapper for prompts compatible with NNsightChatModel."""

    def __init__(self, data: List[Dict[str, Any]], model: NNsightChatModel):
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


class NNsightExperimentRunner:
    """NNsight-based experiment runner with caching, visualization, and resume functionality."""

    def __init__(self, run_config: ExperimentRunConfig):
        self.run_config = run_config
        self.exp_manager = ExperimentManager(run_config.cache_dir)

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
        log_file = os.path.join(log_dir, f"nnsight_experiment_run_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("NNsightExperimentRunner")

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model response to extract answer."""
        return parse_response(response, thinking=True)

    def batch_get_resid_activations(self, prompts: List[str], model: NNsightChatModel):
        """Get residual stream activations for a batch of prompts using nnsight."""
        self.logger.info(f"Extracting activations for {len(prompts)} prompts")
        
        return batch_get_resid_activations(
            model=model,
            prompts=prompts,
            layers=None,  # Extract all layers
            position="last"  # Extract final position
        )

    def batch_get_generations(
        self, prompts: List[str], model: NNsightChatModel, temperature=0.7, max_new_tokens=100
    ):
        """Get generations for a batch of prompts using nnsight."""
        self.logger.info(f"Generating for {len(prompts)} prompts")
        
        # Generate for each prompt individually (nnsight handles batching internally)
        generations = []
        for prompt in prompts:
            tokens = model.to_tokens(prompt)
            
            # Use nnsight's native generation (no steering)
            with model.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=model.tokenizer.eos_token_id,
            ) as generator:
                output = model.model.generator.output.save()
            
            # Decode the generated portion
            full_text = model.to_string(output[0])
            generated_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            generations.append(generated_text)
        
        return generations

    def process_batch(
        self,
        prompts: List[str],
        correct_tups: Tuple[List[str], List[str]],
        model: NNsightChatModel,
        get_activations=True,
        temperature=0.7,
        max_new_tokens=100,
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
        
        responses = [self.parse_response(response) for response in generations]
        pred_letters, pred_answers = zip(*responses)

        corrects = [
            pred == correct for pred, correct in zip(pred_letters, correct_letters)
        ]

        return activations, generations, pred_letters, pred_answers, corrects

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

        batch_size = next(
            (
                m.batch_size
                for m in self.run_config.models
                if m.name == config.model_name
            ),
            2,
        )

        # Process training data
        if not cache.has_generations():
            train_dataloader = DataLoader(
                NNsightPromptDataset(train_dataset, model), batch_size=batch_size, shuffle=False
            )
            test_dataloader = DataLoader(
                NNsightPromptDataset(test_dataset, model), batch_size=batch_size, shuffle=False
            )

            train_results, train_activations = self.process_dataset(
                train_dataloader, model, config.train_size, config
            )
            test_results, test_activations = self.process_dataset(
                test_dataloader, model, len(test_dataset), config
            )

            # Cache results
            cache.save_pickle(train_results, cache.get_train_generations_path())
            cache.save_pickle(test_results, cache.get_test_generations_path())
            cache.save_pickle(train_activations, cache.get_train_activations_path())
            cache.save_pickle(test_activations, cache.get_test_activations_path())

        return True

    def process_dataset(
        self,
        dataloader: DataLoader,
        model: NNsightChatModel,
        max_samples: int,
        config: ExperimentConfig,
    ):
        """Process entire dataset."""
        results = []
        activations_list = []
        sample_count = 0

        for prompts, correct_tups in dataloader:
            activations, generations, pred_letters, pred_answers, corrects = (
                self.process_batch(
                    prompts,
                    correct_tups,
                    model,
                    get_activations=True,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                )
            )

            sample_count += len(prompts)

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

            self.logger.info(
                f"Processed {sample_count}/{max_samples} samples. Accuracy: {np.mean(corrects):.2f}"
            )

            if sample_count >= max_samples:
                break

        return results, activations_list

    def train_and_cache_probes(
        self, model: NNsightChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Train and cache probes."""
        if cache.has_probes():
            self.logger.info(
                f"Probes already cached for {config.model_name} on {config.dataset_name}"
            )
            return True

        self.logger.info(
            f"Training probes for {config.model_name} on {config.dataset_name}"
        )

        # Load cached data
        train_results = cache.load_pickle(cache.get_train_generations_path())
        test_results = cache.load_pickle(cache.get_test_generations_path())
        train_activations = cache.load_pickle(cache.get_train_activations_path())
        test_activations = cache.load_pickle(cache.get_test_activations_path())

        layers = list(range(model.cfg.n_layers))
        all_coef_vectors = []
        auc_scores = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for layer in layers:
                train_data = self.prepare_data(
                    model, train_results, train_activations, layer
                )
                test_data = self.prepare_data(
                    model, test_results, test_activations, layer
                )

                clf = self.train_classifier(train_data)
                auc_score = self.evaluate_classifier(clf, test_data)
                auc_scores.append(auc_score)

                coef_vector = self.extract_diff_vector(clf)
                all_coef_vectors.append(coef_vector)

                self.logger.info(f"Layer {layer} AUC: {auc_score:.4f}")

        # Cache results
        cache.save_pickle(all_coef_vectors, cache.get_probe_coefficients_path())
        cache.save_json(auc_scores, cache.get_auc_scores_path())

        # Visualization removed - AUC scores logged above

        best_layer = layers[np.argmax(auc_scores)]
        best_auc = max(auc_scores)
        self.logger.info(
            f"Best AUC: {best_auc:.4f} at layer {best_layer} for {config.model_name} on {config.dataset_name}"
        )

        return True

    def prepare_data(
        self, model: NNsightChatModel, results: List[Dict], activations: List, layer: int
    ):
        """Prepare data for training classifier."""
        data = []
        for idx, result in enumerate(results):
            if result["pred_answer"] == result["correct_answer"]:
                activation = activations[idx][layer]
                data.append(activation.tolist() + [result["pred_answer"]])

        df = pd.DataFrame(
            data,
            columns=["ac" + str(i) for i in range(model.cfg.d_model)] + ["pred"],
        )
        df = df[df["pred"].isin(["yes", "no"])]
        return df

    def train_classifier(self, train_data: pd.DataFrame):
        """Train logistic regression classifier."""
        X = train_data[[col for col in train_data.columns if col.startswith("ac")]]
        y = train_data["pred"]
        return LogisticRegression(random_state=0, max_iter=1000).fit(X, y)

    def evaluate_classifier(self, clf, test_data: pd.DataFrame):
        """Evaluate classifier performance."""
        X = test_data[[col for col in test_data.columns if col.startswith("ac")]]
        y = test_data["pred"]
        y = y.apply(lambda x: 1 if x == "yes" else 0)
        try:
            return roc_auc_score(y, clf.predict_proba(X)[:, 1])
        except ValueError:
            return 0.5

    def extract_diff_vector(self, clf):
        """Extract coefficient vector from classifier."""
        return clf.coef_[0]

    def run_steering_experiments(
        self, model: NNsightChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Run steering experiments using nnsight."""
        self.logger.info(
            f"Running steering for {config.model_name} on {config.dataset_name}"
        )

        # Load cached data
        test_results = cache.load_pickle(cache.get_test_generations_path())
        all_coef_vectors = cache.load_pickle(cache.get_probe_coefficients_path())

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
            if not cache.has_steering_results(alpha_yes, "yes"):
                results_yes = self.generate_steered_examples(
                    model, yes_test_data, all_coef_vectors, layers, alpha_yes, config
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
                        f"Alpha {alpha_yes:+.1f} (yes): {success_rate:.2f} success, "
                        f"{failure_rate:.2f} failure, {unparsed_rate:.2f} unparsed"
                    )
                else:
                    success_rate = 0
                    self.logger.info(f"Alpha {alpha_yes:+.1f} (yes): No results")

                # Visualization removed - results logged above

            # No to Yes steering
            alpha_no = abs(alpha)  # Positive alpha steers "no" to "yes"
            if not cache.has_steering_results(alpha_no, "no"):
                results_no = self.generate_steered_examples(
                    model, no_test_data, all_coef_vectors, layers, alpha_no, config
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
                        f"Alpha {alpha_no:+.1f} (no): {success_rate:.2f} success, "
                        f"{failure_rate:.2f} failure, {unparsed_rate:.2f} unparsed"
                    )
                else:
                    success_rate = 0
                    self.logger.info(f"Alpha {alpha_no:+.1f} (no): No results")

                # Visualization removed - results logged above

        return True

    def generate_steered_examples(
        self,
        model: NNsightChatModel,
        test_data: List[Dict],
        all_coef_vectors: List,
        layers: List[int],
        alpha: float,
        config: ExperimentConfig,
    ):
        """Generate steered examples using nnsight."""
        steered_results = []
        
        # Note: NNsight implementation currently only supports logistic regression
        # TODO: Add support for CAA steering methods
        self.logger.warning(
            "NNsight implementation uses logistic regression only. "
            "CAA single-layer steering is not yet implemented for nnsight."
        )

        for i, example in enumerate(test_data):
            example_prompt = example["prompt"]
            tokens = model.to_tokens(example_prompt)
            instruction_pos = tokens.size(1)  # End of prompt

            # Generate with steering using nnsight
            generation = generate_with_nnsight_steering(
                model=model,
                tokens=tokens,
                steering_vectors=np.array(all_coef_vectors),
                alpha=alpha,
                instruction_pos=instruction_pos,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                layers=layers,
            )

            new_letter, new_answer = self.parse_response(generation)
            orig = example["pred_answer"]
            
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

            steered_results.append(
                {
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
            )

        return steered_results

    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        cache = self.exp_manager.add_experiment(config)
        exp_key = f"{config.model_name}_{config.dataset_name}"

        try:
            # Load model using nnsight
            self.logger.info(f"Loading model: {config.model_name}")
            model = NNsightChatModel(config.model_name)

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
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

    def run_all_experiments(self):
        """Run all experiments."""
        self.logger.info(f"Starting {len(self.experiment_configs)} experiments")

        # Initialize experiment status
        for config in self.experiment_configs:
            exp_key = f"{config.model_name}_{config.dataset_name}"
            cache = self.exp_manager.add_experiment(config)
            self.experiments_status[exp_key] = cache.get_experiment_status()

        # Run experiments without visualization
        self._run_experiments_simple()

    def _run_experiments_simple(self):
        """Run experiments."""
        for config in self.experiment_configs:
            exp_key = f"{config.model_name}_{config.dataset_name}"

            result = self.run_single_experiment(config)

            if result["success"]:
                self.experiments_status[exp_key] = result["status"]

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