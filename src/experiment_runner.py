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
from models import ChatModel
from utils import generate_with_hooks
from visualizer import create_visualizer


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
        response = (
            response.strip()
            .replace("<eos>", "")
            .replace("<pad>", "")
            .replace("<end_of_turn>", "")
            .strip()
        )
        start_answer_string = "the best answer is:"
        if start_answer_string not in response.lower():
            return "", ""
        answer_part = response.split(start_answer_string)[-1]

        import re

        letter_match = re.search(r"\((.)\)", answer_part)
        if not letter_match:
            return "", ""
        letter = letter_match.group(1)
        text_answer = (
            answer_part.split(")")[-1]
            .strip()
            .split(", ")[0]
            .lower()
            .replace(".", "")
            .strip()
        )
        return letter, text_answer

    def batch_get_resid_activations(self, prompts: List[str], model: ChatModel):
        """Get residual stream activations for a batch of prompts."""
        print(f"DEBUG: batch_get_resid_activations with {len(prompts)} prompts")
        layers = list(range(model.cfg.n_layers))
        print(f"DEBUG: Model has {model.cfg.n_layers} layers")
        tokens = model.to_tokens(prompts, prepend_bos=True)
        print(f"DEBUG: Tokenized to shape {tokens.shape}")
        print("DEBUG: Running model.run_with_cache...")
        _, cache = model.run_with_cache(tokens, pos_slice=-1)
        print("DEBUG: Finished model.run_with_cache")

        activations = np.zeros((len(prompts), model.cfg.n_layers, model.cfg.d_model))

        for layer in layers:
            layer_activations = cache["resid_post", layer]
            # Convert to float32 before converting to numpy to avoid BFloat16 issues on MPS
            layer_activations = layer_activations.squeeze().detach().float().cpu().numpy()
            activations[:, layer, :] = layer_activations
            del layer_activations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        return activations

    def batch_get_generations(
        self, prompts: List[str], model: ChatModel, temperature=0.7, max_new_tokens=100
    ):
        """Get generations for a batch of prompts."""
        print(f"DEBUG: batch_get_generations with {len(prompts)} prompts, max_new_tokens={max_new_tokens}")
        tokens = model.to_tokens(prompts, prepend_bos=True)
        print(f"DEBUG: Tokenized to shape {tokens.shape}")
        print("DEBUG: Starting model.generate...")
        token_generations = model.generate(
            tokens, max_new_tokens=max_new_tokens, temperature=temperature, verbose=True
        )
        print("DEBUG: Finished model.generate")
        generations = model.to_string(token_generations)
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
        print(f"DEBUG: process_batch called with {len(prompts)} prompts")
        correct_answers, correct_letters = correct_tups

        print("DEBUG: Starting batch_get_resid_activations...")
        activations = (
            self.batch_get_resid_activations(prompts, model)
            if get_activations
            else None
        )
        print("DEBUG: Finished batch_get_resid_activations")
        
        print("DEBUG: Starting batch_get_generations...")
        generations = self.batch_get_generations(
            prompts, model, temperature=temperature, max_new_tokens=max_new_tokens
        )
        print("DEBUG: Finished batch_get_generations")
        generations = [gen[len(prompt) :] for gen, prompt in zip(generations, prompts)]

        print("DEBUG: Parsing responses...")
        responses = [self.parse_response(response) for response in generations]
        pred_letters, pred_answers = zip(*responses)

        corrects = [
            pred == correct for pred, correct in zip(pred_letters, correct_letters)
        ]

        # Log details for first batch
        if log_first_batch:
            print("\n" + "="*80)
            print("FIRST BATCH DETAILED LOGGING")
            print("="*80)
            for i, prompt in enumerate(prompts):
                print(f"\n--- SAMPLE {i+1} ---")
                print("PROMPT:")
                print(prompt)
                print(f"\nGENERATED RESPONSE:")
                print(repr(generations[i]))
                print(f"\nPARSED LETTER: {pred_letters[i]}")
                print(f"PARSED ANSWER: {pred_answers[i]}")
                print(f"CORRECT LETTER: {correct_letters[i]}")
                print(f"CORRECT ANSWER: {correct_answers[i]}")
                print(f"CORRECT: {corrects[i]}")
            print("="*80 + "\n")

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

        print("DEBUG: Train size:", len(train_dataset))
        print("DEBUG: Test size:", len(test_dataset))
        
        batch_size = next(
            (
                m.batch_size
                for m in self.run_config.models
                if m.name == config.model_name
            ),
            2,
        )
        print(f"DEBUG: Using batch size: {batch_size}")

        # Process training data
        if not (self.run_config.use_cache and cache.has_generations()):
            print("DEBUG: Creating train dataloader...")
            train_dataloader = DataLoader(
                PromptDataset(train_dataset, model), batch_size=batch_size, shuffle=False
            )
            print("DEBUG: Creating test dataloader...")
            test_dataloader = DataLoader(
                PromptDataset(test_dataset, model), batch_size=batch_size, shuffle=False
            )

            print("DEBUG: Starting to process training dataset...")
            train_results, train_activations = self.process_dataset(
                train_dataloader, model, config.train_size, config
            )
            print("DEBUG: Starting to process test dataset...")
            test_results, test_activations = self.process_dataset(
                test_dataloader, model, len(test_dataset), config
            )

            # Cache results
            if self.run_config.use_cache:
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
    ):
        """Process entire dataset."""
        print(f"DEBUG: process_dataset called with max_samples={max_samples}")
        print(f"DEBUG: Dataloader length: {len(dataloader)}")
        results = []
        activations_list = []
        sample_count = 0

        for batch_idx, (prompts, correct_tups) in enumerate(dataloader):
            print(f"DEBUG: Processing batch {batch_idx + 1}/{len(dataloader)} with {len(prompts)} prompts")
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
        self, model: ChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Train and cache probes."""
        if self.run_config.use_cache and cache.has_probes():
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
        if self.run_config.use_cache:
            cache.save_pickle(all_coef_vectors, cache.get_probe_coefficients_path())
            cache.save_json(auc_scores, cache.get_auc_scores_path())

        # Update visualizer
        if hasattr(self.visualizer, "update_auc_scores"):
            self.visualizer.update_auc_scores(
                config.model_name, config.dataset_name, auc_scores
            )

        best_layer = layers[np.argmax(auc_scores)]
        best_auc = max(auc_scores)
        self.logger.info(
            f"Best AUC: {best_auc:.4f} at layer {best_layer} for {config.model_name} on {config.dataset_name}"
        )

        return True

    def prepare_data(
        self, model: ChatModel, results: List[Dict], activations: List, layer: int
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
        self, model: ChatModel, config: ExperimentConfig, cache: ExperimentCache
    ) -> bool:
        """Run steering experiments."""
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

        layers = list(range(model.cfg.n_layers))

        for alpha in config.alpha_range:
            # Yes to No steering
            alpha_yes = -abs(alpha)  # Negative alpha steers "yes" to "no"
            if not (self.run_config.use_cache and cache.has_steering_results(alpha_yes, "yes")):
                results_yes = self.generate_steered_examples(
                    model, yes_test_data, all_coef_vectors, layers, alpha_yes, config
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
                    model, no_test_data, all_coef_vectors, layers, alpha_no, config
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
        all_coef_vectors: List,
        layers: List[int],
        alpha: float,
        config: ExperimentConfig,
    ):
        """Generate steered examples."""
        steered_results = []

        for i, example in enumerate(test_data):
            example_prompt = example["prompt"]
            example_tokens = model.to_tokens(example_prompt, prepend_bos=False)

            generation = generate_with_hooks(
                model,
                example_tokens,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                alpha=alpha,
                steering_vectors=np.array(all_coef_vectors),
                layers=layers,
            )

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
