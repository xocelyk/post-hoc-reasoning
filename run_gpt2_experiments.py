#!/usr/bin/env python3
"""
GPT-2 specific experiment runner for steering experiments.
Adapts the main experiment framework to work with GPT-2 small which doesn't have chat templates.
"""

import os
import sys
import gc
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, List, Tuple
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loading import load_all_datasets
from utils import generate_with_hooks
from transformer_lens import HookedTransformer


class GPT2ChatModel:
    """Wrapper for GPT-2 that mimics the ChatModel interface but handles non-chat formatting."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
        )
    
    def format_prompt_for_gpt2(self, chat_messages: List[Dict[str, str]]) -> str:
        """Convert chat format to a simple text format suitable for GPT-2."""
        formatted_parts = []
        
        for message in chat_messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                # Format user messages as questions
                formatted_parts.append(f"Q: {content}")
            elif role == "assistant":
                # Format assistant messages as answers
                formatted_parts.append(f"A: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def __getattr__(self, attr):
        # Delegate to the underlying HookedTransformer
        return getattr(self.model, attr)


class GPT2PromptDataset(Dataset):
    """Dataset wrapper for GPT-2 prompts (converted from chat format)."""
    
    def __init__(self, data: List[Dict[str, Any]], model: GPT2ChatModel):
        self.data = []
        self.model = model
        
        for item in data:
            # Convert chat format to GPT-2 text format
            prompt_text = model.format_prompt_for_gpt2(item["prompt"])
            
            self.data.append({
                "prompt": prompt_text,
                "correct_answer": item["correct_answer"],
                "correct_letter": item["correct_letter"]
            })
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[str, Tuple[str, str]]:
        item = self.data[idx]
        return item["prompt"], (item["correct_answer"], item["correct_letter"])


class GPT2ExperimentRunner:
    """Simplified experiment runner for GPT-2 experiments."""
    
    def __init__(self, cache_dir: str = "cache_gpt2"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.cache_dir, "gpt2_experiment.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("GPT2ExperimentRunner")
    
    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse GPT-2 response to extract answer."""
        response = response.strip()
        
        # Look for pattern like "The best answer is: (A)"
        import re
        
        # Try to find letter in parentheses
        letter_match = re.search(r'\(([AB])\)', response)
        if letter_match:
            letter = letter_match.group(1)
            
            # Try to extract yes/no from context
            if "yes" in response.lower():
                text_answer = "yes"
            elif "no" in response.lower():  
                text_answer = "no"
            else:
                text_answer = ""
            
            return letter, text_answer
        
        # Fallback: look for yes/no directly
        if "yes" in response.lower() and "no" not in response.lower():
            return "A", "yes"  # Assume A is yes
        elif "no" in response.lower() and "yes" not in response.lower():
            return "B", "no"   # Assume B is no
        
        return "", ""
    
    def batch_get_resid_activations(self, prompts: List[str], model: GPT2ChatModel):
        """Get residual stream activations for a batch of prompts."""
        layers = list(range(model.cfg.n_layers))
        tokens = model.to_tokens(prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, pos_slice=-1)
        
        activations = np.zeros((len(prompts), model.cfg.n_layers, model.cfg.d_model))
        
        for layer in layers:
            layer_activations = cache["resid_post", layer]
            layer_activations = layer_activations.squeeze().detach().cpu().numpy()
            activations[:, layer, :] = layer_activations
            del layer_activations
            torch.cuda.empty_cache()
            gc.collect()
        
        return activations
    
    def batch_get_generations(self, prompts: List[str], model: GPT2ChatModel, 
                            temperature=0.7, max_new_tokens=100):
        """Get generations for a batch of prompts."""
        tokens = model.to_tokens(prompts, prepend_bos=True)
        token_generations = model.generate(
            tokens, max_new_tokens=max_new_tokens, temperature=temperature
        )
        generations = model.to_string(token_generations)
        return generations
    
    def process_batch(self, prompts: List[str], correct_tups: Tuple[List[str], List[str]], 
                     model: GPT2ChatModel, get_activations=True, temperature=0.7, max_new_tokens=100):
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
        generations = [gen[len(prompt) + 12:] for gen, prompt in zip(generations, prompts)]
        generations = [gen[:gen.find("\n")] for gen in generations]
        
        responses = [self.parse_response(response) for response in generations]
        pred_letters, pred_answers = zip(*responses)
        for i, gen in enumerate(generations):
            print(f"{i+1}. {gen}")
            print(f"{responses[i]}")
            print()
        
        corrects = [
            pred == correct for pred, correct in zip(pred_letters, correct_letters)
        ]
        
        return activations, generations, pred_letters, pred_answers, corrects
    
    def process_dataset(self, dataloader: DataLoader, model: GPT2ChatModel, max_samples: int):
        """Process entire dataset."""
        results = []
        activations_list = []
        sample_count = 0
        
        for prompts, correct_tups in dataloader:
            activations, generations, pred_letters, pred_answers, corrects = (
                self.process_batch(prompts, correct_tups, model, get_activations=True)
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
            
            accuracy = np.mean(corrects)
            self.logger.info(f"Processed {sample_count}/{max_samples} samples. Accuracy: {accuracy:.2f}")
            
            if sample_count >= max_samples:
                break
        
        return results, activations_list
    
    def prepare_data(self, model: GPT2ChatModel, results: List[Dict], activations: List, layer: int):
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
    
    def generate_steered_examples(self, model: GPT2ChatModel, test_data: List[Dict], 
                                all_coef_vectors: List, layers: List[int], alpha: float):
        """Generate steered examples."""
        steered_results = []
        
        for example in test_data:
            example_prompt = example["prompt"]
            example_tokens = model.to_tokens(example_prompt, prepend_bos=False)
            
            generation = generate_with_hooks(
                model,
                example_tokens,
                temperature=0.7,
                max_new_tokens=100,
                alpha=alpha,
                steering_vectors=np.array(all_coef_vectors),
                layers=layers,
            )
            
            new_letter, new_answer = self.parse_response(generation)
            orig = example["pred_answer"]
            success = (orig == "yes" and new_answer == "no") or (orig == "no" and new_answer == "yes")
            
            steered_results.append({
                "original_prompt": example_prompt,
                "steered_generation": generation,
                "original_answer": orig,
                "new_answer": new_answer,
                "original_letter": example["pred_letter"],
                "new_letter": new_letter,
                "alpha": alpha,
                "success": success,
            })
        
        return steered_results
    
    def run_experiment(self, dataset_name: str, train_size: int = 50, test_size: int = 200,
                      alpha_range: List[float] = [0, 1, 2, 3]):
        """Run complete experiment for a dataset."""
        self.logger.info(f"Starting experiment for {dataset_name}")
        
        # Load model
        model = GPT2ChatModel("gpt2", device="cpu")
        
        # Load dataset
        datasets = load_all_datasets()
        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset = datasets[dataset_name]
        
        # Train/test split
        train_dataset, test_dataset = train_test_split(
            dataset, train_size=train_size, test_size=test_size, random_state=42
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            GPT2PromptDataset(train_dataset, model), batch_size=2, shuffle=False
        )
        test_dataloader = DataLoader(
            GPT2PromptDataset(test_dataset, model), batch_size=2, shuffle=False
        )
        
        # Process datasets
        self.logger.info("Processing training data...")
        train_results, train_activations = self.process_dataset(
            train_dataloader, model, train_size
        )
        
        self.logger.info("Processing test data...")
        test_results, test_activations = self.process_dataset(
            test_dataloader, model, test_size
        )
        
        # Train probes
        self.logger.info("Training probes...")
        layers = list(range(min(6, model.cfg.n_layers)))  # Use first 6 layers for speed
        all_coef_vectors = []
        auc_scores = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for layer in layers:
                train_data = self.prepare_data(model, train_results, train_activations, layer)
                test_data = self.prepare_data(model, test_results, test_activations, layer)
                
                if len(train_data) < 10 or len(test_data) < 10:
                    self.logger.warning(f"Insufficient data for layer {layer}")
                    auc_scores.append(0.5)
                    all_coef_vectors.append(np.zeros(model.cfg.d_model))
                    continue
                
                clf = self.train_classifier(train_data)
                auc_score = self.evaluate_classifier(clf, test_data)
                auc_scores.append(auc_score)
                all_coef_vectors.append(clf.coef_[0])
                
                self.logger.info(f"Layer {layer} AUC: {auc_score:.4f}")
        
        best_layer = layers[np.argmax(auc_scores)]
        best_auc = max(auc_scores)
        self.logger.info(f"Best AUC: {best_auc:.4f} at layer {best_layer}")
        
        # Run steering experiments
        self.logger.info("Running steering experiments...")
        
        # Create test subsets for steering
        yes_test_data = [
            result for result in test_results
            if result["pred_answer"] == "yes" and result["correct_answer"] == "yes"
        ]
        no_test_data = [
            result for result in test_results
            if result["pred_answer"] == "no" and result["correct_answer"] == "no"
        ]
        
        steering_results = {}
        
        for alpha in alpha_range:
            # Yes to No steering
            if yes_test_data:
                alpha_yes = -abs(alpha)
                results_yes = self.generate_steered_examples(
                    model, yes_test_data[:10], all_coef_vectors, layers, alpha_yes
                )
                success_rate = sum(r["success"] for r in results_yes) / len(results_yes)
                steering_results[f"alpha_{alpha_yes}_yes"] = success_rate
                self.logger.info(f"Alpha {alpha_yes:+.1f} (yes->no): {success_rate:.2f} success rate")
            
            # No to Yes steering  
            if no_test_data:
                alpha_no = abs(alpha)
                results_no = self.generate_steered_examples(
                    model, no_test_data[:10], all_coef_vectors, layers, alpha_no
                )
                success_rate = sum(r["success"] for r in results_no) / len(results_no)
                steering_results[f"alpha_{alpha_no}_no"] = success_rate
                self.logger.info(f"Alpha {alpha_no:+.1f} (no->yes): {success_rate:.2f} success rate")
        
        # Save results
        results_file = os.path.join(self.cache_dir, f"{dataset_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "dataset": dataset_name,
                "auc_scores": auc_scores,
                "best_layer": best_layer,
                "best_auc": best_auc,
                "steering_results": steering_results
            }, f, indent=2)
        
        self.logger.info(f"Experiment completed. Results saved to {results_file}")
        return steering_results


def main():
    """Main entry point."""
    runner = GPT2ExperimentRunner()
    
    # Run experiment on sports_understanding dataset
    dataset_name = "sports_understanding"
    
    try:
        results = runner.run_experiment(
            dataset_name=dataset_name,
            train_size=50,      # Small for testing
            test_size=100,      # Small for testing
            alpha_range=[0, 1, 2]  # Small range for testing
        )
        
        print("\n=== FINAL RESULTS ===")
        for key, value in results.items():
            print(f"{key}: {value:.3f}")
            
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()