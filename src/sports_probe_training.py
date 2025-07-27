# %% Import necessary libraries
import gc
import json
import os
import pickle
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset

load_dotenv()
from matplotlib import pyplot as plt
from transformer_lens.utils import Slice

from data_loading import create_cot_dataset, create_dataset
from models import ChatModel
from utils import generate_with_hooks

THINKING = True
DATASET_NAME = "logical_deduction"


# %%
def load_model():
    """Load and configure the model"""
    model = ChatModel(
        "google/gemma-2-9b-it",
        device="cuda",
        n_devices=2,
        cache_dir=os.environ["HF_HOME"],
    )
    print(f"Model loaded: {model.model_name}")
    print(f"Number of layers: {model.cfg.n_layers}")
    print(f"Model dimension: {model.cfg.d_model}")
    return model


def load_dataset(train_size=100):
    """Load and split the sports understanding dataset"""
    print("Loading sports understanding dataset...")
    examples = create_dataset(DATASET_NAME)
    cot_dataset = create_cot_dataset(DATASET_NAME, examples, thinking=THINKING)
    print(f"Loaded {len(cot_dataset)} examples")

    # Split into train and test
    train_dataset, test_dataset = train_test_split(
        cot_dataset, train_size=train_size, random_state=42
    )
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, test_dataset


# Load model and dataset
model = load_model()
train_dataset, test_dataset = load_dataset()


# %% Function to extract residual activations
def get_resid_activations(prompts, model, batch_size=1):
    """Extract residual activations from all layers for given prompts in batches"""
    layers = list(range(model.cfg.n_layers))
    all_activations = []

    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, pos_slice=-1)

        batch_activations = torch.zeros(
            (len(batch_prompts), model.cfg.n_layers, model.cfg.d_model)
        )

        for layer in layers:
            layer_activations = cache["resid_post", layer]
            layer_activations = layer_activations.squeeze().detach().cpu()
            batch_activations[:, layer, :] = layer_activations
            del layer_activations
            torch.cuda.empty_cache()
            gc.collect()

        all_activations.append(batch_activations)

        # Clear cache after each batch to save memory
        del cache, tokens
        torch.cuda.empty_cache()
        gc.collect()

    # Concatenate all batches
    activations = np.concatenate(all_activations, axis=0)
    return activations


# %%
# Helper functions
def parse_response(response: str, thinking: bool = True) -> Tuple[str, str]:
    # TODO: Make more robust; this only works for gemma
    response = (
        response.strip()
        .replace("<eos>", "")
        .replace("<pad>", "")
        .replace("<end_of_turn>", "")
        .strip()
    )
    if thinking:
        start_answer_string = "the best answer is:"
        if start_answer_string not in response.lower():
            return "", ""
        answer_part = response.split(start_answer_string)[-1]
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
    else:
        letter = "A" if "(A)" in response else "B"
        text_answer = "yes" if "yes" in response.lower() else "no"
    return letter, text_answer


# %%
def format_prompt(model, prompt) -> str:
    prompt_replaced = []
    last_msg = None
    for msg in prompt:
        if last_msg is None:
            last_msg = deepcopy(msg)
            continue
        if last_msg["role"] == msg["role"]:
            last_msg["content"] += "\n" + msg["content"]
        else:
            prompt_replaced.append(last_msg)
            last_msg = deepcopy(msg)
    prompt_replaced.append(last_msg)
    return model.apply_chat_template(prompt_replaced)


# %% Function to generate model predictions
def generate_predictions(dataset, model, temperature=0.7, max_new_tokens=100):
    """Generate predictions for given prompts"""
    prompts = [format_prompt(model, item["prompt"]) for item in dataset]
    predictions = []

    # Process one prompt at a time to minimize memory usage
    for i, prompt in enumerate(prompts):
        # Generate tokens and prediction for single prompt
        tokens = model.to_tokens([prompt], prepend_bos=True)
        generation = model.generate(
            tokens[:, :-2],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

        # Extract response
        response = model.tokenizer.decode(generation[0][tokens.shape[1] - 2 :])
        letter, text_answer = parse_response(response, thinking=THINKING)
        print(f"Response {i}: {response}")

        predictions.append(
            {
                "prompt": prompt,
                "response": response,
                "pred_letter": letter,
                "pred_answer": text_answer,
                "correct_letter": dataset[i]["correct_letter"],
                "correct_answer": dataset[i]["correct_answer"],
            }
        )

        # Clean up tensors
        del tokens, generation
        torch.cuda.empty_cache()
        gc.collect()

    return predictions


# %% Function to extract activations for a single layer (memory efficient)
@torch.inference_mode()
def get_layer_activations(prompts, model, layer, batch_size=1):
    """Extract activations for a single layer to save memory"""
    all_activations = []

    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, pos_slice=-1)

        # Extract only the specified layer
        layer_activations = cache["resid_post", layer]
        layer_activations = layer_activations.squeeze().detach().cpu()
        all_activations.append(layer_activations)

        # Clear cache immediately
        del cache, tokens, layer_activations
        torch.cuda.empty_cache()
        gc.collect()

    # Concatenate all batches
    activations = torch.stack(all_activations, dim=0)
    return activations


def prepare_probe_data_layer(results, dataset, model, layer, batch_size=1):
    """Prepare data for training a probe on a specific layer - memory efficient"""
    data = []
    prompts = [format_prompt(model, item["prompt"]) for item in dataset]
    # Get activations for this layer only
    layer_activations = get_layer_activations(prompts, model, layer, batch_size)

    for idx, result in enumerate(results):
        activation = layer_activations[idx]
        data.append(activation.tolist() + [result["pred_answer"]])

    # Clear activations to save memory
    del layer_activations
    torch.cuda.empty_cache()
    gc.collect()

    df = pd.DataFrame(
        data, columns=pd.Index([f"ac{i}" for i in range(model.cfg.d_model)] + ["pred"])
    )
    df = df[df["pred"].isin(["yes", "no"])]
    return df


# Function to train a probe
def train_probe(train_data):
    """Train a logistic regression probe"""
    X = train_data[[col for col in train_data.columns if col.startswith("ac")]]
    y = train_data["pred"]
    return LogisticRegression(random_state=0).fit(X, y)


# Function to evaluate a probe
def evaluate_probe(clf, test_data):
    """Evaluate probe using AUROC"""
    X = test_data[[col for col in test_data.columns if col.startswith("ac")]]
    y = test_data["pred"]
    y = y.apply(lambda x: 1 if x == "yes" else 0)
    try:
        return roc_auc_score(y, clf.predict_proba(X)[:, 1])
    except ValueError:
        return 0


# Function to extract coefficient vector
def extract_coef_vector(clf):
    """Extract coefficient vector from trained probe"""
    return clf.coef_[0]


def train_probes_all_layers(model, train_dataset, test_dataset, run_name: str = "2"):
    """Train probes for all layers of the model and return results"""

    save_dir = f"../results/probes/{model.model_name}/{run_name}/{DATASET_NAME}/"
    os.makedirs(save_dir, exist_ok=True)
    train_predictions = generate_predictions(train_dataset, model)
    test_predictions = generate_predictions(test_dataset, model)

    print("Training probes for all layers...")
    layers = list(range(model.cfg.n_layers))
    all_probes = []
    all_coef_vectors = []
    auc_scores = []

    for layer in layers:
        print(f"\nTraining probe for layer {layer}...")

        # Prepare data for this layer
        train_data = prepare_probe_data_layer(
            train_predictions, train_dataset, model, layer
        )
        test_data = prepare_probe_data_layer(
            test_predictions, test_dataset, model, layer
        )

        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"  Skipping layer {layer} - insufficient data")
            all_probes.append(None)
            all_coef_vectors.append(None)
            auc_scores.append(0)
            continue

        # Train probe
        clf = train_probe(train_data)

        # Evaluate probe
        auc_score = evaluate_probe(clf, test_data)
        auc_scores.append(auc_score)

        # Extract coefficient vector
        coef_vector = extract_coef_vector(clf)

        print(f"  AUROC: {auc_score:.4f}")

        all_probes.append(clf)
        all_coef_vectors.append(coef_vector)

    print(f"\nProbe training completed for {len(layers)} layers")

    # Save train and test datasets
    print("\nSaving datasets...")
    dataset_path = os.path.join(save_dir, "datasets.pkl")
    datasets = {
        "train_data": train_dataset,
        "test_data": test_dataset,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
    }
    with open(dataset_path, "wb") as f:
        pickle.dump(datasets, f)
    print(f"Datasets saved to {dataset_path}")

    print("\nSaving probes and results...")
    os.makedirs(save_dir, exist_ok=True)

    # Save probes using pickle
    probe_path = os.path.join(save_dir, f"probes.pkl")
    with open(probe_path, "wb") as f:
        pickle.dump(all_probes, f)
    print(f"Probes saved to {probe_path}")

    # Save AUC scores as JSON
    scores_path = os.path.join(save_dir, f"auc_scores.json")
    scores_data = {
        "layer_scores": {str(layer): score for layer, score in zip(layers, auc_scores)},
        "best_layer": int(layers[np.argmax(auc_scores)]),
        "best_score": float(np.max(auc_scores)),
    }
    with open(scores_path, "w") as f:
        json.dump(scores_data, f, indent=2)
    print(f"AUC scores saved to {scores_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(layers, auc_scores, marker="o")
    plt.grid(True)
    plt.xlabel("Layer")
    plt.ylabel("AUROC Score")
    plt.title("AUROC Scores by Layer")
    plt.ylim(0.0, 1.0)

    # Add horizontal lines for mean and best scores
    mean_score = float(np.mean(auc_scores))
    max_score = float(np.max(auc_scores))
    plt.axhline(
        y=mean_score, color="r", linestyle="--", label=f"Mean ({mean_score:.3f})"
    )
    plt.axhline(y=max_score, color="g", linestyle="--", label=f"Best ({max_score:.3f})")

    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, f"auc_scores.png")
    plt.savefig(plot_path)
    print(f"AUC scores plot saved to {plot_path}")
    plt.close()

    return all_probes, all_coef_vectors, auc_scores


def load_probes(model, run_name: str = "2"):
    save_dir = f"../results/probes/{model.model_name}/{run_name}/{DATASET_NAME}/"
    probe_path = os.path.join(save_dir, "probes.pkl")
    with open(probe_path, "rb") as f:
        all_probes = pickle.load(f)
    return all_probes


def load_probes_from_position(
    model, run_name: str = "2", position=None, use_all_positions=False
):
    """Load probes from position-specific directory"""
    save_dir = f"../results/probes/{model.model_name}/{run_name}/{DATASET_NAME}/"

    # Determine position-specific directory
    if use_all_positions:
        pos_suffix = "all_positions"
    elif position is not None:
        pos_suffix = f"position_{position}"
    else:
        pos_suffix = "last_position"

    save_dir = os.path.join(save_dir, pos_suffix)
    probe_path = os.path.join(save_dir, "probes.pkl")

    if not os.path.exists(probe_path):
        raise FileNotFoundError(
            f"Probes not found at {probe_path}. Train probes first."
        )

    with open(probe_path, "rb") as f:
        all_probes = pickle.load(f)
    return all_probes


# %%
def save_residuals_over_time(model, dataset, run_name: str = "2"):
    """Save residual activations over time for each example in the dataset"""
    save_dir = (
        f"../results/probes/{model.model_name}/{run_name}/{DATASET_NAME}/residuals/"
    )
    os.makedirs(save_dir, exist_ok=True)

    print("Generating residual activations over time...")
    for i, example in enumerate(dataset):
        print(f"\nProcessing example {i+1}/{len(dataset)}")

        # Format prompt and generate response
        prompt = format_prompt(model, example["prompt"])
        tokens = model.to_tokens([prompt], prepend_bos=True)

        # Generate response and get cache
        generation = model.generate(
            tokens[:, :-2],
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

        # Run with cache to get activations
        with torch.inference_mode():
            _, cache = model.run_with_cache(generation)

        # Extract residual activations for each layer
        residuals = {}
        for layer in range(model.cfg.n_layers):
            layer_activations = cache["resid_post", layer]
            layer_activations = layer_activations.squeeze().detach().cpu()
            residuals[f"layer_{layer}"] = layer_activations

        # Accumulate data for all examples
        if i == 0:
            all_data = {
                "prompts": [prompt],
                "responses": [model.tokenizer.decode(generation[0])],
                "residuals": [residuals],
            }
        else:
            all_data["prompts"].append(prompt)
            all_data["responses"].append(model.tokenizer.decode(generation[0]))
            all_data["residuals"].append(residuals)

        # Save accumulated data after processing each example
        data_path = os.path.join(save_dir, "all_residuals.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(all_data, f)

        # Clean up to save memory
        del tokens, generation, cache, residuals
        torch.cuda.empty_cache()
        gc.collect()

    print("\nResidual activations saved successfully")


# train_probes_all_layers(model, train_dataset, test_dataset, run_name="5")
# save_residuals_over_time(model, train_dataset, run_name="5")
# %%
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict], model: ChatModel):
        self.data = data
        self.model = model

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        item = self.data[idx]
        prompt = format_prompt(self.model, item["prompt"])
        return prompt, item["correct_answer"], item["correct_letter"]


# %%
def right_to_left_pad(tokens, model):
    """Move padding tokens from right to left side of sequences"""
    # Get number of padding tokens at end of each sequence
    pad_lens = (tokens == model.tokenizer.pad_token_id).sum(dim=1)

    # Create mask of padding tokens same shape as input
    pad_mask = torch.full_like(tokens, model.tokenizer.pad_token_id)

    # Create position indices for each sequence
    positions = (
        torch.arange(tokens.shape[1], device=tokens.device)
        .unsqueeze(0)
        .expand_as(tokens)
    )

    # Roll tokens left by padding length to move padding to front
    shifted_tokens = torch.stack(
        [torch.roll(seq, shift.item()) for seq, shift in zip(tokens, pad_lens)]
    )
    return shifted_tokens


# %%
# %%
def generate_and_save_predictions(
    model, test_dataset, test_loader, all_probes, run_name
):
    """Generate predictions using probes and model, and save results"""
    preds_per_layer = defaultdict(lambda: np.empty((0, 2)))
    model_preds = []

    for batch in test_loader:
        tokens = model.to_tokens(batch[0], prepend_bos=True)
        tokens = right_to_left_pad(tokens, model)

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, pos_slice=-1)

        # Get probe predictions for each layer
        for layer_num in range(model.cfg.n_layers):
            probe = all_probes[layer_num]
            activations = cache["resid_post", layer_num]
            activations = activations[:, 0, :].cpu().float()
            pred = probe.predict_proba(activations)
            preds_per_layer[layer_num] = np.concatenate(
                [preds_per_layer[layer_num], pred], axis=0
            )

        # Generate model predictions
        generation_tokens = model.generate(
            tokens[:, :-2], max_new_tokens=100, temperature=0.7, do_sample=True
        )
        response = model.to_string(generation_tokens)
        letters, text_answers = zip(
            *[parse_response(r, thinking=THINKING) for r in response]
        )
        model_preds.extend(text_answers)

    # Get test answers
    test_answers = [item["correct_answer"] for item in test_dataset][: len(model_preds)]

    results = {
        "model_predictions": model_preds,
        "probe_predictions_per_layer": dict(preds_per_layer),
        "test_answers": test_answers,
    }

    # Save results
    save_dir = f"../results/predictions/{model.model_name}/{run_name}/{DATASET_NAME}"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "predictions.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"Saved predictions to {save_dir}/predictions.pkl")
    return results


# Generate and save predictions
# data = TestDataset(test_dataset, model)
# all_probes = load_probes(model, run_name="5")
# test_loader = DataLoader(data, batch_size=5, shuffle=False)
# results = generate_and_save_predictions(model, test_dataset, test_loader, all_probes, run_name="6")
# %%
# Load predictions and compute AUROC scores
from sklearn.metrics import roc_auc_score


def compute_auroc_scores(results):
    """Compute AUROC scores for each layer's probe predictions"""
    test_answers = results["test_answers"]
    model_preds = results["model_predictions"]
    probe_preds = results["probe_predictions_per_layer"]

    # Convert answers to binary (yes=1, no=0)
    y_true = np.array([1 if ans.lower() == "yes" else 0 for ans in test_answers])
    y_model = np.array([1 if pred.lower() == "yes" else 0 for pred in model_preds])

    # Compute model accuracy
    model_acc = np.mean(y_true == y_model)

    # Compute AUROC for each layer against ground truth and model predictions
    scores = {"vs_truth": {}, "vs_model": {}, "acc_truth": {}, "acc_model": {}}
    for layer, preds in probe_preds.items():
        # Use probability of "yes" class
        y_pred = preds[:, 1]

        # AUROC against ground truth
        auroc_truth = roc_auc_score(y_true, y_pred)
        scores["vs_truth"][layer] = auroc_truth

        # AUROC against model predictions
        auroc_model = roc_auc_score(y_model, y_pred)
        scores["vs_model"][layer] = auroc_model

        # Accuracy against ground truth
        acc_truth = np.mean(y_true == preds.argmax(axis=1))
        scores["acc_truth"][layer] = acc_truth

        # Accuracy against model predictions
        acc_model = np.mean(y_model == preds.argmax(axis=1))
        scores["acc_model"][layer] = acc_model

    layers = list(probe_preds.keys())

    # Plot accuracy results in separate figure
    plt.figure(figsize=(10, 6))
    plt.plot(
        layers,
        [scores["acc_truth"][l] for l in layers],
        label="Probe Accuracy vs Ground Truth",
        marker="o",
        color="blue",
        linestyle="--",
    )
    plt.plot(
        layers,
        [scores["acc_model"][l] for l in layers],
        label="Probe Accuracy vs Model",
        marker="o",
        color="red",
        linestyle="--",
    )
    plt.axhline(
        y=model_acc,
        color="r",
        linestyle="--",
        label=f"Model Accuracy ({model_acc:.3f})",
    )
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Probe Accuracy Across Layers")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot results
    plt.figure(figsize=(10, 6))

    plt.plot(
        layers,
        [scores["vs_truth"][l] for l in layers],
        label="Probe vs Ground Truth",
        marker="o",
    )
    plt.plot(
        layers,
        [scores["vs_model"][l] for l in layers],
        label="Probe vs Model",
        marker="o",
    )

    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title("Probe AUROC Across Layers")
    plt.legend()
    plt.grid(True)
    plt.show()

    return scores


# # Load saved predictions
# save_dir = f"../results/predictions/google/gemma-2-9b-it/6/logical_deduction"
# # save_dir = f"../results/predictions/{model.model_name}/6/{DATASET_NAME}"
# with open(os.path.join(save_dir, "predictions.pkl"), "rb") as f:
#     results = pickle.load(f)

# # Compute and print AUROC scores
# scores = compute_auroc_scores(results)
# print("\nAUROC scores per layer:")
# scores.keys()
# for layer, score in scores["vs_truth"].items():
#     print(f"{layer=}")
#     print(f"{score=}")
#     print(f"Layer {layer}: {score:.3f}")


# %%
def sample(logits, temperature=0.7):
    scaled_logits = logits / temperature
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    sampled_tokens = torch.multinomial(probs, num_samples=1)
    return sampled_tokens


def generate_and_cache_all_layers(batch, model, max_new_tokens=100, temperature=0.7):
    """Generate response and cache activations for all layers"""
    batch_size = batch.shape[0]

    # Initialize cache storage for all layers
    all_layer_caches = {
        layer: torch.empty(batch_size, 0, model.cfg.d_model)
        for layer in range(model.cfg.n_layers)
    }

    # Start with the input tokens (excluding the last 2 tokens as in original code)
    response = batch[:, :-2].to("cpu")

    # Generate tokens one by one
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits, cache = model.run_with_cache(
                response.to(model.device), pos_slice=-1
            )

            # Store activations for all layers at this position
            for layer in range(model.cfg.n_layers):
                layer_activations = cache["resid_post", layer]
                all_layer_caches[layer] = torch.cat(
                    [all_layer_caches[layer], layer_activations.cpu()], dim=1
                )

            # Sample next token
            sampled_tokens = sample(logits[:, -1, :], temperature)
            response = torch.cat([response, sampled_tokens.cpu()], dim=1)

            # Check if all sequences have reached EOS
            if (response[:, -1] == model.tokenizer.eos_token_id).all():
                break

    eos_mask = (response == model.tokenizer.eos_token_id)[:, batch.shape[1] - 2 :]
    for layer in all_layer_caches:
        all_layer_caches[layer] = [
            all_layer_caches[layer][i][~eos_mask[i]] for i in range(batch_size)
        ]
    response = [
        resp[batch.shape[1] - 2 :][~eos_mask[i]] for i, resp in enumerate(response)
    ]
    return response, all_layer_caches


def generate_and_save_trainset_cache(
    model,
    train_dataset,
    batch_size=4,
    max_new_tokens=100,
    temperature=0.7,
    run_name="trainset_cache",
):
    """Generate responses and save activations for all layers on the entire trainset"""
    print(
        f"Generating and caching activations for {len(train_dataset)} training examples..."
    )

    # Create dataloader
    train_data = TestDataset(train_dataset, model)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # Storage for all results
    all_responses = []
    all_layer_caches = {layer: [] for layer in range(model.cfg.n_layers)}
    all_prompts = []
    all_correct_answers = []
    all_correct_letters = []

    for batch_idx, (prompts, correct_answers, correct_letters) in enumerate(
        train_loader
    ):
        print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")

        # Tokenize prompts
        tokens = model.to_tokens(prompts, prepend_bos=True)
        tokens_padded = right_to_left_pad(tokens, model)

        # Generate responses and cache all layers
        responses, layer_caches = generate_and_cache_all_layers(
            tokens_padded, model, max_new_tokens, temperature
        )

        # Store results
        all_responses.extend([model.to_string(resp) for resp in responses])
        all_prompts.extend(prompts)
        all_correct_answers.extend(correct_answers)
        all_correct_letters.extend(correct_letters)

        # Store layer caches
        for layer in range(model.cfg.n_layers):
            all_layer_caches[layer].extend(layer_caches[layer])

        # Clean up
        del tokens, tokens_padded, responses, layer_caches
        torch.cuda.empty_cache()
        gc.collect()

    # Concatenate all layer caches
    # Compile results
    results = {
        "responses": all_responses,
        "prompts": all_prompts,
        "correct_answers": all_correct_answers,
        "correct_letters": all_correct_letters,
        "layer_caches": all_layer_caches,
        "metadata": {
            "num_examples": len(all_responses),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "num_layers": model.cfg.n_layers,
            "model_name": model.model_name,
        },
    }

    # Save results
    save_dir = f"../results/cache/{model.model_name}/{run_name}/{DATASET_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    # Save main results
    results_path = os.path.join(save_dir, "trainset_cache.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved main results to {results_path}")

    # Save layer caches separately for easier access
    for layer in range(model.cfg.n_layers):
        layer_cache_path = os.path.join(save_dir, f"layer_{layer}_cache.pt")
        torch.save(all_layer_caches[layer], layer_cache_path)
        print(f"Saved layer {layer} cache to {layer_cache_path}")

    # Save metadata
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(results["metadata"], f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print(f"Successfully processed {len(all_responses)} training examples")
    return results


def load_trainset_cache(model, run_name="trainset_cache"):
    """Load cached trainset results"""
    save_dir = f"../results/cache/{model.model_name}/{run_name}/{DATASET_NAME}"

    # Load main results
    results_path = os.path.join(save_dir, "trainset_cache.pkl")
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Load layer caches
    layer_caches = {}
    for layer in range(model.cfg.n_layers):
        layer_cache_path = os.path.join(save_dir, f"layer_{layer}_cache.pt")
        layer_caches[layer] = torch.load(layer_cache_path)

    results["layer_caches"] = layer_caches
    return results


# # %% Example usage for trainset caching
# # Generate and save cache for entire trainset
# results = generate_and_save_trainset_cache(
#     model=model,
#     train_dataset=train_dataset,
#     batch_size=4,  # Adjust based on your GPU memory
#     max_new_tokens=100,
#     temperature=0.7,
#     run_name="trainset_cache_v2"
# )

# %%
results = generate_and_save_trainset_cache(
    model=model,
    train_dataset=test_dataset,
    batch_size=16,  # Adjust based on your GPU memory
    max_new_tokens=100,
    temperature=0.7,
    run_name="testset_cache_v2",
)
# %%
results = load_trainset_cache(model, run_name="trainset_cache_v2")
results_test = load_trainset_cache(model, run_name="testset_cache_v2")
# %%
batch_size = 4

Y_train = []
for i, label in enumerate(results["correct_answers"]):
    seq_len = len(results["layer_caches"][0][i])
    Y_train.extend([label] * seq_len)
Y_binary_train = [1 if y == "yes" else 0 for y in Y_train]

Y_test = []
for i, label in enumerate(results_test["correct_answers"]):
    seq_len = len(results_test["layer_caches"][0][i])
    Y_test.extend([label] * seq_len)
Y_binary_test = [1 if y == "yes" else 0 for y in Y_test]


layer_scores = []
for layer in range(model.cfg.n_layers):
    X = torch.cat(results["layer_caches"][layer], dim=0)
    probe = LogisticRegression(random_state=0)
    probe.fit(X, Y_binary_train)
    X_test = torch.cat(results_test["layer_caches"][layer], dim=0)
    y_pred_proba = probe.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(Y_binary_test, y_pred_proba)
    print(f"Layer {layer} AUROC score: {auc_score:.4f}")

    # Save probe
    save_dir = f"../results/probes/{model.model_name}/sequence_probes/{DATASET_NAME}/"
    os.makedirs(save_dir, exist_ok=True)
    probe_path = os.path.join(save_dir, f"probe_layer_{layer}.pkl")
    with open(probe_path, "wb") as f:
        pickle.dump(probe, f)

    # Store score for plotting
    layer_scores.append(auc_score)

# %%
# Plot scores
plt.figure(figsize=(10, 6))
plt.plot(range(model.cfg.n_layers), layer_scores, marker="o")
plt.xlabel("Layer")
plt.ylabel("AUROC Score")
plt.title("AUROC Scores by Layer")
plt.ylim(0, 1)
plt.grid(True)
plt.savefig(os.path.join(save_dir, "auroc_scores.png"))
plt.close()

# %%
prompt = format_prompt(model, test_dataset[sample_num]["prompt"])
batch = model.to_tokens([prompt], prepend_bos=True)
response, all_cache = generate_and_cache_all_layers(
    batch, model, max_new_tokens=100, temperature=0.7
)
# %%
save_dir = f"../results/probes/{model.model_name}/sequence_probes/{DATASET_NAME}/"
probe_path = os.path.join(save_dir, f"probe_layer_24.pkl")
with open(probe_path, "rb") as f:
    probe = pickle.load(f)
# %%
# Load probe for layer 24
sample_num = 17

data = results_test
# Get activations for layer 24 and run inference
activations = data["layer_caches"][24][sample_num]
predictions = probe.predict_proba(activations)[:, 1]  # Get probability of "yes" class
print(f"Mean prediction: {predictions.mean():.3f}")
print(f"Predictions over time: {predictions}")
# Get correct answer from test dataset
pred_answer = data["pred_answers"][sample_num].lower()
print(f"Pred answer: {pred_answer}")
#
# Plot predictions over time
plt.figure(figsize=(10, 6))
plt.plot(predictions, marker="o")
plt.xlabel("Token Position")
plt.ylabel('Probability of "Yes"')
plt.title("Probe Predictions Over Generation")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# %%
# plt.plot(probe.predict_proba(activations)[:, 1].cumsum())# %%
# parse_response(results['responses'][10])
results_test["pred_answers"]
# %%
[
    results_test["pred_answers"][i]
    for i in range(len(results_test["pred_answers"]))
    if results_test["pred_answers"][i] == "yes"
]
# %%
# results_test['all_layer_caches'][24][17]
print(
    sum(
        [
            probe.predict_proba(results_test["layer_caches"][24][i])[:, 1].sum()
            for i in range(len(results_test["pred_answers"]))
            if results_test["pred_answers"][i] == "yes"
        ]
    )
    / len(results_test["pred_answers"])
)
print(
    sum(
        [
            probe.predict_proba(results_test["layer_caches"][24][i])[:, 1].sum()
            for i in range(len(results_test["pred_answers"]))
            if results_test["pred_answers"][i] == "no"
        ]
    )
    / len(results_test["pred_answers"])
)
# %%
probe_preds = [
    probe.predict_proba(results_test["layer_caches"][24][i])[:, 1]
    for i in range(len(results_test["pred_answers"]))
    if results_test["pred_answers"][i] == "no"
]
# for pred in probe_preds:
# Collect all predictions into one array
all_preds = []
for pred in probe_preds:
    x = np.linspace(0, 1, pred.shape[0])
    all_preds.append(np.column_stack((x, pred)))
all_preds = np.vstack(all_preds)

# Create single plot
plt.figure(figsize=(10, 6))
plt.scatter(all_preds[:, 0], all_preds[:, 1], alpha=0.3)

# Calculate and plot line of best fit
z = np.polyfit(all_preds[:, 0], all_preds[:, 1], 1)
p = np.poly1d(z)
plt.plot(all_preds[:, 0], p(all_preds[:, 0]), "r--", alpha=0.8)

plt.xlabel("Normalized Position")
plt.ylabel("Probe Prediction")
plt.title("Probe Predictions Over Generation (All Samples)")
plt.show()


# %%
results = load_trainset_cache(model, run_name="trainset_cache_v2")
# %%
results_test = load_trainset_cache(model, run_name="testset_cache_v2")
# %%
results_test["pred_answers"] = [
    parse_response(results_test["responses"][i])[1]
    for i in range(len(results_test["responses"]))
]
results["pred_answers"] = [
    parse_response(results["responses"][i])[1] for i in range(len(results["responses"]))
]


# %%
def analyze_token_positions(model, results, results_test, token_pcts=None):
    """Analyze probe performance at different token positions in the sequence.

    Args:
        model: The model being analyzed
        results: Dictionary containing training data and results
        results_test: Dictionary containing test data and results
        token_pcts: List of token percentages to analyze. Defaults to [0.1, 0.2, ..., 0.9]

    Returns:
        train_accs_by_pct: Dict mapping token percentages to list of training accuracies
        test_aurocs_by_pct: Dict mapping token percentages to list of test AUROCs
    """
    if token_pcts is None:
        token_pcts = [i / 10 for i in range(1, 10)]

    num_samples = len(results["layer_caches"][0])
    num_layers = model.cfg.n_layers

    # Lists to store metrics for each token percentage
    train_accs_by_pct = {pct: [] for pct in token_pcts}
    test_aurocs_by_pct = {pct: [] for pct in token_pcts}

    for token_pct in token_pcts:
        print(f"\nAnalyzing token percentage: {token_pct}")

        for layer_idx in range(num_layers):
            training_data_list = []
            for sample_idx in range(num_samples):
                resid = results["layer_caches"][layer_idx][sample_idx]
                seq_len, _ = resid.shape
                training_data_list.append(resid[int(seq_len * token_pct), :])
            training_data = torch.stack(training_data_list)

            # Train probe for this layer
            X = training_data.numpy()
            y = np.array([1 if ans == "yes" else 0 for ans in results["pred_answers"]])

            # Initialize and train probe
            probe = LogisticRegression(random_state=42, max_iter=1000)
            probe.fit(X, y)

            # Store probe accuracy and AUROC
            train_acc = probe.score(X, y)
            train_accs_by_pct[token_pct].append(train_acc)

            # Get test data for this layer
            test_data_list = []
            for sample_idx in range(len(results_test["layer_caches"][0])):
                resid = results_test["layer_caches"][layer_idx][sample_idx]
                seq_len, _ = resid.shape
                test_data_list.append(resid[int(seq_len * token_pct), :])
            test_data = torch.stack(test_data_list).numpy()
            test_y = np.array(
                [1 if ans == "yes" else 0 for ans in results_test["pred_answers"]]
            )

            # Compute AUROC on test data
            test_probs = probe.predict_proba(test_data)[:, 1]
            test_auroc = roc_auc_score(test_y, test_probs)
            test_aurocs_by_pct[token_pct].append(test_auroc)

            print(
                f"Layer {layer_idx} - Train acc: {train_acc:.3f}, Test AUROC: {test_auroc:.3f}"
            )

            # Clean up to save memory
            del X, y, training_data, training_data_list
            gc.collect()

    return train_accs_by_pct, test_aurocs_by_pct


def plot_token_position_results(token_pcts, test_aurocs_by_pct, num_layers):
    """Plot the results of token position analysis.

    Args:
        token_pcts: List of token percentages analyzed
        test_aurocs_by_pct: Dict mapping token percentages to list of test AUROCs
        num_layers: Number of model layers
    """
    # Plot AUROC across layers for each token percentage
    plt.figure(figsize=(12, 8))
    for token_pct in token_pcts:
        plt.plot(
            range(num_layers),
            test_aurocs_by_pct[token_pct],
            marker="o",
            label=f"Token {int(token_pct*100)}%",
        )
    plt.xlabel("Layer")
    plt.ylabel("Test AUROC")
    plt.title("Test AUROC by Layer for Different Token Positions")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create color gradient plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(token_pcts)))
    plt.figure(figsize=(12, 8))
    for idx, token_pct in enumerate(token_pcts):
        plt.plot(
            range(num_layers),
            test_aurocs_by_pct[token_pct],
            marker="o",
            label=f"Token {int(token_pct*100)}%",
            color=colors[idx],
        )
    plt.xlabel("Layer")
    plt.ylabel("Test AUROC")
    plt.title("Test AUROC by Layer for Different Token Positions")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and plot average AUROC across layers
    avg_aurocs = []
    for token_pct in token_pcts:
        avg_auroc = np.mean(test_aurocs_by_pct[token_pct])
        avg_aurocs.append(avg_auroc)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(token_pcts)), avg_aurocs, color=colors)
    plt.xticks(range(len(token_pcts)), [f"{int(pct*100)}%" for pct in token_pcts])
    plt.xlabel("Token Position")
    plt.ylabel("Average Test AUROC")
    plt.title("Average Test AUROC Across Layers by Token Position")
    plt.grid(True, axis="y")
    plt.show()


# %%
test_dataset[0]["prompt"][-2]
