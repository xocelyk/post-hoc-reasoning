import argparse
import gc
import json
import logging
import os
import pickle
import re
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from data_loading import load_all_datasets
from models import ChatModel
from utils import generate_with_hooks

# %%
CACHE_DIR = "cache"
RESULTS_DIR = "results"


def get_generation_cache_paths(model_name: str, dataset_name: str):
    """
    Returns the paths for generation and activation cache for a given model and dataset.
    """
    train_cache = os.path.join(
        CACHE_DIR, model_name, dataset_name, "train_generations.pkl"
    )
    test_cache = os.path.join(
        CACHE_DIR, model_name, dataset_name, "test_generations.pkl"
    )
    train_activations_cache = os.path.join(
        CACHE_DIR, model_name, dataset_name, "train_activations.pkl"
    )
    test_activations_cache = os.path.join(
        CACHE_DIR, model_name, dataset_name, "test_activations.pkl"
    )
    return train_cache, test_cache, train_activations_cache, test_activations_cache


# %%
class PromptDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, Tuple[str, str]]:
        return self.data[idx]["prompt"], (
            self.data[idx]["correct_answer"],
            self.data[idx]["correct_letter"],
        )


# Helper functions
def parse_response(response: str) -> Tuple[str, str]:
    # TODO: Make more robust; this only works for gemma
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


def batch_get_resid_activations(prompts, model: ChatModel):
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


def batch_get_generations(
    prompts, model: ChatModel, temperature=0.7, max_new_tokens=100
):
    tokens = model.to_tokens(prompts, prepend_bos=True)
    token_generations = model.generate(
        tokens, max_new_tokens=max_new_tokens, temperature=temperature
    )
    generations = model.to_string(token_generations)
    return generations


def process_batch(
    prompts,
    correct_tups,
    model: ChatModel,
    get_activations=True,
    temperature=0.7,
    max_new_tokens=100,
):
    correct_answers, correct_letters = correct_tups

    activations = (
        batch_get_resid_activations(prompts, model) if get_activations else None
    )
    generations = batch_get_generations(
        prompts, model, temperature=temperature, max_new_tokens=max_new_tokens
    )
    generations = [gen[len(prompt) :] for gen, prompt in zip(generations, prompts)]

    responses = [parse_response(response) for response in generations]
    pred_letters, pred_answers = zip(*responses)

    corrects = [pred == correct for pred, correct in zip(pred_letters, correct_letters)]

    for (
        prompt,
        generation,
        pred_letter,
        pred_answer,
        correct_letter,
        correct_answer,
    ) in zip(
        prompts,
        generations,
        pred_letters,
        pred_answers,
        correct_letters,
        correct_answers,
    ):
        prompt_question = prompt.strip().split("Q:")[-1].split("\n")[0].strip()
        print(f"Prompt: {prompt_question}")
        print(
            f"Generation: {generation.strip().replace('<eos>', '').replace('<pad>', '').replace('<end_of_turn>', '').strip()}"
        )
        print(
            f"Predicted: {pred_letter} - {pred_answer}, Correct: {correct_letter} - {correct_answer}"
        )
        print()
        print("-" * 100)
        print()

    return activations, generations, pred_letters, pred_answers, corrects


def process_dataset(
    dataloader,
    model: ChatModel,
    max_samples=None,
    max_gen=None,
    get_activations=True,
    temperature=0.7,
    max_new_tokens=100,
):
    results = []
    activations_list = []
    all_corrects = []
    sample_count = 0

    for prompts, correct_tups in dataloader:
        activations, generations, pred_letters, pred_answers, corrects = process_batch(
            prompts,
            correct_tups,
            model,
            get_activations=get_activations,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        all_corrects.extend(corrects)
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
            if get_activations:
                activations_list.append(activations[i])

        print(
            f"Processed {sample_count} samples. Accuracy: {np.mean(all_corrects):.2f}"
        )

        if max_samples and sample_count >= max_samples:
            break
        if max_gen and sample_count >= max_gen:
            break

    return results, activations_list


def save_pickle(data, filepath):
    print(f"Saving to {filepath}")
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def setup_logging(model_name, dataset_name):
    log_dir = os.path.join(RESULTS_DIR, model_name, dataset_name, "logs")
    ensure_dir(log_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info(f"Logging started for model: {model_name}, dataset: {dataset_name}")


def save_json(data, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def generate_and_save_generations(
    model: ChatModel,
    dataset_name: str,
    train_size: int = 100,
    batch_size: int = 2,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    use_cache: bool = True,
    max_gen: int = None,
):
    model_name = getattr(model, "model_name", "google/gemma-2-9b-it")
    setup_logging(model_name, dataset_name)
    print(
        f"Generating and saving generations for dataset: {dataset_name} with model: {model_name}"
    )

    dataset_cache_path = os.path.join(
        CACHE_DIR, model_name, dataset_name, "dataset.pkl"
    )

    if use_cache and os.path.exists(dataset_cache_path):
        print("Loading cached dataset...")
        dataset = load_pickle(dataset_cache_path)
    else:
        print("Loading dataset from source...")
        datasets = load_all_datasets()
        dataset = datasets[dataset_name]

        if use_cache:
            ensure_dir(os.path.dirname(dataset_cache_path))
            save_pickle(dataset, dataset_cache_path)
            print("Dataset cached")

    print(f"Dataset loaded: {dataset_name}, total samples: {len(dataset)}")

    train_dataset, test_dataset = train_test_split(
        dataset, train_size=train_size, random_state=42
    )
    print(
        f"Dataset split: train size = {len(train_dataset)}, test size = {len(test_dataset)}"
    )

    (
        train_cache_path,
        test_cache_path,
        train_activations_cache,
        test_activations_cache,
    ) = get_generation_cache_paths(model_name, dataset_name)

    if (
        use_cache
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
        and os.path.exists(train_activations_cache)
        and os.path.exists(test_activations_cache)
    ):
        print("Generations and activations already cached.")
        return
    else:
        train_dataloader = DataLoader(
            PromptDataset(train_dataset), batch_size=batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            PromptDataset(test_dataset), batch_size=batch_size, shuffle=False
        )

        print("Processing training data...")
        train_results, train_activations = process_dataset(
            train_dataloader,
            model,
            max_samples=train_size,
            max_gen=max_gen,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        print(f"Training data processed: {len(train_results)} samples")

        print("Processing test data...")
        test_results, test_activations = process_dataset(
            test_dataloader,
            model,
            max_gen=max_gen,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        print(f"Test data processed: {len(test_results)} samples")

        if use_cache:
            save_pickle(train_results, train_cache_path)
            save_pickle(test_results, test_cache_path)
            save_pickle(train_activations, train_activations_cache)
            save_pickle(test_activations, test_activations_cache)
            print("Generations and activations cached")


def run_steering_experiment(
    model: ChatModel,
    dataset_name: str,
    alpha_range: List[int] = [0, 1, 2, 3, 5, 7],
    use_cache: bool = True,
):
    model_name = getattr(model, "model_name", "google/gemma-2-9b-it")
    setup_logging(model_name, dataset_name)
    print(
        f"Running steering experiments for dataset: {dataset_name} with model: {model_name}"
    )

    (
        train_cache_path,
        test_cache_path,
        train_activations_cache,
        test_activations_cache,
    ) = get_generation_cache_paths(model_name, dataset_name)
    train_results = load_pickle(train_cache_path)
    test_results = load_pickle(test_cache_path)
    train_activations = load_pickle(train_activations_cache)
    test_activations = load_pickle(test_activations_cache)
    print(
        f"Results loaded: {len(train_results)} train samples, {len(test_results)} test samples"
    )

    layers = list(range(model.cfg.n_layers))
    probes_cache_path = os.path.join(
        RESULTS_DIR, model_name, dataset_name, "all_coef_vectors.pkl"
    )
    layers_cache_path = os.path.join(
        RESULTS_DIR, model_name, dataset_name, "selected_layers.pkl"
    )

    print("Training classifiers and saving probes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_coef_vectors, selected_layers = train_and_save_probes(
            model,
            train_results,
            train_activations,
            test_results,
            test_activations,
            layers,
            dataset_name,
        )
    print("Classifiers trained and probes saved")

    if use_cache:
        save_pickle(all_coef_vectors, probes_cache_path)
        save_pickle(selected_layers, layers_cache_path)
        print(f"Coefficient vectors saved at {probes_cache_path}")
        print(f"Selected layers saved at {layers_cache_path}")

    yes_test_data, no_test_data = create_test_subsets(test_results)
    print(f"Test subsets created: yes = {len(yes_test_data)}, no = {len(no_test_data)}")

    print("Performing steering...")
    perform_steering(
        model,
        yes_test_data,
        no_test_data,
        all_coef_vectors,
        selected_layers,
        dataset_name,
        alpha_range=alpha_range,
        use_cache=use_cache,
    )
    print(f"Steering experiment completed for dataset: {dataset_name}")


def train_and_save_probes(
    model: ChatModel,
    train_results,
    train_activations,
    test_results,
    test_activations,
    layers,
    dataset_name,
):
    results_dir = os.path.join(
        RESULTS_DIR,
        getattr(model, "model_name", "google/gemma-2-9b-it"),
        dataset_name,
        "probes",
    )
    ensure_dir(results_dir)

    all_coef_vectors = []
    auc_scores = []

    for layer in layers:
        print(f"Training classifier for layer {layer}")
        train_data = prepare_data(model, train_results, train_activations, layer)
        test_data = prepare_data(model, test_results, test_activations, layer)

        clf = train_classifier(train_data)

        auc_score = evaluate_classifier(clf, test_data)
        auc_scores.append(auc_score)
        print(f"AUROC for layer {layer}: {auc_score:.4f}")

        probe_path = os.path.join(results_dir, f"layer_{layer}.pkl")
        save_pickle(clf, probe_path)
        print(f"Probe saved for layer {layer} at {probe_path}")

        coef_vector = extract_diff_vector(clf)
        all_coef_vectors.append(coef_vector)

    best_layer = layers[np.argmax(auc_scores)]
    selected_layers = layers

    print(f"Best layer: {best_layer}")
    print(f"Selected layers for steering: {selected_layers}")

    return np.array(all_coef_vectors), selected_layers


def create_test_subsets(test_results):
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
    return yes_test_data, no_test_data


def perform_steering(
    model: ChatModel,
    yes_test_data,
    no_test_data,
    all_coef_vectors,
    selected_layers,
    dataset_name,
    alpha_range: List[int],
    use_cache: bool = True,
    steer_temperature: float = 0.7,
    max_new_tokens: int = 100,
    **kwargs,
):
    torch.cuda.empty_cache()
    gc.collect()

    model_name = getattr(model, "model_name", "google/gemma-2-9b-it")
    steering_cache_path = os.path.join(
        RESULTS_DIR, model_name, dataset_name, "steering_results.pkl"
    )

    if os.path.exists(steering_cache_path):
        print("Loading existing steering results...")
        steering_results = load_pickle(steering_cache_path)
    else:
        steering_results = {"yes": {}, "no": {}}

    for alpha in alpha_range:
        alpha_yes = -alpha
        if alpha_yes in steering_results.get("yes", {}):
            print(
                f"Steering results for alpha {alpha_yes} already exist for 'yes' data, skipping..."
            )
        else:
            print(
                f"Generating steered examples for 'yes' data with alpha = {alpha_yes}"
            )
            results_yes = generate_steered_examples(
                model,
                yes_test_data,
                all_coef_vectors,
                selected_layers,
                alpha=alpha_yes,
                temperature=steer_temperature,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            steering_results.setdefault("yes", {})[alpha_yes] = results_yes
            success_rate = (
                sum(r["success"] for r in results_yes) / len(results_yes)
                if results_yes
                else 0
            )
            print(
                f"Success rate for 'yes' data with alpha = {alpha_yes}: {success_rate:.2f}"
            )

        alpha_no = alpha
        if alpha_no in steering_results.get("no", {}):
            print(
                f"Steering results for alpha {alpha_no} already exist for 'no' data, skipping..."
            )
        else:
            print(f"Generating steered examples for 'no' data with alpha = {alpha_no}")
            results_no = generate_steered_examples(
                model,
                no_test_data,
                all_coef_vectors,
                selected_layers,
                alpha=alpha_no,
                temperature=steer_temperature,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            steering_results.setdefault("no", {})[alpha_no] = results_no
            success_rate = (
                sum(r["success"] for r in results_no) / len(results_no)
                if results_no
                else 0
            )
            print(
                f"Success rate for 'no' data with alpha = {alpha_no}: {success_rate:.2f}"
            )

    if use_cache:
        save_pickle(steering_results, steering_cache_path)
        print(f"Steering results saved at {steering_cache_path}")


def generate_steered_examples(
    model: ChatModel,
    test_data,
    all_coef_vectors,
    selected_layers,
    alpha,
    temperature: float = 0.7,
    max_new_tokens: int = 100,
    **kwargs,
):
    steered_results = []

    for i, example in enumerate(test_data):
        print(f"Generating steered example {i+1}/{len(test_data)} with alpha = {alpha}")
        example_prompt = example["prompt"]
        example_tokens = model.to_tokens(example_prompt, prepend_bos=False)

        generation = generate_with_hooks(
            model,
            example_tokens,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            steering_vectors=all_coef_vectors,
            layers=selected_layers,
            **kwargs,
        )

        new_letter, new_answer = parse_response(generation)

        original_prompt = example_prompt.strip().split("Q:")[-1].split("\n")[0].strip()

        print(f"Original prompt: {original_prompt}")
        print(f"Steered generation: {generation}")
        print(f"Original answer: {example['pred_answer']}, New answer: {new_answer}")

        orig = example["pred_answer"]
        success = (orig == "yes" and new_answer == "no") or (
            orig == "no" and new_answer == "yes"
        )
        print("-" * 50)

        steered_results.append(
            {
                "original_prompt": original_prompt,
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


def prepare_data(model: ChatModel, results, activations, layer):
    data = []
    for idx, result in enumerate(results):
        if result["pred_answer"] == result["correct_answer"]:
            activation = activations[idx][layer]
            data.append(activation.tolist() + [result["pred_answer"]])
    df = pd.DataFrame(
        data, columns=["ac" + str(i) for i in range(model.cfg.d_model)] + ["pred"]
    )
    df = df[df["pred"].isin(["yes", "no"])]
    return df


def train_classifier(train_data):
    X = train_data[[col for col in train_data.columns if col.startswith("ac")]]
    y = train_data["pred"]
    return LogisticRegression(random_state=0).fit(X, y)


def evaluate_classifier(clf, test_data):
    X = test_data[[col for col in test_data.columns if col.startswith("ac")]]
    y = test_data["pred"]
    y = y.apply(lambda x: 1 if x == "yes" else 0)
    try:
        return roc_auc_score(y, clf.predict_proba(X)[:, 1])
    except ValueError:
        return 0


def extract_diff_vector(clf):
    return clf.coef_[0]


def normalize_vectors(vectors):
    return [vec / np.linalg.norm(vec) for vec in vectors]


# New main entry point using argparse for ease-of-use.
def main():
    parser = argparse.ArgumentParser(
        description="Run research experiments using transformer_lens models and various datasets."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        help="Name of the model to use (via transformer_lens).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sports_understanding",
        help="Dataset to use (e.g., sports_understanding, anachronisms, social_chemistry, logical_deduction).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "steer", "all"],
        default="all",
        help="Experiment mode to run.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=100,
        help="Train set size for generation experiments.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for processing."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation."
    )
    parser.add_argument(
        "--alpha_range",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Alpha range for steering experiments.",
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Use cached generations if available."
    )
    args = parser.parse_args()

    model = ChatModel(args.model)

    if args.mode in ["generate", "all"]:
        generate_and_save_generations(
            model,
            args.dataset,
            train_size=args.train_size,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_cache=args.use_cache,
        )
    if args.mode in ["steer", "all"]:
        run_steering_experiment(
            model, args.dataset, alpha_range=args.alpha_range, use_cache=args.use_cache
        )


if __name__ == "__main__":
    main()
