#!/usr/bin/env python3
"""Run orthogonal steering baseline experiments.

This script mirrors the functionality of ``run_transformer_lens_experiments.py``
but applies random orthogonal directions instead of learned probe directions.
It loads existing probe coefficients from cached experiments and runs steered
Generation along orthogonal directions of matched magnitude.
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass, field
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from typing import Any, Dict, List

import numpy as np
import yaml

# Local imports from existing project
from src.config import ModelConfig, DatasetConfig  # type: ignore


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSettings:
    """Settings related to cache locations."""

    cache_dir: str = "cache/experiments_orthogonal"
    use_existing_probes: bool = True
    probe_cache_dir: str = "cache/experiments"


@dataclass
class SteeringSettings:
    """Settings controlling orthogonal steering."""

    alpha_values: List[float] = field(
        default_factory=lambda: [4, 8, 12, 16, 20]
    )
    samples_per_alpha: int = 20
    num_orthogonal_vectors: int = 1
    use_best_probe_layer: bool = True
    seed: int = 42


@dataclass
class OrthogonalRunConfig:
    """Full configuration for running orthogonal steering experiments."""

    experiment: ExperimentSettings
    steering: SteeringSettings
    models: List[ModelConfig]
    datasets: List[DatasetConfig]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_config(path: str) -> OrthogonalRunConfig:
    """Load YAML configuration into dataclasses."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    experiment = ExperimentSettings(**data.get("experiment", {}))
    steering = SteeringSettings(**data.get("steering", {}))
    models = [ModelConfig(**m) for m in data.get("models", [])]
    datasets = [DatasetConfig(**d) for d in data.get("datasets", [])]
    return OrthogonalRunConfig(
        experiment=experiment,
        steering=steering,
        models=models,
        datasets=datasets,
    )


def sample_orthogonal_direction(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a random direction orthogonal to ``w`` with ``||r|| = ||w||``."""
    w = w.astype(np.float32)
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        raise ValueError("w has zero norm")
    u = w / w_norm
    g = rng.standard_normal(w.shape).astype(np.float32)
    g_orth = g - np.dot(g, u) * u
    g_norm = np.linalg.norm(g_orth)
    while g_norm < 1e-8:
        g = rng.standard_normal(w.shape).astype(np.float32)
        g_orth = g - np.dot(g, u) * u
        g_norm = np.linalg.norm(g_orth)
    r = (w_norm / g_norm) * g_orth
    return r


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def run_experiment(config: OrthogonalRunConfig) -> None:
    from src.models import ChatModel  # type: ignore
    rng = np.random.default_rng(config.steering.seed)
    from src.parsing_utils import parse_response  # type: ignore
    from src.utils import generate_with_steering  # type: ignore

    for model_cfg in config.models:
        model_name_sanitised = model_cfg.name.replace("/", "_")
        print(f"Running orthogonal steering for model: {model_cfg.name}")
        model = ChatModel(model_cfg.name, device=model_cfg.device, dtype=model_cfg.dtype)

        for dataset_cfg in config.datasets:
            print(f"  Dataset: {dataset_cfg.name}")
            base_path = os.path.join(
                config.experiment.probe_cache_dir,
                model_name_sanitised,
                dataset_cfg.name,
            )
            if not os.path.exists(base_path):
                print(f"    Skipping {dataset_cfg.name} – no cache found")
                continue

            split_dirs = [
                os.path.join(base_path, d)
                for d in os.listdir(base_path)
                if d.startswith("split_")
            ]
            for split_dir in split_dirs:
                for exp_hash in os.listdir(split_dir):
                    orig_exp_dir = os.path.join(split_dir, exp_hash)
                    probes_dir = os.path.join(orig_exp_dir, "probes")
                    coeff_path = os.path.join(probes_dir, "coefficients.pkl")
                    if not os.path.exists(coeff_path):
                        continue

                    # Load metadata for generation settings
                    meta_path = os.path.join(orig_exp_dir, "metadata", "config.json")
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    temperature = meta.get("temperature", 0.7)
                    max_new_tokens = meta.get("max_new_tokens", 100)

                    # Load test generations for prompts and original answers
                    test_path = os.path.join(orig_exp_dir, "data", "test_generations.pkl")
                    if not os.path.exists(test_path):
                        continue
                    with open(test_path, "rb") as f:
                        test_data = pickle.load(f)

                    # Load probe coefficients and determine best layer
                    with open(coeff_path, "rb") as f:
                        coef_dict = pickle.load(f)
                    auc_path = os.path.join(probes_dir, "auc_scores.json")
                    if os.path.exists(auc_path):
                        with open(auc_path, "r") as f:
                            auc_scores = json.load(f)
                        if isinstance(auc_scores, dict):
                            scores = {int(k): v for k, v in auc_scores.items()}
                        else:
                            scores = {i: s for i, s in enumerate(auc_scores)}
                        best_layer = max(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                    else:
                        best_layer = list(coef_dict.keys())[0]
                    w = np.array(coef_dict[best_layer], dtype=np.float32)

                    # Prepare output directory
                    new_exp_dir = orig_exp_dir.replace(
                        config.experiment.probe_cache_dir,
                        config.experiment.cache_dir,
                    )
                    steering_dir = os.path.join(new_exp_dir, "orthogonal_steering")
                    os.makedirs(steering_dir, exist_ok=True)

                    n_layers = model.cfg.n_layers
                    d_model = w.shape[0]

                    for alpha in config.steering.alpha_values:
                        indices = rng.choice(
                            len(test_data),
                            size=config.steering.samples_per_alpha,
                            replace=False,
                        )
                        alpha_results: List[Dict[str, Any]] = []
                        for idx in indices:
                            example = test_data[idx]
                            if isinstance(example, dict):
                                prompt = example.get("prompt", "")
                                # Handle response being a tuple (letter, answer)
                                response = example.get("response", None)
                                if isinstance(response, tuple) and len(response) == 2:
                                    # Response is already parsed as (letter, answer)
                                    original_answer = response[1]
                                else:
                                    # Try to get generation text and parse it
                                    original_gen = (
                                        example.get("generation")
                                        or example.get("response")
                                        or ""
                                    )
                                    _, original_answer = parse_response(original_gen)
                            else:
                                # Fallback for simple list structures
                                prompt = example[0]
                                original_gen = example[1] if len(example) > 1 else ""
                                _, original_answer = parse_response(original_gen)

                            prompt_tokens = model.to_tokens(prompt, prepend_bos=True)

                            seed_val = int(rng.integers(0, 2**32 - 1))
                            r = sample_orthogonal_direction(w, np.random.default_rng(seed_val))
                            steering_vec = np.zeros((n_layers, d_model), dtype=np.float32)
                            steering_vec[best_layer] = r

                            steered_text = generate_with_steering(
                                model,
                                prompt_tokens,
                                steering_vec,
                                alpha=alpha,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                layers=[best_layer],
                            )
                            _, steered_answer = parse_response(steered_text)

                            is_valid = steered_answer in {"yes", "no"} and original_answer in {
                                "yes",
                                "no",
                            }
                            success = is_valid and steered_answer != original_answer

                            alpha_results.append(
                                {
                                    "prompt": prompt,
                                    "alpha": float(alpha),
                                    "sample_idx": int(idx),
                                    "orthogonal_vector_seed": seed_val,
                                    "original_answer": original_answer,
                                    "steered_answer": steered_answer,
                                    "success": success,
                                    "is_valid_parse": is_valid,
                                }
                            )

                        out_path = os.path.join(steering_dir, f"steering_alpha_{alpha}.pkl")
                        with open(out_path, "wb") as f:
                            pickle.dump(alpha_results, f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run orthogonal steering baseline experiments",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Currently unused; for API compatibility",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Ignored option for compatibility with other runners",
    )
    args = parser.parse_args()

    run_config = load_config(args.config)
    run_experiment(run_config)


if __name__ == "__main__":
    main()
