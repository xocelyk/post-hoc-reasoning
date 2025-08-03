import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    model_name: str
    dataset_name: str
    train_size: int
    test_size: int
    split_seed: int
    alpha_range: List[float]
    temperature: float = 0.7
    max_new_tokens: int = 100

    def get_hash(self) -> str:
        """Generate a unique hash for this experiment configuration."""
        # Include steering method in hash if available
        steering_method = getattr(self, 'steering_method', 'default')
        config_str = (
            f"{self.model_name}_{self.dataset_name}_{self.train_size}_{self.test_size}_"
            f"{self.split_seed}_{self.temperature}_{self.max_new_tokens}_{steering_method}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def get_cache_dir(self, base_cache_dir: str = "cache") -> str:
        """Get the cache directory for this experiment configuration."""
        return os.path.join(
            base_cache_dir,
            "experiments",
            self.model_name.replace("/", "_"),
            self.dataset_name,
            f"split_{self.split_seed}_{self.train_size}_{self.test_size}",
            self.get_hash(),
        )


class ExperimentCache:
    """Manages caching for experiment results with hierarchical organization."""

    def __init__(self, config: ExperimentConfig, base_cache_dir: str = "cache"):
        self.config = config
        self.base_cache_dir = base_cache_dir
        self.cache_dir = config.get_cache_dir(base_cache_dir)
        self._ensure_cache_dirs()

    def _ensure_cache_dirs(self):
        """Create all necessary cache directories."""
        dirs = [
            self.cache_dir,
            os.path.join(self.cache_dir, "data"),
            os.path.join(self.cache_dir, "probes"),
            os.path.join(self.cache_dir, "steering"),
            os.path.join(self.cache_dir, "metadata"),
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def save_experiment_config(self):
        """Save the experiment configuration to cache."""
        config_path = os.path.join(self.cache_dir, "metadata", "config.json")
        config_dict = {
            "model_name": self.config.model_name,
            "dataset_name": self.config.dataset_name,
            "train_size": self.config.train_size,
            "test_size": self.config.test_size,
            "split_seed": self.config.split_seed,
            "alpha_range": self.config.alpha_range,
            "temperature": self.config.temperature,
            "max_new_tokens": self.config.max_new_tokens,
            "cache_hash": self.config.get_hash(),
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def load_experiment_config(self) -> Optional[Dict]:
        """Load the experiment configuration from cache."""
        config_path = os.path.join(self.cache_dir, "metadata", "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return None

    # Data caching methods
    def get_dataset_path(self) -> str:
        return os.path.join(self.cache_dir, "data", "dataset.pkl")

    def get_train_test_split_path(self) -> str:
        return os.path.join(self.cache_dir, "data", "train_test_split.pkl")

    def get_train_generations_path(self) -> str:
        return os.path.join(self.cache_dir, "data", "train_generations.pkl")

    def get_test_generations_path(self) -> str:
        return os.path.join(self.cache_dir, "data", "test_generations.pkl")

    def get_train_activations_path(self) -> str:
        return os.path.join(self.cache_dir, "data", "train_activations.pkl")

    def get_test_activations_path(self) -> str:
        return os.path.join(self.cache_dir, "data", "test_activations.pkl")

    # Probe caching methods
    def get_probes_path(self) -> str:
        return os.path.join(self.cache_dir, "probes", "trained_probes.pkl")

    def get_probe_coefficients_path(self) -> str:
        return os.path.join(self.cache_dir, "probes", "coefficients.pkl")

    def get_auc_scores_path(self) -> str:
        return os.path.join(self.cache_dir, "probes", "auc_scores.json")

    def get_probe_metadata_path(self) -> str:
        return os.path.join(self.cache_dir, "probes", "metadata.json")

    # Steering caching methods
    def get_steering_results_path(self, alpha: float, label: str) -> str:
        """Get path for steering results for a specific alpha and label."""
        filename = f"steering_alpha_{alpha}_{label}.pkl"
        return os.path.join(self.cache_dir, "steering", filename)

    def get_steering_summary_path(self) -> str:
        return os.path.join(self.cache_dir, "steering", "summary.json")

    # Generic save/load methods
    def save_pickle(self, data: Any, filepath: str):
        """Save data as pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_pickle(self, filepath: str) -> Any:
        """Load data from pickle file."""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None

    def save_json(self, data: Dict, filepath: str):
        """Save data as JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: str) -> Optional[Dict]:
        """Load data from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    # High-level cache status methods
    def has_dataset(self) -> bool:
        """Check if dataset is cached."""
        return os.path.exists(self.get_dataset_path())

    def has_train_test_split(self) -> bool:
        """Check if train/test split is cached."""
        return os.path.exists(self.get_train_test_split_path())

    def has_generations(self) -> bool:
        """Check if generations are cached."""
        return os.path.exists(self.get_train_generations_path()) and os.path.exists(
            self.get_test_generations_path()
        )

    def has_activations(self) -> bool:
        """Check if activations are cached."""
        return os.path.exists(self.get_train_activations_path()) and os.path.exists(
            self.get_test_activations_path()
        )

    def has_probes(self) -> bool:
        """Check if probes are trained and cached."""
        return (
            os.path.exists(self.get_probes_path())
            and os.path.exists(self.get_probe_coefficients_path())
            and os.path.exists(self.get_auc_scores_path())
        )

    def has_steering_results(self, alpha: float, label: str) -> bool:
        """Check if steering results exist for specific alpha and label."""
        return os.path.exists(self.get_steering_results_path(alpha, label))

    def get_completed_steering(self) -> List[Tuple[float, str]]:
        """Get list of completed steering experiments (alpha, label) pairs."""
        completed = []
        steering_dir = os.path.join(self.cache_dir, "steering")
        if os.path.exists(steering_dir):
            for filename in os.listdir(steering_dir):
                if filename.startswith("steering_alpha_") and filename.endswith(".pkl"):
                    # Parse filename: steering_alpha_2.0_yes.pkl
                    parts = filename[15:-4].rsplit(
                        "_", 1
                    )  # Remove "steering_alpha_" and ".pkl"
                    if len(parts) == 2:
                        try:
                            alpha = float(parts[0])
                            label = parts[1]
                            completed.append((alpha, label))
                        except ValueError:
                            continue
        return completed

    def get_experiment_status(self) -> Dict[str, bool]:
        """Get the current status of all experiment components."""
        return {
            "dataset": self.has_dataset(),
            "train_test_split": self.has_train_test_split(),
            "generations": self.has_generations(),
            "activations": self.has_activations(),
            "probes": self.has_probes(),
            "steering_complete": len(self.get_completed_steering())
            >= len(self.config.alpha_range) * 2,
        }

    def clean_cache(self):
        """Remove all cached data for this experiment."""
        import shutil

        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def get_cache_size(self) -> int:
        """Get total size of cached data in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size


class ExperimentManager:
    """Manages multiple experiment configurations and their caches."""

    def __init__(self, base_cache_dir: str = "cache"):
        self.base_cache_dir = base_cache_dir
        self.experiments: Dict[str, ExperimentCache] = {}

    def add_experiment(self, config: ExperimentConfig) -> ExperimentCache:
        """Add an experiment configuration and return its cache manager."""
        exp_id = config.get_hash()
        if exp_id not in self.experiments:
            self.experiments[exp_id] = ExperimentCache(config, self.base_cache_dir)
            self.experiments[exp_id].save_experiment_config()
        return self.experiments[exp_id]

    def get_experiment(self, exp_id: str) -> Optional[ExperimentCache]:
        """Get experiment cache by ID."""
        return self.experiments.get(exp_id)

    def list_experiments(self) -> List[Tuple[str, ExperimentConfig]]:
        """List all experiments with their IDs and configurations."""
        experiments = []
        experiments_dir = os.path.join(self.base_cache_dir, "experiments")
        if os.path.exists(experiments_dir):
            for model_dir in os.listdir(experiments_dir):
                model_path = os.path.join(experiments_dir, model_dir)
                if os.path.isdir(model_path):
                    for dataset_dir in os.listdir(model_path):
                        dataset_path = os.path.join(model_path, dataset_dir)
                        if os.path.isdir(dataset_path):
                            for split_dir in os.listdir(dataset_path):
                                split_path = os.path.join(dataset_path, split_dir)
                                if os.path.isdir(split_path):
                                    for exp_dir in os.listdir(split_path):
                                        exp_path = os.path.join(split_path, exp_dir)
                                        config_path = os.path.join(
                                            exp_path, "metadata", "config.json"
                                        )
                                        if os.path.exists(config_path):
                                            with open(config_path, "r") as f:
                                                config_data = json.load(f)
                                                config = ExperimentConfig(
                                                    model_name=config_data[
                                                        "model_name"
                                                    ],
                                                    dataset_name=config_data[
                                                        "dataset_name"
                                                    ],
                                                    train_size=config_data[
                                                        "train_size"
                                                    ],
                                                    test_size=config_data["test_size"],
                                                    split_seed=config_data[
                                                        "split_seed"
                                                    ],
                                                    alpha_range=config_data[
                                                        "alpha_range"
                                                    ],
                                                    temperature=config_data.get(
                                                        "temperature", 0.7
                                                    ),
                                                    max_new_tokens=config_data.get(
                                                        "max_new_tokens", 100
                                                    ),
                                                )
                                                experiments.append((exp_dir, config))
        return experiments

    def get_experiments_summary(self) -> pd.DataFrame:
        """Get a summary of all experiments as a DataFrame."""
        experiments = self.list_experiments()
        if not experiments:
            return pd.DataFrame()

        rows = []
        for exp_id, config in experiments:
            cache = ExperimentCache(config, self.base_cache_dir)
            status = cache.get_experiment_status()
            completed_steering = cache.get_completed_steering()

            rows.append(
                {
                    "experiment_id": exp_id,
                    "model": config.model_name,
                    "dataset": config.dataset_name,
                    "train_size": config.train_size,
                    "test_size": config.test_size,
                    "split_seed": config.split_seed,
                    "has_data": status["generations"] and status["activations"],
                    "has_probes": status["probes"],
                    "steering_progress": f"{len(completed_steering)}/{len(config.alpha_range) * 2}",
                    "cache_size_mb": cache.get_cache_size() / (1024 * 1024),
                }
            )

        return pd.DataFrame(rows)
