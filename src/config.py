import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

from cache_manager import ExperimentConfig


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    device: str = "auto"
    dtype: str = "bfloat16"
    batch_size: int = 2
    backend: str = "auto"  # Backend selection: "auto", "nnsight", "transformer_lens"


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    name: str
    train_size: int = 200
    test_size: int = 800
    split_seed: int = 42


@dataclass
class SteeringConfig:
    """Configuration for steering experiments."""

    alpha_range: List[float] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    temperature: float = 0.7
    max_new_tokens: int = 100


@dataclass
class ExperimentRunConfig:
    """Configuration for the overall experiment run."""

    models: List[ModelConfig]
    datasets: List[DatasetConfig]
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    cache_dir: str = "cache"
    use_cache: bool = True
    interactive: bool = True
    max_concurrent_models: int = 1
    save_generations: bool = True
    evaluate_confabulation: bool = (
        False  # Whether to use GPT-4 for confabulation analysis
    )


class ConfigValidator:
    """Validates experiment configurations."""

    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """Validate a model configuration."""
        errors = []

        if not config.name:
            errors.append("Model name cannot be empty")

        if config.device not in ["auto", "cpu", "cuda", "mps"]:
            errors.append(f"Invalid device: {config.device}")

        if config.dtype not in ["float32", "float16", "bfloat16", "int8"]:
            errors.append(f"Invalid dtype: {config.dtype}")

        if config.batch_size <= 0:
            errors.append("Batch size must be positive")

        if config.backend not in ["auto", "nnsight", "transformer_lens"]:
            errors.append(f"Invalid backend: {config.backend}")

        return errors

    @staticmethod
    def validate_dataset_config(config: DatasetConfig) -> List[str]:
        """Validate a dataset configuration."""
        errors = []

        if not config.name:
            errors.append("Dataset name cannot be empty")

        # Check if dataset exists in data loading
        try:
            from data_loading import list_available_datasets

            available = list_available_datasets()
            if config.name not in available:
                errors.append(
                    f"Dataset '{config.name}' not available. Available: {available}"
                )
        except ImportError:
            # If we can't import, just warn
            pass

        if config.train_size <= 0:
            errors.append("Train size must be positive")

        if config.test_size <= 0:
            errors.append("Test size must be positive")

        if config.split_seed < 0:
            errors.append("Split seed must be non-negative")

        return errors

    @staticmethod
    def validate_steering_config(config: SteeringConfig) -> List[str]:
        """Validate a steering configuration."""
        errors = []

        if not config.alpha_range:
            errors.append("Alpha range cannot be empty")

        if any(not isinstance(a, (int, float)) for a in config.alpha_range):
            errors.append("All alpha values must be numeric")

        if config.temperature <= 0:
            errors.append("Temperature must be positive")

        if config.max_new_tokens <= 0:
            errors.append("Max new tokens must be positive")

        return errors

    @staticmethod
    def validate_experiment_config(config: ExperimentRunConfig) -> List[str]:
        """Validate the full experiment configuration."""
        errors = []

        if not config.models:
            errors.append("At least one model must be specified")

        if not config.datasets:
            errors.append("At least one dataset must be specified")

        # Validate each model
        for i, model in enumerate(config.models):
            model_errors = ConfigValidator.validate_model_config(model)
            for error in model_errors:
                errors.append(f"Model {i}: {error}")

        # Validate each dataset
        for i, dataset in enumerate(config.datasets):
            dataset_errors = ConfigValidator.validate_dataset_config(dataset)
            for error in dataset_errors:
                errors.append(f"Dataset {i}: {error}")

        # Validate steering config
        steering_errors = ConfigValidator.validate_steering_config(config.steering)
        for error in steering_errors:
            errors.append(f"Steering: {error}")

        if config.max_concurrent_models <= 0:
            errors.append("Max concurrent models must be positive")

        return errors


class ConfigLoader:
    """Loads and processes experiment configurations."""

    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty configuration file: {filepath}")

        return data

    @staticmethod
    def create_model_configs(models_data: List[Union[str, Dict]]) -> List[ModelConfig]:
        """Create ModelConfig objects from configuration data."""
        configs = []
        for model_data in models_data:
            if isinstance(model_data, str):
                # Simple string format: just model name
                configs.append(ModelConfig(name=model_data))
            elif isinstance(model_data, dict):
                # Dictionary format with parameters
                configs.append(
                    ModelConfig(
                        name=model_data["name"],
                        device=model_data.get("device", "auto"),
                        dtype=model_data.get("dtype", "bfloat16"),
                        batch_size=model_data.get("batch_size", 2),
                        backend=model_data.get("backend", "auto"),
                    )
                )
            else:
                raise ValueError(f"Invalid model configuration: {model_data}")

        return configs

    @staticmethod
    def create_dataset_configs(
        datasets_data: List[Union[str, Dict]],
    ) -> List[DatasetConfig]:
        """Create DatasetConfig objects from configuration data."""
        configs = []
        for dataset_data in datasets_data:
            if isinstance(dataset_data, str):
                # Simple string format: just dataset name
                configs.append(DatasetConfig(name=dataset_data))
            elif isinstance(dataset_data, dict):
                # Dictionary format with parameters
                configs.append(
                    DatasetConfig(
                        name=dataset_data["name"],
                        train_size=dataset_data.get("train_size", 200),
                        test_size=dataset_data.get("test_size", 800),
                        split_seed=dataset_data.get("split_seed", 42),
                    )
                )
            else:
                raise ValueError(f"Invalid dataset configuration: {dataset_data}")

        return configs

    @staticmethod
    def create_steering_config(steering_data: Dict) -> SteeringConfig:
        """Create SteeringConfig from configuration data."""
        return SteeringConfig(
            alpha_range=steering_data.get("alpha_range", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            temperature=steering_data.get("temperature", 0.7),
            max_new_tokens=steering_data.get("max_new_tokens", 100),
        )

    @staticmethod
    def load_experiment_config(filepath: str) -> ExperimentRunConfig:
        """Load complete experiment configuration from YAML file."""
        data = ConfigLoader.load_yaml(filepath)

        # Create model configurations
        models_data = data.get("models", [])
        if not models_data:
            raise ValueError("No models specified in configuration")
        models = ConfigLoader.create_model_configs(models_data)

        # Create dataset configurations
        datasets_data = data.get("datasets", [])
        if not datasets_data:
            raise ValueError("No datasets specified in configuration")
        datasets = ConfigLoader.create_dataset_configs(datasets_data)

        # Create steering configuration
        steering_data = data.get("steering", {})
        steering = ConfigLoader.create_steering_config(steering_data)

        # Create main config
        config = ExperimentRunConfig(
            models=models,
            datasets=datasets,
            steering=steering,
            cache_dir=data.get("cache_dir", "cache"),
            use_cache=data.get("use_cache", True),
            interactive=data.get("interactive", True),
            max_concurrent_models=data.get("max_concurrent_models", 1),
            save_generations=data.get("save_generations", True),
            evaluate_confabulation=data.get("evaluate_confabulation", False),
        )

        # Validate configuration
        errors = ConfigValidator.validate_experiment_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

        return config

    @staticmethod
    def save_experiment_config(config: ExperimentRunConfig, filepath: str):
        """Save experiment configuration to YAML file."""
        # Convert to dictionary
        data = {
            "models": [
                {
                    "name": model.name,
                    "device": model.device,
                    "dtype": model.dtype,
                    "batch_size": model.batch_size,
                    "backend": model.backend,
                }
                for model in config.models
            ],
            "datasets": [
                {
                    "name": dataset.name,
                    "train_size": dataset.train_size,
                    "test_size": dataset.test_size,
                    "split_seed": dataset.split_seed,
                }
                for dataset in config.datasets
            ],
            "steering": {
                "alpha_range": config.steering.alpha_range,
                "temperature": config.steering.temperature,
                "max_new_tokens": config.steering.max_new_tokens,
            },
            "cache_dir": config.cache_dir,
            "use_cache": config.use_cache,
            "interactive": config.interactive,
            "max_concurrent_models": config.max_concurrent_models,
            "save_generations": config.save_generations,
            "evaluate_confabulation": config.evaluate_confabulation,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save to file
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


def create_experiment_configs(
    run_config: ExperimentRunConfig,
) -> List[ExperimentConfig]:
    """Convert ExperimentRunConfig to list of ExperimentConfig objects."""
    experiment_configs = []

    for model in run_config.models:
        for dataset in run_config.datasets:
            exp_config = ExperimentConfig(
                model_name=model.name,
                dataset_name=dataset.name,
                train_size=dataset.train_size,
                test_size=dataset.test_size,
                split_seed=dataset.split_seed,
                alpha_range=run_config.steering.alpha_range,
                temperature=run_config.steering.temperature,
                max_new_tokens=run_config.steering.max_new_tokens,
            )
            experiment_configs.append(exp_config)

    return experiment_configs


def create_default_config() -> ExperimentRunConfig:
    """Create a default experiment configuration."""
    return ExperimentRunConfig(
        models=[
            ModelConfig(name="google/gemma-2-2b-it"),
            ModelConfig(name="google/gemma-2-9b-it"),
        ],
        datasets=[
            DatasetConfig(name="sports_understanding"),
            DatasetConfig(name="social_chemistry"),
        ],
        steering=SteeringConfig(alpha_range=[0, 1, 2, 4, 6, 8]),
    )


def save_default_configs():
    """Save example configuration files."""
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)

    # Basic config (transformer_lens)
    basic_config = ExperimentRunConfig(
        models=[ModelConfig(name="google/gemma-2-9b-it", backend="transformer_lens")],
        datasets=[DatasetConfig(name="sports_understanding")],
        steering=SteeringConfig(alpha_range=[0, 2, 4, 6]),
    )
    ConfigLoader.save_experiment_config(
        basic_config, os.path.join(configs_dir, "basic.yaml")
    )

    # Multi-model config (mixed backends)
    multi_model_config = ExperimentRunConfig(
        models=[
            ModelConfig(name="google/gemma-2-2b-it", batch_size=4, backend="transformer_lens"),
            ModelConfig(name="google/gemma-2-9b-it", batch_size=2, backend="auto"),
        ],
        datasets=[
            DatasetConfig(name="sports_understanding"),
            DatasetConfig(name="social_chemistry"),
        ],
        steering=SteeringConfig(alpha_range=[0, 1, 2, 3, 4, 5, 6, 7, 8]),
        max_concurrent_models=2,
    )
    ConfigLoader.save_experiment_config(
        multi_model_config, os.path.join(configs_dir, "multi_model.yaml")
    )

    # NNsight-specific config with DeepSeek
    nnsight_config = ExperimentRunConfig(
        models=[
            ModelConfig(name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", backend="nnsight", batch_size=1),
            ModelConfig(name="google/gemma-2-9b-it", backend="nnsight", batch_size=2),
        ],
        datasets=[
            DatasetConfig(name="sports_understanding"),
            DatasetConfig(name="logical_deduction"),
        ],
        steering=SteeringConfig(alpha_range=[0, 2, 4, 6, 8]),
    )
    ConfigLoader.save_experiment_config(
        nnsight_config, os.path.join(configs_dir, "nnsight.yaml")
    )

    # Full dataset config with auto backend
    full_config = ExperimentRunConfig(
        models=[ModelConfig(name="google/gemma-2-9b-it", backend="auto")],
        datasets=[
            DatasetConfig(name="sports_understanding"),
            DatasetConfig(name="social_chemistry"),
            DatasetConfig(name="logical_deduction"),
            DatasetConfig(name="quora_question_pairs"),
        ],
        steering=SteeringConfig(alpha_range=list(range(0, 9))),
        evaluate_confabulation=True,
    )
    ConfigLoader.save_experiment_config(
        full_config, os.path.join(configs_dir, "full_datasets.yaml")
    )


def get_recommended_backend_for_model(model_name: str) -> str:
    """
    Get the recommended backend for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Recommended backend name
    """
    model_lower = model_name.lower()
    
    # Models that require specific backends
    if "deepseek" in model_lower:
        return "nnsight"  # DeepSeek models require nnsight
    elif "gpt2" in model_lower and "openai-community" in model_lower:
        return "transformer_lens"  # GPT-2 works well with transformer_lens
    elif any(pattern in model_lower for pattern in ["gemma", "llama", "mistral"]):
        return "auto"  # These models work with both, try auto-selection
    else:
        return "auto"  # Default to auto-detection


def validate_model_backend_compatibility(model: ModelConfig) -> List[str]:
    """
    Validate backend compatibility for a model configuration.
    
    Args:
        model: ModelConfig instance
        
    Returns:
        List of warning messages
    """
    warnings = []
    model_lower = model.name.lower()
    
    # Check for known incompatibilities
    if "deepseek" in model_lower and model.backend == "transformer_lens":
        warnings.append(
            f"DeepSeek models ('{model.name}') are not supported by transformer_lens. "
            "Consider using backend 'nnsight' instead."
        )
    
    # Check for suboptimal choices
    if "gpt2" in model_lower and model.backend == "nnsight":
        warnings.append(
            f"GPT-2 models ('{model.name}') work better with transformer_lens backend. "
            "Consider using backend 'transformer_lens' for better performance."
        )
    
    return warnings


def suggest_backend_optimization(config: ExperimentRunConfig) -> Dict[str, Any]:
    """
    Analyze experiment configuration and suggest backend optimizations.
    
    Args:
        config: ExperimentRunConfig instance
        
    Returns:
        Dictionary with optimization suggestions
    """
    suggestions = {
        "warnings": [],
        "recommendations": [],
        "optimized_models": []
    }
    
    for model in config.models:
        # Check compatibility
        warnings = validate_model_backend_compatibility(model)
        suggestions["warnings"].extend(warnings)
        
        # Generate recommendations
        recommended = get_recommended_backend_for_model(model.name)
        if model.backend == "auto" and recommended != "auto":
            suggestions["recommendations"].append(
                f"Model '{model.name}': Consider explicitly setting backend to '{recommended}' "
                "for better performance."
            )
        elif model.backend != recommended and recommended != "auto":
            suggestions["recommendations"].append(
                f"Model '{model.name}': Recommended backend is '{recommended}', "
                f"currently set to '{model.backend}'."
            )
        
        # Create optimized model config
        optimized_model = ModelConfig(
            name=model.name,
            device=model.device,
            dtype=model.dtype,
            batch_size=model.batch_size,
            backend=recommended if recommended != "auto" else model.backend
        )
        suggestions["optimized_models"].append(optimized_model)
    
    return suggestions


if __name__ == "__main__":
    # Create example configuration files
    save_default_configs()
    print("Example configuration files created in 'configs/' directory")
