# Post-Hoc Reasoning Experiments

A comprehensive framework for conducting post-hoc reasoning experiments using probe training and activation steering across different language model backends.

## Overview

This repository provides tools for:
- **Probe Training**: Train linear probes on model activations to detect reasoning patterns
- **Activation Steering**: Modify model behavior using learned probe vectors
- **Multi-Backend Support**: Use either transformer_lens or nnsight for broader model compatibility
- **Comprehensive Caching**: Hierarchical caching system for efficient experiment management
- **Interactive Visualization**: Real-time experiment progress tracking

## Supported Models

### Transformer_lens Backend
- GPT-2 variants (`openai-community/gpt2`, `gpt2-medium`, etc.)
- Gemma models (`google/gemma-2-2b-it`, `google/gemma-2-9b-it`)
- Selected models with transformer_lens support

### NNsight Backend  
- **DeepSeek models** (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`)
- **Any HuggingFace model** (Llama, Mistral, Qwen, etc.)
- Gemma models (broader compatibility)
- Meta-Llama models (`meta-llama/Llama-2-7b-chat-hf`)
- Mistral models (`mistralai/Mistral-7B-Instruct-v0.1`)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd post-hoc-reasoning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install backend-specific packages**:
   
   For transformer_lens support:
   ```bash
   pip install transformer_lens
   ```
   
   For nnsight support:
   ```bash
   pip install nnsight
   ```

## Quick Start

### 1. Create Example Configurations
```bash
python run_experiments.py --create-configs
```

This creates several example configuration files in the `configs/` directory:
- `configs/basic.yaml` - Basic transformer_lens setup
- `configs/nnsight.yaml` - NNsight with DeepSeek models
- `configs/multi_model.yaml` - Multiple models with mixed backends
- `configs/full_datasets.yaml` - All datasets with auto backend

### 2. Run Experiments

**Basic usage with configuration file**:
```bash
python run_experiments.py --config configs/basic.yaml
```

**Use nnsight backend for broader model support**:
```bash
python run_experiments.py --config configs/nnsight.yaml
```

**Override backend for all models**:
```bash
python run_experiments.py --config configs/basic.yaml --backend nnsight
```

**Run with specific models**:
```bash
python run_experiments.py --config configs/basic.yaml \
  --models google/gemma-2-9b-it deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --backend auto
```

### 3. Resume Incomplete Experiments
```bash
python run_experiments.py --resume
```

### 4. List Existing Experiments
```bash
python run_experiments.py --list-experiments
```

## Configuration

### YAML Configuration Format

```yaml
# Example configuration file
models:
  - name: "google/gemma-2-9b-it"
    backend: "auto"        # auto, nnsight, transformer_lens  
    device: "auto"         # auto, cpu, cuda, mps
    dtype: "bfloat16"      # bfloat16, float16, float32
    batch_size: 2

  - name: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    backend: "nnsight"     # DeepSeek requires nnsight
    batch_size: 1

datasets:
  - name: "sports_understanding"
    train_size: 200
    test_size: 800
    split_seed: 42

  - name: "logical_deduction"

steering:
  alpha_range: [0, 2, 4, 6, 8]
  temperature: 0.7
  max_new_tokens: 100

# Runtime settings
cache_dir: "cache"
use_cache: true
interactive: true
max_concurrent_models: 1
save_generations: true
```

### Backend Selection

The framework automatically selects the appropriate backend:

- **`auto`**: Tries nnsight first (broader support), falls back to transformer_lens
- **`nnsight`**: Uses nnsight backend explicitly (required for DeepSeek models)
- **`transformer_lens`**: Uses transformer_lens backend explicitly

### Supported Datasets

- `sports_understanding` - Sports reasoning questions
- `logical_deduction` - Logical reasoning tasks  
- `social_chemistry` - Social reasoning scenarios
- `quora_question_pairs` - Question similarity tasks

## Backend Comparison

| Feature | transformer_lens | nnsight |
|---------|-------------------|---------|
| **Model Support** | Limited (GPT-2, some Gemma) | Any HuggingFace model |
| **DeepSeek Support** | ❌ No | ✅ Yes |
| **API Complexity** | Complex hooks system | Clean intervention API |
| **Documentation** | Extensive | Growing |
| **Performance** | Optimized for supported models | Good general performance |
| **Interpretability Tools** | Rich toolkit | Basic tools |

## Command Line Options

### Basic Commands
```bash
# Run with configuration
python run_experiments.py --config configs/basic.yaml

# Create example configs  
python run_experiments.py --create-configs

# Resume incomplete experiments
python run_experiments.py --resume

# List all experiments
python run_experiments.py --list-experiments
```

### Configuration Overrides
```bash
# Override models
python run_experiments.py --config configs/basic.yaml \
  --models google/gemma-2-9b-it deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Override backend
python run_experiments.py --config configs/basic.yaml --backend nnsight

# Override datasets
python run_experiments.py --config configs/basic.yaml \
  --datasets sports_understanding logical_deduction

# Override training parameters
python run_experiments.py --config configs/basic.yaml \
  --train-size 100 --test-size 400 \
  --alpha-range 0 2 4 6
```

### Runtime Options
```bash
# Disable interactive visualization
python run_experiments.py --config configs/basic.yaml --no-interactive

# Custom cache directory
python run_experiments.py --config configs/basic.yaml --cache-dir /path/to/cache

# Save results summary
python run_experiments.py --config configs/basic.yaml \
  --output-summary results.csv
```

## Experiment Workflow

Each experiment consists of three phases:

### 1. Data Generation
- Loads datasets and creates train/test splits
- Generates model responses for all prompts
- Extracts residual stream activations
- **Cached**: Results are saved and reused

### 2. Probe Training  
- Trains linear probes on activations for each layer
- Evaluates probe performance using AUC scores
- Identifies best layers for steering
- **Cached**: Trained probes are saved

### 3. Activation Steering
- Uses probe vectors to steer model behavior
- Tests different steering strengths (alpha values)
- Measures steering success rates
- **Cached**: Steering results are saved

## Caching System

The framework uses a hierarchical caching system:

```
cache/
├── experiments/
│   ├── google_gemma-2-9b-it/
│   │   ├── sports_understanding/
│   │   │   └── split_42_200_800/
│   │   │       ├── generations/
│   │   │       ├── activations/
│   │   │       ├── probes/
│   │   │       └── steering/
│   │   └── logical_deduction/
│   └── deepseek-ai_DeepSeek-R1-Distill-Llama-8B/
└── logs/
```

Benefits:
- **Efficiency**: Skip completed experiments
- **Resumability**: Continue from any point
- **Reproducibility**: Consistent results across runs
- **Storage**: Organized by model, dataset, and configuration

## Advanced Usage

### Custom Model Integration

To add support for new models with nnsight:

```python
# The nnsight backend supports any HuggingFace model
model_config = ModelConfig(
    name="your-org/your-model",
    backend="nnsight",
    batch_size=1
)
```

### Custom Chat Templates

For models requiring special chat formatting:

```python
# Extend the format registry in nnsight_models.py
def _format_turns_custom(self, messages):
    # Custom formatting logic
    return formatted_messages

# Register in __init__
self.format_registry["your-model"] = self._format_turns_custom
```

### Experiment Analysis

Access experiment results programmatically:

```python
from cache_manager import ExperimentManager

manager = ExperimentManager("cache")
summary = manager.get_experiments_summary()
print(summary)
```

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Check if model requires specific backend
python -c "from model_factory import get_recommended_backend; print(get_recommended_backend('your-model'))"
```

**2. CUDA Out of Memory**
```yaml
# Reduce batch size in config
models:
  - name: "large-model"
    batch_size: 1
    dtype: "bfloat16"
```

**3. Backend Compatibility**
```bash
# Use auto backend for maximum compatibility
python run_experiments.py --config configs/basic.yaml --backend auto
```

### Backend-Specific Issues

**transformer_lens Issues**:
- Limited model support
- Complex hook system debugging
- Solution: Use `--backend nnsight` for broader compatibility

**nnsight Issues**:
- Newer framework with evolving API
- Less community documentation
- Solution: Check model compatibility with HuggingFace

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Adding New Datasets

1. Add dataset loading logic to `src/data_loading.py`
2. Update available datasets list
3. Test with existing experiment pipeline

### Adding New Backends

1. Create new model wrapper in `src/`
2. Update `model_factory.py` with backend support
3. Add configuration validation
4. Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{post_hoc_reasoning,
  title={Post-Hoc Reasoning Experiments Framework},
  author={[Author Name]},
  year={2024},
  url={[Repository URL]}
}
```