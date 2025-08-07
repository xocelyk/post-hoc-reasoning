# Post-Hoc Reasoning Repository Guide

## Overview
This repository implements post-hoc reasoning experiments for language models, focusing on probe training and activation steering to modify model behavior. The system extracts internal representations from LLMs, trains classifiers (probes) to predict reasoning patterns, and uses these probes to steer model outputs.

## Repository Structure

### Core Directories
- **`src/`** - Main source code for experiments (reorganized)
  - **`backends/`** - Backend-specific implementations (transformer_lens, nnsight)
  - **`methods/`** - Shared algorithms (steering methods, probes)
  - **`core/`** - Core utilities (config, caching, data loading)
  - **`integrations/`** - External service integrations (W&B)
  - **`runners/`** - Experiment runners (unified runner)
- **`configs/`** - YAML configuration files for experiments
- **`data/`** - Dataset files in JSON format
- **`cache/`** - Cached experiment results and model outputs
- **`scripts/`** - Analysis and utility scripts
- **`claude-scratchpad/`** - Data generation and review scripts

### Main Entry Points

#### 1. `run_nnsight_experiments.py`
Primary script for running experiments using the NNsight backend (recommended for newer models like DeepSeek).

**Usage:**
```bash
# Run with config file
python run_nnsight_experiments.py --config configs/nnsight.yaml

# Resume incomplete experiments
python run_nnsight_experiments.py --resume

# Create example configs
python run_nnsight_experiments.py --create-configs

# Override specific parameters
python run_nnsight_experiments.py --config configs/nnsight.yaml --models deepseek-ai/DeepSeek-R1-Distill-Llama-8B --datasets sports_understanding
```

#### 2. `run_transformer_lens_experiments.py`
Alternative script supporting both TransformerLens and NNsight backends, with automatic backend selection.

**Usage:**
```bash
# Similar usage pattern as nnsight script
python run_transformer_lens_experiments.py --config configs/transformer_lens.yaml

# Force specific backend
python run_transformer_lens_experiments.py --config configs/basic.yaml --backend nnsight

# Use unified runner (experimental)
python run_transformer_lens_experiments.py --config configs/basic.yaml --unified
```

## Architecture

### Experiment Flow
1. **Configuration Loading** - Parse YAML config specifying models, datasets, and parameters
2. **Data Generation** - Generate model outputs and extract activations
3. **Probe Training** - Train classifiers on activations to predict reasoning patterns
4. **Steering** - Apply learned vectors to modify model behavior
5. **Evaluation** - Assess steering effectiveness across different alpha values

### Key Components

#### Configuration System (`src/core/config.py`)
- `ExperimentRunConfig` - Top-level configuration
- `ModelConfig` - Model-specific settings (backend, device, dtype)
- `DatasetConfig` - Dataset parameters (train/test split, seed)
- `SteeringConfig` - Steering methods and hyperparameters

#### Backend Implementations (`src/backends/`)
- **`transformer_lens/`** - TransformerLens-specific implementation
  - `models.py` - ChatModel wrapper
  - `utils.py` - Activation extraction and generation
  - `adapter.py` - Backend adapter interface
- **`nnsight/`** - NNsight-specific implementation
  - `models.py` - NNsightChatModel wrapper
  - `steering.py` - Steering implementation
  - `unified/`** - Advanced NNsight utilities (formerly nnsight_utils)

#### Shared Methods (`src/methods/`)
- **`steering/`** - Backend-agnostic steering algorithms:
  - `caa_single_layer.py` - CAA single-layer steering
  - `caa_incremental.py` - CAA layer-incremental steering
  - `logistic_steering.py` - Logistic regression steering

#### Data Loading (`src/core/data_loading.py`)
Handles various dataset formats:
- **Sports Understanding** - Yes/no questions about sports scenarios
- **Anachronisms** - Detecting temporal inconsistencies
- **Logical Deduction** - Multi-step reasoning problems
- **Social Chemistry** - Appropriateness judgments
- **Snarks** - Sarcasm detection
- **Quora Question Pairs** - Duplicate detection

### Probe Methods

#### 1. Logistic Regression (`logistic-regression`)
- Trains linear classifiers at each layer
- Uses coefficients as steering vectors
- Steers at all layers simultaneously

#### 2. CAA Single Layer (`caa-single-layer`)
- Computes mean activation differences between classes
- Identifies best single layer for steering
- Applies steering only at optimal layer

#### 3. CAA Multi-Layer (`caa-layer-incremental`)
- Computes incremental differences across layers
- Applies RMS normalization
- Steers at all layers with normalized vectors

### Caching System
- **Experiment Cache** - Stores generations, activations, probes, and steering results
- **KV Cache** - Optimizes steering by caching key-value pairs
- **Resume Capability** - Continue interrupted experiments from checkpoint

## Configuration Files

### Example Config Structure
```yaml
models:
  - name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    backend: auto  # auto, nnsight, or transformer_lens
    device: auto
    dtype: bfloat16
    batch_size: 1

datasets:
  - name: sports_understanding
    train_size: 100
    test_size: 100
    split_seed: 42

steering:
  method: caa-single-layer  # or caa-layer-incremental, logistic-regression
  alpha_range: [0, 2, 4, 6, 8, 10]
  temperature: 0.7
  max_new_tokens: 200
  max_gen: 10  # Limit steering examples per alpha

cache_dir: cache
use_cache: true
interactive: false
```

## Working with DeepSeek Models
DeepSeek models have special handling:
- **Extended Generation** - `max_new_tokens` automatically set to 2000
- **Think Tags** - Internal reasoning wrapped in `<think>` tags, filtered for display
- **Backend Selection** - Automatically uses NNsight backend

## Analysis Tools

### Visualization (`scripts/analyze_experiments.py`)
Analyzes cached results and generates performance plots.

### W&B Integration (`src/integrations/wandb.py`)
Optional Weights & Biases logging for experiment tracking.

### Quick Summary (`scripts/quick_summary.py`)
Generates concise summaries of experiment results.

## Common Workflows

### Running a Complete Experiment
1. Create/modify a config file in `configs/`
2. Run experiment: `python run_nnsight_experiments.py --config configs/your_config.yaml`
3. Monitor progress in console and logs
4. Analyze results: `python scripts/analyze_experiments.py`

### Adding a New Dataset
1. Add JSON data file to `data/dataset_name/`
2. Implement format function in `src/data_loading.py`
3. Add dataset to config file
4. Run experiments

### Debugging Failed Experiments
1. Check logs in `cache/logs/`
2. Use `--resume` to continue from failure point
3. Enable verbose mode with `-v` flag
4. Examine cached intermediate results

## Performance Considerations
- **Memory Management** - Automatic garbage collection and cache clearing
- **Batch Processing** - Configurable batch sizes per model
- **Concurrent Experiments** - Control with `max_concurrent_models`
- **KV Caching** - Significant speedup for steering generation

## Important Files
- **`DATASETS.md`** - Documentation of dataset formats
- **`wandb_organization_guide.md`** - W&B setup instructions
- **`requirements.txt`** - Python dependencies
- **`writing/blog-post.md`** - Research context and findings

## Notes for Development
- Always check existing caches before regenerating data
- Use `--no-cache` sparingly (only for debugging)
- Monitor GPU memory usage for large models
- Test configuration changes with small datasets first
- Steering effectiveness varies significantly by model and dataset

## Recent Reorganization (August 2025)
The codebase has been reorganized for better maintainability:
- **Backend separation**: TransformerLens and NNsight code now in separate directories
- **Shared algorithms**: Steering methods extracted to `methods/` directory
- **Core utilities**: Common code centralized in `core/` directory
- **Unified runner**: New `--unified` flag for experimental unified runner
- **Clean imports**: All imports updated to use new structure
- **Backward compatible**: All existing functionality preserved