# Post-Hoc Reasoning Repository Documentation

## Parallel Execution (RTX A6000 Optimized)

The repository supports parallel experiment execution optimized for RTX A6000 GPU (48GB VRAM).

### Usage

```bash
# Parallel execution with existing config
python3 run_transformer_lens_experiments.py --config configs/transformer_lens.yaml --parallel --max-concurrent 3

# Parallel execution with optimized config
python3 run_transformer_lens_experiments.py --config configs/transformer_lens_parallel.yaml --parallel

# Override max concurrent models
python3 run_transformer_lens_experiments.py --config configs/transformer_lens.yaml --parallel --max-concurrent 4
```

### Performance

- **Sequential**: ~2 hours for 24 experiments (6 models × 4 datasets)
- **Parallel**: ~40 minutes for same experiments (3x speedup)
- **Memory-aware scheduling**: Automatically groups experiments by model size

### Implementation Details

- Small models (gemma-2b, qwen3b, phi3): 5-8GB VRAM each, can run 4-6 concurrently
- Medium models (qwen7b, llama7b): ~14GB VRAM each, can run 2-3 concurrently  
- Large models (gemma-9b): ~18GB VRAM each, can run 2 concurrently
- Buffer: 3GB reserved from 48GB total for system overhead

## Results System

The repository provides a comprehensive results analysis system through the `src/results.py` module.

### Loading Results

```python
from src.results import Results

# Load all results from cache
results = Results("cache/experiments")

# Access different result types
steering_results = results.steering      # Steering experiment results
generation_results = results.generation  # Train/test generation results  
probe_results = results.probe            # Probe training results
debiasing_results = results.debiasing    # ACE debiasing experiment results
```

### Available Metrics

#### Probe Results
```python
# Get best probe layer and AUC score
best_layer, best_auc = results.probe.get_best_layer(model="google/gemma-2-2b-it", dataset="sports_understanding")

# Get AUC scores for all layers
auc_curve = results.probe.get_auc_curve(model="google/gemma-2-2b-it", dataset="sports_understanding")

# Get probe coefficients
coefficients = results.probe.get_coefficients(model="google/gemma-2-2b-it", dataset="sports_understanding", layer=12)
```

#### Generation Results (Train/Test Accuracy)
```python
# Get train and test accuracy
train_acc = results.generation.get_accuracy(model="google/gemma-2-2b-it", dataset="sports_understanding", split="train")
test_acc = results.generation.get_accuracy(model="google/gemma-2-2b-it", dataset="sports_understanding", split="test")

# Get error analysis
errors = results.generation.get_errors(model="google/gemma-2-2b-it", dataset="sports_understanding", split="test")
```

#### Steering Results
```python
# Get steering success rate for specific alpha
success_rate = results.steering.get_success_rate(
    model="google/gemma-2-2b-it",
    dataset="sports_understanding", 
    direction="yes",  # or "no"
    alpha=10
)

# Get parse rate (how many outputs were parseable)
parse_rate = results.steering.get_parse_rate(
    model="google/gemma-2-2b-it",
    dataset="sports_understanding",
    direction="yes",
    alpha=10
)

# Get summary statistics across all alphas
stats = results.steering.get_summary_stats()
```

### Comprehensive Summary

```python
# Get full summary with all metrics
summary = results.get_summary()

# Contains:
# - summary['models']: List of all models
# - summary['datasets']: List of all datasets  
# - summary['accuracy_summary']: DataFrame with train/test accuracies
# - summary['probe_summary']: DataFrame with best layers and AUC scores
```

### Exporting Results

```python
# Export all results to CSV files
results.export_all("output_dir/")
# Creates:
# - steering_results.csv
# - generation_results.csv
# - probe_results.csv

# Export steering results in template format
results.steering.to_template_format("results_formatted.csv")
```

### Data Structure

The results are organized hierarchically in the cache:

```
cache/experiments/
  {model_name}/
    {dataset_name}/
      split_{seed}_{train_size}_{test_size}/
        {experiment_hash}/
          data/
            - train_generations.pkl  # Train set responses
            - test_generations.pkl   # Test set responses
            - train_activations.pkl  # Neural activations
            - test_activations.pkl
          probes/
            - auc_scores.json       # AUC for each layer
            - coefficients.pkl      # Probe weights
          steering/
            - steering_alpha_{alpha}_{direction}.pkl  # Steering results
```

### Key Metrics Explained

- **Train/Test Accuracy**: Percentage of correct yes/no predictions on train/test sets
- **Best Probe Layer**: Layer with highest AUC score for detecting reasoning
- **Probe AUC**: Area under ROC curve, measures probe's ability to detect reasoning (0.5 = random, 1.0 = perfect)
- **Steering Success Rate**: Percentage of successful steering (flipping answer from yes→no or no→yes)
- **Parse Rate**: Percentage of model outputs that could be parsed for yes/no answer

### Example Analysis Script

```python
from src.results import Results

# Load results
results = Results("cache/experiments")

# Print summary for all experiments
for model in results.get_summary()['models']:
    for dataset in results.get_summary()['datasets']:
        # Get metrics
        train_acc = results.generation.get_accuracy(model=model, dataset=dataset, split="train")
        test_acc = results.generation.get_accuracy(model=model, dataset=dataset, split="test")
        best_layer, best_auc = results.probe.get_best_layer(model, dataset)
        
        if best_layer is not None:
            print(f"\n{model} - {dataset}:")
            print(f"  Train Accuracy: {train_acc:.1f}%")
            print(f"  Test Accuracy: {test_acc:.1f}%")
            print(f"  Best Probe Layer: {best_layer}")
            print(f"  Best Probe AUC: {best_auc:.3f}")
```
