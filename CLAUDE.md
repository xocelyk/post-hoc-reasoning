# Post-Hoc Reasoning in Chain-of-Thought: Implementation Guide

## Research Overview

This repository implements the experiments from "Post-Hoc Reasoning in Chain-of-Thought: Evidence from Pre-CoT Probes and Activation Steering" (ICLR 2026 submission). The work investigates whether language models engage in **post-hoc reasoning** - determining their answer before generating chain-of-thought explanations, then rationalizing that pre-committed answer.

## Key Research Questions

1. **Do models pre-commit to answers before CoT?** 
   - Tested via linear probes on pre-CoT activations
   - If answers are decodable before reasoning begins, suggests pre-commitment

2. **Are these pre-CoT features causally relevant?**
   - Tested via activation steering during generation
   - If steering along probe directions changes answers, features are causal

3. **How does post-hoc reasoning manifest in CoT?**
   - Analyzed via classification of steered CoT traces
   - Identifies confabulation vs non-entailment patterns

## Codebase Architecture

### Core Experimental Pipeline

```
data/ → experiment_runner.py → cache/experiments/ → results.py → visualizations/
         ↓                      ↓
      reasoning_probes.py    steering_methods.py
```

### Key Components

#### 1. Data Pipeline (`data_loading.py`)

- **Datasets**: anachronisms, logical_deduction, social_chemistry, sports_understanding
- **Format**: Binary yes/no questions with 4-shot CoT demonstrations
- **Randomization**: Answer position (A/B) randomized to control for bias
- **Split**: 500 train / 500 test examples per dataset

#### 2. Probe Training (`reasoning_probes.py`)

Implements difference-of-means probes:
```python
# Compute mean activations for each class
μ_yes = mean(activations[labels=="yes"])
μ_no = mean(activations[labels=="no"]) 
# Probe direction
w = μ_yes - μ_no
```

Key features:
- Extracts activations at last pre-CoT token (":' in "Let's think step by step:")
- Computes probes for all layers
- Evaluates via cosine similarity and AUC scores
- Identifies best layer for steering

#### 3. Steering Methods (`steering_methods.py`)

Three steering strategies implemented:

**CAA Single Layer** (paper's primary method):
- Uses best-performing probe layer
- No normalization applied
- Replicates vector across all layers for compatibility

**CAA Layer Incremental**:
- Distributes concept edits across layers
- Applies RMS normalization to incremental differences
- More nuanced intervention approach

**Logistic Regression**:
- Trains classifiers per layer
- Uses coefficient vectors for steering
- Alternative to difference-of-means approach

#### 4. Generation with Steering (`nnsight_steering.py`, `utils.py`)

Applies steering during autoregressive generation:
```python
# At each generation step, for each layer:
activations[layer] += alpha * steering_vector[layer]
```

Parameters:
- `alpha`: Steering strength (positive→"yes", negative→"no")
- Range: typically 0 to ±20 in steps of 2
- Early stopping when parse failures exceed threshold

#### 5. Results Analysis (`results.py`)

Unified interface for analyzing experiments:

**SteeringResults**: 
- Success rate: % of flipped answers
- Parse rate: % of valid outputs
- Breakdown by model, dataset, direction, alpha

**GenerationResults**:
- Train/test accuracy without steering
- Baseline model performance

**ProbeResults**:
- AUC scores per layer
- Best probe layer identification
- Probe coefficient analysis

#### 6. CoT Classification

Post-hoc analysis of steered generations:

| Pattern | Premises | Conclusion | Interpretation |
|---------|----------|------------|----------------|
| Confabulation | False | Follows | Model fabricates facts for steered answer |
| Non-entailment | True | Doesn't follow | Model ignores reasoning for steered answer |
| Hallucination | False | Doesn't follow | Complete breakdown |
| Sound | True | Follows | Should not occur under steering |

### Experimental Configurations

#### Standard Config (`configs/transformer_lens_by_model_clean/`)
```yaml
steering:
  method: caa-single-layer
  alpha_range: [0, 2, 4, ..., 20]
  temperature: 0.7
  max_new_tokens: 200
```

#### Orthogonal Baseline (`configs/orthogonal_steering/`)
- Tests steering with random orthogonal directions
- Controls for generic perturbation effects
- Same norm as probe vectors but orthogonal direction

### Parallel Execution

The codebase supports parallel model execution:
```bash
python run_transformer_lens_experiments.py --parallel --max-concurrent 3
```

Memory-aware scheduling:
- Small models (2-3B): 5-8GB VRAM each
- Medium models (7B): ~14GB VRAM each  
- Large models (9B): ~18GB VRAM each

### Caching System

Hierarchical cache structure:
```
cache/experiments/
  {model}/
    {dataset}/
      split_{seed}_{train_size}_{test_size}/
        {experiment_hash}/
          data/
            - train_generations.pkl
            - test_generations.pkl
            - train_activations.pkl
            - test_activations.pkl
          probes/
            - auc_scores.json
            - coefficients.pkl
          steering/
            - steering_alpha_{α}_{direction}.pkl
```

## Research Findings

### Key Results from Implementation

1. **Strong Pre-CoT Probes**: AUC > 0.9 for most model-dataset pairs (except Logical Deduction)

2. **Successful Steering**: >50% answer flips at moderate α values

3. **Task-Dependent Effects**: 
   - Logical Deduction shows weaker probes/steering
   - Suggests genuine CoT use for logical tasks
   - Factual/social tasks show stronger post-hoc patterns

4. **Model Size Effects**:
   - Larger models more robust to random perturbations
   - But equally susceptible to targeted steering
   - Suggests feature sparsity increases with scale

### Implications for AI Safety

1. **CoT Monitoring Limitations**: Chain-of-thought may not reflect true reasoning process

2. **Deceptive Alignment Risk**: Models could rationalize predetermined answers

3. **Potential Mitigation**: 
   - Probe-based detection of post-hoc reasoning
   - Steering to correct unfaithful reasoning
   - Training methods to encourage faithful CoT

## Usage Examples

### Running Full Pipeline
```bash
# Single model experiments
python src/main.py --config configs/transformer_lens_by_model_clean/gemma-2-2b-it.yaml

# Parallel execution across models  
python run_transformer_lens_experiments.py --config configs/transformer_lens.yaml --parallel

# Orthogonal baseline experiments
python run_orthogonal_steering_experiments.py --config configs/orthogonal_steering/gemma-2-2b-it.yaml
```

### Analyzing Results
```python
from src.results import Results

# Load all results
results = Results("cache/experiments")

# Get probe performance
best_layer, auc = results.probe.get_best_layer(
    model="google/gemma-2-2b-it",
    dataset="sports_understanding"
)

# Get steering success rate
success_rate = results.steering.get_success_rate(
    model="google/gemma-2-2b-it",
    dataset="sports_understanding",
    direction="yes",
    alpha=10
)

# Export for analysis
results.export_all("output/")
```

### Visualizing Results
```bash
# Generate steering plots
python scripts/analyze_experiments.py

# View probe AUC curves
python src/visualizer.py --plot-type probe-auc

# Generate CoT classification analysis
jupyter notebook classify_rollouts.ipynb
```

## Technical Notes

### Model Backends

- **TransformerLens**: Primary backend for Gemma and smaller models
- **nnsight**: Cleaner intervention API, better for larger models
- Both support activation caching and efficient batch processing

### Memory Optimization

- Activation caching trades disk for RAM
- KV cache persistence across steering values
- Garbage collection between experiments
- FP16/BF16 precision options

### Known Limitations

1. **Parsing Failures**: High α values cause degenerate outputs
2. **Template Dependence**: Results sensitive to prompt format
3. **Compute Requirements**: Full experiments need 48GB+ VRAM
4. **Dataset Size**: 500 examples may limit statistical power