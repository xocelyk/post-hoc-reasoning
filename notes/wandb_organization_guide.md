# W&B Organization Guide for Post-Hoc Reasoning Experiments

## Hierarchy Structure

```
Entity (Team/Personal)
└── Project (e.g., "post-hoc-reasoning")
    └── Runs (Individual Experiments)
        └── Metrics, Artifacts, Visualizations
```

## 1. Run Naming Convention

Each run should be automatically named based on the experiment configuration:

```
{model}-{dataset}-{method}-{timestamp}

Examples:
- gemma-2b-sports-caa-single-layer-20241206-143022
- deepseek-1.5b-logical-deduction-logistic-regression-20241206-145512
```

## 2. Run Organization with Tags

Use tags to categorize runs for easy filtering:

```python
tags = [
    model_name.split('/')[-1],      # e.g., "gemma-2b"
    dataset_name,                    # e.g., "sports_understanding"
    steering_method,                 # e.g., "caa-single-layer"
    f"split_{split_seed}",          # e.g., "split_42"
    "transformer_lens" or "nnsight" # backend used
]
```

## 3. Metric Organization

### Probe Training Metrics (Grouped by Layer)
```
probe/caa-single-layer/train_auc     # Per layer
probe/caa-single-layer/test_auc      # Per layer
probe/caa-single-layer/similarity_score # Per layer
probe/caa-single-layer/best_layer    # Single value
probe/caa-single-layer/best_score    # Single value
```

### Steering Metrics (Grouped by Alpha/Direction)
```
steering/summary/alpha_-10_yes/success_rate
steering/summary/alpha_-10_yes/failure_rate
steering/summary/alpha_-10_yes/unparsed_rate
steering/summary/alpha_10_no/success_rate
steering/summary/alpha_10_no/failure_rate
steering/summary/alpha_10_no/unparsed_rate
```

### Individual Example Tracking
```
steering/individual/alpha         # Logged per example
steering/individual/success      # Boolean
steering/individual/parsable     # Boolean
steering/individual/answer_changed # Boolean
```

## 4. Visualization Organization

### Plots
- `probe/caa-single-layer/layer_scores_plot` - AUC by layer
- `steering/success_rate_plot` - Success rate vs alpha for both directions

### Tables
- `steering/results_table/alpha_-10_yes` - Example results for specific conditions
- `steering/examples/alpha_-10_yes` - Detailed HTML examples

## 5. Configuration Tracking

All experiment parameters are automatically logged:
```python
config = {
    # Model Configuration
    "model_name": "google/gemma-2-2b-it",
    "model_backend": "transformer_lens",
    
    # Dataset Configuration
    "dataset_name": "sports_understanding",
    "train_size": 500,
    "test_size": 200,
    "split_seed": 42,
    
    # Method Configuration
    "steering_method": "caa-single-layer",
    "alpha_range": [-10, -5, -2, -1, 0, 1, 2, 5, 10],
    
    # Generation Configuration
    "temperature": 0.7,
    "max_new_tokens": 300,
}
```

## 6. Artifacts for Complete Results

Save complete results as artifacts:
```
Artifacts:
├── probe_results_{model}_{dataset}.json
├── steering_results_{model}_{dataset}.pkl
└── experiment_cache_{model}_{dataset}.tar.gz
```

## 7. Comparing Experiments

W&B makes it easy to compare across:

### Different Models on Same Dataset
- Filter by tag: `dataset_name == "sports_understanding"`
- Compare: Success rates across models

### Different Methods on Same Model/Dataset
- Filter by: `model == "gemma-2b" AND dataset == "sports"`
- Compare: CAA vs Logistic Regression performance

### Different Alpha Values
- Use the success rate plot
- Table view of all alpha conditions

## 8. Example W&B Queries

### Find Best Performing Configuration
```
runs.summary.best_steering_success_rate > 0.8
```

### Find Experiments with High Unparsed Rate
```
runs.summary.avg_unparsed_rate > 0.2
```

### Compare Specific Models
```
tags IN ["gemma-2b", "deepseek-1.5b"] AND 
dataset == "logical_deduction"
```

## 9. Team Collaboration Features

### Shared Views
Create saved views for common comparisons:
- "Model Comparison - Sports Dataset"
- "CAA Method Performance"
- "Unparsed Response Analysis"

### Reports
Create reports to document findings:
- Method comparison results
- Best configurations per dataset
- Failure mode analysis

### Comments
Comment on specific runs or visualizations to discuss with your colleague.

## 10. Best Practices

1. **Consistent Naming**: Let the automatic naming handle run names
2. **Rich Logging**: Log both aggregated metrics and individual examples
3. **Meaningful Tags**: Use tags that help filter experiments
4. **Regular Artifacts**: Save complete results for offline analysis
5. **Descriptive Notes**: Add notes to runs about any issues or observations

## Example Dashboard Layout

```
┌─────────────────────────────────────────┐
│ Project: post-hoc-reasoning             │
├─────────────────────────────────────────┤
│ Filters:                                │
│ - Model: [gemma-2b] [deepseek] [All]   │
│ - Dataset: [sports] [logical] [All]     │
│ - Method: [caa-single] [logistic] [All] │
├─────────────────────────────────────────┤
│ Success Rate vs Alpha Plot              │
│ [Interactive plot showing all runs]     │
├─────────────────────────────────────────┤
│ Best Configurations Table               │
│ Model | Dataset | Method | Best α | SR  │
├─────────────────────────────────────────┤
│ Recent Steering Examples                │
│ [Table with prompt->response samples]   │
└─────────────────────────────────────────┘
```

This organization makes it easy to:
- Track experiments in real-time
- Compare different configurations
- Identify best performing setups
- Debug issues with specific examples
- Collaborate with your colleague effectively