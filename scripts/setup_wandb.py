#!/usr/bin/env python3
"""
Setup guide and example for integrating W&B into post-hoc reasoning experiments.

This script will help you:
1. Set up W&B authentication
2. Configure project settings
3. Show example integration code
"""

import os
import sys

def setup_instructions():
    """Print setup instructions for W&B."""
    print("="*60)
    print("W&B SETUP INSTRUCTIONS")
    print("="*60)
    print("""
1. First, install wandb if not already installed:
   pip install wandb

2. Login to W&B (one-time setup):
   wandb login
   
   This will prompt you for your API key. Get it from:
   https://wandb.ai/authorize
   
3. Set up your project:
   - Ask your colleague for the project name (e.g., "post-hoc-reasoning")
   - Ask for the team/entity name if using a shared workspace
   
4. Environment variables (optional but recommended):
   export WANDB_PROJECT="your-project-name"
   export WANDB_ENTITY="your-team-name"  # Only if using team workspace
   
5. For offline/debugging mode:
   export WANDB_MODE=offline  # Logs locally, can sync later
   
6. To disable W&B temporarily:
   export WANDB_DISABLED=true
""")

def example_integration():
    """Show example integration code."""
    print("\n" + "="*60)
    print("EXAMPLE INTEGRATION CODE")
    print("="*60)
    
    example_code = '''
import wandb
from typing import Dict, Any, Optional

class WandbLogger:
    """Wrapper for W&B logging in experiments."""
    
    def __init__(self, 
                 project: str,
                 entity: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize W&B run."""
        self.run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=name,
            tags=tags,
            reinit=True  # Allow multiple runs in same script
        )
        
    def log_steering_result(self, 
                           alpha: float,
                           direction: str,
                           prompt: str,
                           original_answer: str,
                           steered_response: str,
                           steered_answer: str,
                           success: bool,
                           parsable: bool,
                           example_idx: int):
        """Log a single steering result."""
        wandb.log({
            "steering/alpha": alpha,
            "steering/direction": direction,
            "steering/success": success,
            "steering/parsable": parsable,
            "steering/answer_changed": steered_answer != original_answer,
            "steering/example_idx": example_idx,
        })
        
        # Log text as a table for easy viewing
        table = wandb.Table(
            columns=["Alpha", "Direction", "Original", "Steered", "Success", "Parsable"],
            data=[[alpha, direction, original_answer, steered_answer, success, parsable]]
        )
        wandb.log({"steering/examples": table})
        
        # Log full responses for detailed analysis
        wandb.log({
            "steering/prompt": wandb.Html(f"<pre>{prompt}</pre>"),
            "steering/original_answer": original_answer,
            "steering/steered_response": wandb.Html(f"<pre>{steered_response}</pre>"),
            "steering/steered_answer": steered_answer,
        })
    
    def log_probe_results(self, layer: int, auc: float, method: str):
        """Log probe training results."""
        wandb.log({
            f"probes/{method}/layer_{layer}_auc": auc,
            "probes/layer": layer,
        })
    
    def log_summary(self, results: Dict[str, Any]):
        """Log experiment summary."""
        wandb.summary.update(results)
    
    def finish(self):
        """Finish the W&B run."""
        wandb.finish()

# Example usage in experiment
def run_experiment_with_wandb():
    # Initialize logger
    logger = WandbLogger(
        project="post-hoc-reasoning",
        entity="your-team",  # Optional
        config={
            "model": "gemma-2b",
            "dataset": "sports_understanding",
            "method": "caa-single-layer",
            "alpha_range": [-10, -5, -2, 0, 2, 5, 10],
        },
        name="gemma-2b-sports-caa",
        tags=["steering", "caa", "gemma"]
    )
    
    # During steering experiments
    for alpha in [-10, -5, -2, 0, 2, 5, 10]:
        for i, example in enumerate(test_examples):
            # ... run steering ...
            
            logger.log_steering_result(
                alpha=alpha,
                direction="yes_to_no",
                prompt=example["prompt"],
                original_answer="yes",
                steered_response=steered_response,
                steered_answer=parsed_answer,
                success=(parsed_answer == "no"),
                parsable=(parsed_answer in ["yes", "no"]),
                example_idx=i
            )
    
    # Log summary
    logger.log_summary({
        "total_examples": len(test_examples),
        "best_alpha": best_alpha,
        "best_success_rate": best_rate,
    })
    
    logger.finish()
'''
    print(example_code)

def integration_points():
    """Show where to integrate W&B in the existing code."""
    print("\n" + "="*60)
    print("INTEGRATION POINTS IN EXISTING CODE")
    print("="*60)
    
    print("""
Key places to add W&B logging:

1. In experiment_runner.py - EnhancedExperimentRunner.__init__():
   - Initialize W&B run with experiment config
   - Log model, dataset, and method parameters

2. In train_probes_and_compute_steering_vectors():
   - Log probe AUC scores for each layer
   - Log best layer selection
   - Log similarity scores

3. In generate_steered_examples():
   - Log each steering attempt with:
     * Alpha value and direction
     * Original vs steered answer
     * Success/failure/unparsed status
     * Full prompt and response (in collapsible format)

4. In run_steering_experiments():
   - Log summary statistics for each alpha
   - Create plots of success rate vs alpha
   - Log category breakdowns (success/failure/unparsed)

5. Real-time monitoring features:
   - Use wandb.alert() for anomalies (e.g., all unparsed)
   - Create custom charts for steering success rates
   - Log examples that fail to parse for debugging
""")

def minimal_integration_example():
    """Show minimal code changes needed."""
    print("\n" + "="*60)
    print("MINIMAL INTEGRATION EXAMPLE")
    print("="*60)
    
    print("""
Here's a minimal diff to add W&B to experiment_runner.py:

```python
# Add imports at top
import wandb
from typing import Optional

# Add to EnhancedExperimentRunner.__init__
def __init__(self, run_config: ExperimentRunConfig, use_wandb: bool = True):
    # ... existing code ...
    
    self.use_wandb = use_wandb and not os.environ.get("WANDB_DISABLED")
    self.wandb_run = None
    
    if self.use_wandb:
        try:
            import wandb
            # Initialize W&B
            self.wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "post-hoc-reasoning"),
                entity=os.environ.get("WANDB_ENTITY"),  # Optional team name
                config={
                    "models": [m.name for m in run_config.models],
                    "datasets": [d.name for d in run_config.datasets],
                    "cache_dir": run_config.cache_dir,
                },
                reinit=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

# Add to generate_steered_examples() inner loop
if self.use_wandb:
    # Log each example
    wandb.log({
        "steering/alpha": alpha,
        "steering/success": success,
        "steering/parsable": category != "unparsed",
        "steering/answer_changed": steered_answer != original_answer,
    })
    
    # Log examples table periodically
    if global_idx % 10 == 0:  # Every 10 examples
        table_data = []
        for r in steered_results[-10:]:
            table_data.append([
                alpha,
                r.get("original_answer", ""),
                r.get("new_answer", ""),
                r.get("category", ""),
            ])
        
        table = wandb.Table(
            columns=["Alpha", "Original", "Steered", "Status"],
            data=table_data
        )
        wandb.log({"steering/recent_examples": table})

# Add cleanup in run_all_experiments()
def run_all_experiments(self):
    try:
        # ... existing code ...
    finally:
        if self.wandb_run:
            wandb.finish()
```
""")

def advanced_features():
    """Show advanced W&B features."""
    print("\n" + "="*60)
    print("ADVANCED W&B FEATURES")
    print("="*60)
    
    print("""
Advanced features for better experiment tracking:

1. Custom visualizations:
```python
# Create steering success rate plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(alphas, success_rates, 'o-')
ax.set_xlabel('Alpha')
ax.set_ylabel('Success Rate')
ax.set_title('Steering Effectiveness')
wandb.log({"steering/success_plot": wandb.Image(fig)})
plt.close()
```

2. Artifact tracking for model outputs:
```python
# Save all steering results as artifact
artifact = wandb.Artifact(
    f"steering_results_{model}_{dataset}", 
    type="steering_output"
)
artifact.add_file("steering_results.json")
wandb.log_artifact(artifact)
```

3. Real-time alerts:
```python
# Alert if steering completely fails
if success_rate == 0:
    wandb.alert(
        title="Steering Failure",
        text=f"No successful steering at alpha={alpha}",
        level=wandb.AlertLevel.WARN
    )
```

4. Hyperparameter sweeps:
```python
# Define sweep configuration
sweep_config = {
    'method': 'grid',
    'parameters': {
        'alpha': {'values': [-10, -5, -2, -1, 0, 1, 2, 5, 10]},
        'method': {'values': ['caa-single-layer', 'caa-incremental']},
    }
}
sweep_id = wandb.sweep(sweep_config, project="post-hoc-reasoning")
```

5. Custom metrics:
```python
# Define custom metrics
wandb.define_metric("steering/example_idx")
wandb.define_metric("steering/*", step_metric="steering/example_idx")
```
""")

if __name__ == "__main__":
    setup_instructions()
    example_integration()
    integration_points()
    minimal_integration_example()
    advanced_features()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Run 'pip install wandb' in your virtual environment
2. Run 'wandb login' and enter your API key
3. Ask your colleague for the project and entity names
4. Test with a simple script first
5. Add the minimal integration to experiment_runner.py
6. Run experiments and monitor at https://wandb.ai/

For testing without affecting the shared project:
- Set WANDB_MODE=offline for local logging
- Or create a test project first
""")