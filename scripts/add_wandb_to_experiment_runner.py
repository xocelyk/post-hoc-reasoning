#!/usr/bin/env python3
"""
Example of how to add W&B integration to experiment_runner.py

This shows the key modifications needed to add real-time logging.
"""

def show_modifications():
    """Show the key modifications needed."""
    
    print("="*60)
    print("MODIFICATIONS FOR experiment_runner.py")
    print("="*60)
    
    print("""
1. Add import at the top:
```python
from wandb_integration import WandbExperimentLogger
```

2. Modify __init__ to initialize W&B:
```python
def __init__(self, run_config: ExperimentRunConfig):
    self.run_config = run_config
    self.exp_manager = ExperimentManager(run_config.cache_dir)
    self.visualizer = create_visualizer(run_config.interactive)
    
    # Initialize W&B logger
    self.wandb_logger = None
    if getattr(run_config, 'use_wandb', True):
        self.wandb_logger = WandbExperimentLogger(
            experiment_config={
                "models": [m.name for m in run_config.models],
                "datasets": [d.name for d in run_config.datasets],
                "steering_alpha_range": run_config.steering.alpha_range,
                "cache_dir": run_config.cache_dir,
            }
        )
    
    # ... rest of init
```

3. In train_probes_and_compute_steering_vectors(), after computing similarity scores:
```python
# Log probe results to W&B
if self.wandb_logger:
    for i, (layer, score) in enumerate(zip(layers, similarity_scores)):
        self.wandb_logger.log_probe_training(
            layer=layer,
            train_auc=score,  # Using similarity score as proxy
            test_auc=score,
            similarity_score=score,
            method=steering_method_name
        )
    
    # Log best layer selection
    best_idx = np.argmax(similarity_scores)
    self.wandb_logger.log_best_layer_selection(
        best_layer=layers[best_idx],
        best_score=similarity_scores[best_idx],
        method=steering_method_name,
        all_scores=similarity_scores
    )
```

4. In generate_steered_examples(), inside the loop after parsing:
```python
# Log to W&B
if self.wandb_logger:
    self.wandb_logger.log_steering_example(
        alpha=alpha,
        direction="yes_to_no" if alpha < 0 else "no_to_yes",
        prompt=example_prompt,
        original_answer=example["pred_answer"],
        steered_response=generation[-500:],  # Last 500 chars
        steered_answer=steered_answer,
        target_answer=target_answer,
        category=category,
        example_idx=global_idx,
        model_name=config.model_name,
        dataset_name=config.dataset_name
    )
```

5. In run_steering_experiments(), after processing each alpha:
```python
# Log summary for this alpha/direction
if self.wandb_logger and results_yes:
    self.wandb_logger.log_steering_summary(
        alpha=alpha_yes,
        direction="yes",
        total_examples=len(results_yes),
        success_count=success_count,
        failure_count=failure_count,
        unparsed_count=unparsed_count
    )
    
    # Log results table
    self.wandb_logger.log_steering_results_table(
        results=results_yes[:20],  # First 20 examples
        alpha=alpha_yes,
        direction="yes"
    )
```

6. At the end of run_steering_experiments():
```python
# Create success rate plot
if self.wandb_logger:
    alpha_values = []
    yes_to_no_rates = []
    no_to_yes_rates = []
    
    for alpha in self.run_config.steering.alpha_range:
        alpha_values.append(alpha)
        
        # Calculate success rates from cached results
        yes_results = cache.load_pickle(cache.get_steering_results_path(-abs(alpha), "yes"))
        no_results = cache.load_pickle(cache.get_steering_results_path(abs(alpha), "no"))
        
        yes_rate = sum(1 for r in yes_results if r["category"] == "success") / len(yes_results) if yes_results else 0
        no_rate = sum(1 for r in no_results if r["category"] == "success") / len(no_results) if no_results else 0
        
        yes_to_no_rates.append(yes_rate)
        no_to_yes_rates.append(no_rate)
    
    self.wandb_logger.create_steering_success_plot(
        alpha_values, yes_to_no_rates, no_to_yes_rates
    )
```

7. In run_single_experiment(), at the end:
```python
# Log experiment summary
if self.wandb_logger:
    summary = {
        "model": config.model_name,
        "dataset": config.dataset_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "best_probe_layer": best_layer,
        "best_probe_score": best_score,
        "steering_method": config.steering_method,
    }
    self.wandb_logger.log_experiment_summary(summary)
```

8. In run_all_experiments(), in the finally block:
```python
finally:
    # ... existing cleanup ...
    
    # Finish W&B run
    if self.wandb_logger:
        self.wandb_logger.finish()
```
""")

def show_environment_setup():
    """Show environment setup instructions."""
    print("\n" + "="*60)
    print("ENVIRONMENT SETUP")
    print("="*60)
    
    print("""
1. Install wandb:
   pip install wandb

2. Login (one-time):
   wandb login
   # Enter your API key from https://wandb.ai/authorize

3. Set environment variables:
   export WANDB_PROJECT="your-project-name"  # Ask your colleague
   export WANDB_ENTITY="your-team-name"      # If using team workspace

4. Optional settings:
   # For testing/debugging:
   export WANDB_MODE=offline
   
   # To disable W&B:
   export WANDB_DISABLED=true
   
   # To use a specific W&B server:
   export WANDB_BASE_URL="https://api.wandb.ai"
""")

def show_usage_example():
    """Show usage example."""
    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)
    
    print("""
Once integrated, just run experiments normally:

python run_transformer_lens_experiments.py --config configs/test.yaml

W&B will automatically:
1. Create a new run with a descriptive name
2. Log all probe training results in real-time
3. Log steering examples as they're generated
4. Create visualizations of success rates
5. Save summary statistics

Monitor progress at:
https://wandb.ai/YOUR-ENTITY/YOUR-PROJECT

Real-time features you'll see:
- Live updating metrics
- Example tables showing prompt → response transformations
- Success rate plots
- Alerts if steering completely fails
- Full experiment configuration tracking
""")

def show_testing_code():
    """Show code for testing W&B integration."""
    print("\n" + "="*60)
    print("TEST SCRIPT")
    print("="*60)
    
    test_code = '''
#!/usr/bin/env python3
"""Test W&B integration before running full experiments."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from wandb_integration import WandbExperimentLogger, setup_wandb

# Test setup
if not setup_wandb():
    print("Please complete W&B setup first")
    sys.exit(1)

# Test logging
logger = WandbExperimentLogger(
    experiment_config={
        "test": True,
        "model": "test-model",
        "dataset": "test-dataset"
    }
)

# Log some test data
for layer in range(5):
    logger.log_probe_training(
        layer=layer,
        train_auc=0.5 + layer * 0.1,
        test_auc=0.5 + layer * 0.08,
        method="test"
    )

logger.log_best_layer_selection(
    best_layer=3,
    best_score=0.82,
    method="test"
)

# Log test steering examples
for i in range(10):
    logger.log_steering_example(
        alpha=2.0,
        direction="yes_to_no",
        prompt="Test prompt",
        original_answer="yes",
        steered_response="Test response... no",
        steered_answer="no",
        target_answer="no",
        category="success" if i < 7 else "failure",
        example_idx=i
    )

logger.log_steering_summary(
    alpha=2.0,
    direction="yes_to_no",
    total_examples=10,
    success_count=7,
    failure_count=3,
    unparsed_count=0
)

logger.finish()
print("Test complete! Check your W&B dashboard.")
'''
    
    print(test_code)

if __name__ == "__main__":
    show_modifications()
    show_environment_setup()
    show_usage_example()
    show_testing_code()
    
    print("\n" + "="*60)
    print("QUICK START")
    print("="*60)
    print("""
1. First test W&B is working:
   python setup_wandb.py
   
2. Save the test script and run it:
   python test_wandb_integration.py
   
3. Add wandb_integration.py to your src/ directory
   
4. Modify experiment_runner.py with the changes shown above
   
5. Run experiments with W&B enabled:
   python run_transformer_lens_experiments.py --config configs/test.yaml
   
6. Monitor at https://wandb.ai/YOUR-ENTITY/YOUR-PROJECT
""")