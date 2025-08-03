#!/usr/bin/env python3
"""
Notebook-style experiment runner for post-hoc reasoning experiments.

This file replicates the functionality of:
python3 run_experiments.py --config configs/mini_gemma_experiment.yaml --no-cache --no-interactive

Split into cells with #%# breaks for step-by-step execution and debugging.
"""

#%% Setup and Imports
import os
import sys
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

print("📦 Importing modules...")
from config import ConfigLoader, ExperimentRunConfig
from experiment_runner import EnhancedExperimentRunner
from nnsight_experiment_runner import NNsightExperimentRunner

print("✓ All modules imported successfully")

#%% Configuration Loading
print("📖 Loading configuration from configs/mini_gemma_experiment.yaml")

config_path = "configs/mini_gemma_experiment.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

# Load the experiment configuration
config = ConfigLoader.load_experiment_config(config_path)

print("✓ Configuration loaded successfully")
print(f"  Models: {[m.name for m in config.models]}")
print(f"  Backends: {[m.backend for m in config.models]}")
print(f"  Datasets: {[d.name for d in config.datasets]}")
print(f"  Alpha range: {config.steering.alpha_range}")

#%% Apply Command-line Equivalent Overrides
print("\n🔧 Applying command-line equivalent overrides (--no-cache --no-interactive)")

# Apply the equivalent of --no-cache --no-interactive
config.cache_dir = "cache"  # Default cache dir
config.use_cache = False    # --no-cache
config.interactive = False  # --no-interactive

print(f"  Cache directory: {config.cache_dir}")
print(f"  Use cache: {config.use_cache}")
print(f"  Interactive mode: {config.interactive}")

#%% Backend Analysis and Runner Selection
print("\n🔍 Analyzing backend configuration...")

# Check if any model requires nnsight
has_nnsight = any(
    model.backend == "nnsight" or 
    (model.backend == "auto" and "deepseek" in model.name.lower())
    for model in config.models
)

# Check if any model explicitly requires transformer_lens
has_transformer_lens = any(
    model.backend == "transformer_lens"
    for model in config.models
)

# Select runner type
if has_nnsight and has_transformer_lens:
    print("⚠️  Warning: Mixed backends detected. Using NNsight runner for compatibility.")
    runner_type = "nnsight"
elif has_nnsight:
    runner_type = "nnsight"
else:
    runner_type = "transformer_lens"

print(f"🔧 Selected runner type: {runner_type}")

#%% Initialize Experiment Runner
print(f"\n🚀 Initializing {runner_type} experiment runner...")

if runner_type == "nnsight":
    runner = NNsightExperimentRunner(config)
else:
    runner = EnhancedExperimentRunner(config)

print("✓ Experiment runner initialized")

#%% Display Experiment Plan
print("\n📋 Experiment Plan:")
print("=" * 50)

for model in config.models:
    print(f"\n🤖 Model: {model.name}")
    print(f"   Backend: {model.backend}")
    print(f"   Batch size: {model.batch_size}")
    print(f"   Temperature: {model.temperature}")
    print(f"   Max tokens: {model.max_new_tokens}")
    
    for dataset in config.datasets:
        print(f"\n  📊 Dataset: {dataset.name}")
        print(f"     Train size: {dataset.train_size}")
        print(f"     Test size: {dataset.test_size}")
        print(f"     Split seed: {dataset.split_seed}")
        
        print(f"     🎯 Steering alphas: {config.steering.alpha_range}")

print("\n" + "=" * 50)

#%% Run All Experiments
print("\n🚀 Starting all experiments...")
print("This will run through the complete pipeline:")
print("1. Load and prepare datasets")
print("2. Generate model activations")
print("3. Train probes")
print("4. Run steering experiments")
print("5. Evaluate results")

try:
    runner.run_all_experiments()
    print("\n🎉 All experiments completed successfully!")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Experiments interrupted by user")
    print("You can resume by running the cells again or using --resume with the CLI")
    
except Exception as e:
    print(f"\n❌ Error running experiments: {str(e)}")
    import traceback
    traceback.print_exc()

#%% Get Results Summary
print("\n📊 Generating results summary...")

try:
    summary_df = runner.get_results_summary()
    
    if not summary_df.empty:
        print("\n📈 Experiment Results Summary:")
        print("=" * 60)
        
        # Display summary in a readable format
        for idx, row in summary_df.iterrows():
            print(f"\n🔬 Experiment {idx + 1}:")
            for col, val in row.items():
                if col not in ['experiment_id']:  # Skip long IDs
                    print(f"   {col}: {val}")
        
        print("\n💾 Full summary available in summary_df variable")
        print("    You can save it with: summary_df.to_csv('results_summary.csv', index=False)")
    else:
        print("⚠️  No results found in summary")
        
except Exception as e:
    print(f"❌ Error generating summary: {str(e)}")

#%% Optional: Save Results
print("\n💾 Optional: Save results to CSV")
print("Uncomment the lines below to save results:")
print("# summary_df.to_csv('mini_gemma_experiment_results.csv', index=False)")
print("# print('✓ Results saved to mini_gemma_experiment_results.csv')")

# Uncomment to save:
# summary_df.to_csv('mini_gemma_experiment_results.csv', index=False)
# print('✓ Results saved to mini_gemma_experiment_results.csv')

print("\n✅ Notebook execution complete!")
print("You can now examine the results or run individual cells for debugging.")