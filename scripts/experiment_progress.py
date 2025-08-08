#!/usr/bin/env python3
"""
Comprehensive experiment progress tracker for post-hoc reasoning experiments.

This tool provides a hierarchical view of experiment progress showing:
- Data generation status
- Probe training status with AUC scores
- Steering progress with detailed alpha values for each direction

Usage:
    python scripts/experiment_progress.py [cache_dir] [options]
    
Options:
    --detailed       Show all alpha values and detailed information
    --missing-only   Show only incomplete experiments
    --model MODEL    Filter by specific model
    --dataset DATA   Filter by specific dataset
    --json          Output results as JSON
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# Try to import rich for better visualization
try:
    from rich.console import Console
    from rich.tree import Tree
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' for better visualization: pip install rich")

@dataclass
class SteeringInfo:
    """Information about steering experiments for a specific direction."""
    direction: str
    completed_alphas: List[float] = field(default_factory=list)
    expected_alphas: List[float] = field(default_factory=list)
    missing_alphas: List[float] = field(default_factory=list)
    completion_percentage: float = 0.0

@dataclass
class ExperimentStatus:
    """Status of a single experiment."""
    model: str
    dataset: str
    split_info: str
    experiment_hash: str
    path: str
    
    # Data components
    has_dataset: bool = False
    has_train_test_split: bool = False
    has_train_generations: bool = False
    has_test_generations: bool = False
    has_train_activations: bool = False
    has_test_activations: bool = False
    
    # Analysis components  
    has_probes: bool = False
    has_auc_scores: bool = False
    best_auc_score: Optional[float] = None
    best_auc_layer: Optional[int] = None
    
    # Steering components
    steering_yes: Optional[SteeringInfo] = None
    steering_no: Optional[SteeringInfo] = None
    
    # Config
    has_config: bool = False
    
    # Computed metrics
    data_generation_complete: bool = False
    probe_training_complete: bool = False
    steering_complete: bool = False

def load_json_safe(filepath: str) -> Optional[Any]:
    """Safely load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def determine_expected_alphas(existing_alphas: List[float], direction: str) -> List[float]:
    """Determine the expected range of alpha values based on existing ones."""
    if not existing_alphas:
        return []
    
    # Filter out zero if present to determine the range
    non_zero_alphas = [a for a in existing_alphas if a != 0]
    
    if not non_zero_alphas:
        return [0]
    
    # Determine the range based on the sign of alphas
    if direction == "yes":
        # For yes direction, we expect negative alphas
        min_alpha = min(non_zero_alphas)
        max_alpha = max([a for a in non_zero_alphas if a < 0]) if any(a < 0 for a in non_zero_alphas) else 0
        step = 2  # Common step size
        expected = list(range(int(min_alpha), int(max_alpha) + 1, step))
        if 0 not in expected:
            expected.append(0)
    else:
        # For no direction, we expect positive alphas
        min_alpha = min([a for a in non_zero_alphas if a > 0]) if any(a > 0 for a in non_zero_alphas) else 0
        max_alpha = max(non_zero_alphas)
        step = 2  # Common step size
        expected = list(range(int(min_alpha), int(max_alpha) + 1, step))
        if 0 not in expected:
            expected.insert(0, 0)
    
    return sorted(expected)

def analyze_experiment_directory(exp_path: str) -> ExperimentStatus:
    """Analyze a single experiment directory and return its status."""
    path_parts = Path(exp_path).parts
    
    # Extract model, dataset, split info, and hash
    if len(path_parts) >= 4:
        model = path_parts[-4]
        dataset = path_parts[-3] 
        split_info = path_parts[-2]
        exp_hash = path_parts[-1]
    else:
        model = dataset = split_info = exp_hash = "unknown"
    
    status = ExperimentStatus(
        model=model,
        dataset=dataset,
        split_info=split_info,
        experiment_hash=exp_hash,
        path=exp_path
    )
    
    # Check data directory
    data_dir = os.path.join(exp_path, "data")
    if os.path.exists(data_dir):
        status.has_dataset = os.path.exists(os.path.join(data_dir, "dataset.pkl"))
        status.has_train_test_split = os.path.exists(os.path.join(data_dir, "train_test_split.pkl"))
        status.has_train_generations = os.path.exists(os.path.join(data_dir, "train_generations.pkl"))
        status.has_test_generations = os.path.exists(os.path.join(data_dir, "test_generations.pkl"))
        status.has_train_activations = os.path.exists(os.path.join(data_dir, "train_activations.pkl"))
        status.has_test_activations = os.path.exists(os.path.join(data_dir, "test_activations.pkl"))
    
    status.data_generation_complete = status.has_train_generations and status.has_test_generations
    
    # Check probes directory
    probes_dir = os.path.join(exp_path, "probes")
    if os.path.exists(probes_dir):
        # Check for default paths
        auc_path = os.path.join(probes_dir, "auc_scores.json")
        coef_path = os.path.join(probes_dir, "coefficients.pkl")
        
        # Also check for method-specific files (e.g., auc_scores_caa-single-layer.json)
        if not os.path.exists(auc_path):
            # Look for any auc_scores_*.json file
            for filename in os.listdir(probes_dir):
                if filename.startswith("auc_scores_") and filename.endswith(".json"):
                    auc_path = os.path.join(probes_dir, filename)
                    break
        
        if not os.path.exists(coef_path):
            # Look for any coefficients_*.pkl file
            for filename in os.listdir(probes_dir):
                if filename.startswith("coefficients_") and filename.endswith(".pkl"):
                    coef_path = os.path.join(probes_dir, filename)
                    break
        
        status.has_auc_scores = os.path.exists(auc_path)
        status.has_probe_coefficients = os.path.exists(coef_path)
        
        if status.has_auc_scores:
            auc_data = load_json_safe(auc_path)
            if auc_data:
                if isinstance(auc_data, list):
                    # Old format: list of scores
                    status.best_auc_score = max(auc_data)
                    status.best_auc_layer = auc_data.index(status.best_auc_score)
                elif isinstance(auc_data, dict):
                    # New format: dict with layer keys
                    # Convert string keys to int if needed
                    layer_scores = {int(k) if isinstance(k, str) else k: v for k, v in auc_data.items()}
                    if layer_scores:
                        # Find best score and layer (earliest layer wins ties)
                        best_score = max(layer_scores.values())
                        best_layers = [layer for layer, score in layer_scores.items() if score == best_score]
                        status.best_auc_layer = min(best_layers)  # Earliest layer wins
                        status.best_auc_score = best_score
    
    status.probe_training_complete = status.has_auc_scores and status.has_probe_coefficients
    
    # Check steering directory
    steering_dir = os.path.join(exp_path, "steering")
    yes_alphas = []
    no_alphas = []
    
    if os.path.exists(steering_dir):
        for filename in os.listdir(steering_dir):
            if filename.startswith("steering_alpha_") and filename.endswith(".pkl"):
                parts = filename.replace("steering_alpha_", "").replace(".pkl", "").split("_")
                if len(parts) >= 2:
                    try:
                        alpha = float(parts[0])
                        direction = parts[1]
                        if direction == "yes":
                            yes_alphas.append(alpha)
                        elif direction == "no":
                            no_alphas.append(alpha)
                    except ValueError:
                        continue
    
    # Process steering info for each direction
    if yes_alphas:
        yes_alphas = sorted(set(yes_alphas))
        expected_yes = determine_expected_alphas(yes_alphas, "yes")
        missing_yes = [a for a in expected_yes if a not in yes_alphas]
        
        status.steering_yes = SteeringInfo(
            direction="yes",
            completed_alphas=yes_alphas,
            expected_alphas=expected_yes,
            missing_alphas=missing_yes,
            completion_percentage=len(yes_alphas) / len(expected_yes) * 100 if expected_yes else 0
        )
    
    if no_alphas:
        no_alphas = sorted(set(no_alphas))
        expected_no = determine_expected_alphas(no_alphas, "no")
        missing_no = [a for a in expected_no if a not in no_alphas]
        
        status.steering_no = SteeringInfo(
            direction="no",
            completed_alphas=no_alphas,
            expected_alphas=expected_no,
            missing_alphas=missing_no,
            completion_percentage=len(no_alphas) / len(expected_no) * 100 if expected_no else 0
        )
    
    # Check if steering is complete
    if status.steering_yes and status.steering_no:
        status.steering_complete = (
            status.steering_yes.completion_percentage == 100 and
            status.steering_no.completion_percentage == 100
        )
    
    # Check config
    config_path = os.path.join(exp_path, "metadata", "config.json")
    status.has_config = os.path.exists(config_path)
    
    return status

def find_all_experiments(cache_dir: str) -> List[ExperimentStatus]:
    """Find and analyze all experiments in the cache directory."""
    experiments = []
    experiments_dir = os.path.join(cache_dir, "experiments")
    
    if not os.path.exists(experiments_dir):
        return experiments
    
    # Walk through the directory structure
    for model_dir in os.listdir(experiments_dir):
        model_path = os.path.join(experiments_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        for dataset_dir in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
                
            for split_dir in os.listdir(dataset_path):
                split_path = os.path.join(dataset_path, split_dir)
                if not os.path.isdir(split_path):
                    continue
                    
                for exp_dir in os.listdir(split_path):
                    exp_path = os.path.join(split_path, exp_dir)
                    if os.path.isdir(exp_path):
                        status = analyze_experiment_directory(exp_path)
                        experiments.append(status)
    
    return experiments

def create_progress_bar(percentage: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int(width * percentage / 100)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {percentage:.0f}%"

def format_alpha_list(alphas: List[float], missing: List[float] = None) -> str:
    """Format a list of alpha values with missing ones highlighted."""
    if not alphas:
        return "None"
    
    formatted = [f"{int(a)}" if a == int(a) else f"{a}" for a in sorted(alphas)]
    result = f"[{', '.join(formatted)}]"
    
    if missing:
        missing_formatted = [f"{int(a)}" if a == int(a) else f"{a}" for a in sorted(missing)]
        result += f" ❌ Missing: [{', '.join(missing_formatted)}]"
    
    return result

def print_summary_rich(experiments: List[ExperimentStatus], args):
    """Print a rich summary using the rich library."""
    console = Console()
    
    # Title
    console.print(Panel.fit("🧪 POST-HOC REASONING EXPERIMENT PROGRESS", style="bold blue"))
    
    # Group by model
    model_groups = defaultdict(list)
    for exp in experiments:
        if not args.missing_only or not (exp.data_generation_complete and exp.probe_training_complete and exp.steering_complete):
            model_groups[exp.model].append(exp)
    
    for model, model_experiments in sorted(model_groups.items()):
        if args.model and model != args.model:
            continue
        
        # Create model tree
        model_tree = Tree(f"📊 [bold cyan]{model}[/bold cyan]")
        
        # Group by dataset
        dataset_groups = defaultdict(list)
        for exp in model_experiments:
            dataset_groups[exp.dataset].append(exp)
        
        for dataset, dataset_experiments in sorted(dataset_groups.items()):
            if args.dataset and dataset != args.dataset:
                continue
            
            # For now, just use the first experiment (assuming one per dataset)
            exp = dataset_experiments[0]
            
            # Create dataset branch
            dataset_branch = model_tree.add(f"[bold green]{dataset}[/bold green]")
            
            # Data generation status
            data_icon = "✅" if exp.data_generation_complete else "❌"
            data_text = f"{data_icon} Data Generation: {'complete' if exp.data_generation_complete else 'incomplete'}"
            if exp.has_train_generations and not exp.has_test_generations:
                data_text += " (only train)"
            elif exp.has_test_generations and not exp.has_train_generations:
                data_text += " (only test)"
            dataset_branch.add(data_text)
            
            # Probe training status
            probe_icon = "✅" if exp.probe_training_complete else "❌"
            probe_text = f"{probe_icon} Probe Training: {'complete' if exp.probe_training_complete else 'not trained'}"
            if exp.best_auc_score:
                probe_text += f" (Best AUC: {exp.best_auc_score:.4f}"
                if exp.best_auc_layer is not None:
                    probe_text += f" @ Layer {exp.best_auc_layer}"
                probe_text += ")"
            dataset_branch.add(probe_text)
            
            # Steering progress
            steering_branch = dataset_branch.add("🎮 Steering Progress:")
            
            if exp.steering_yes or exp.steering_no:
                # YES direction
                if exp.steering_yes:
                    yes_progress = create_progress_bar(exp.steering_yes.completion_percentage)
                    yes_text = f"YES → NO: {yes_progress} ({len(exp.steering_yes.completed_alphas)}/{len(exp.steering_yes.expected_alphas)} alphas)"
                    yes_branch = steering_branch.add(yes_text)
                    
                    if args.detailed or exp.steering_yes.missing_alphas:
                        alpha_text = format_alpha_list(exp.steering_yes.completed_alphas, 
                                                      exp.steering_yes.missing_alphas if exp.steering_yes.missing_alphas else None)
                        yes_branch.add(alpha_text)
                
                # NO direction
                if exp.steering_no:
                    no_progress = create_progress_bar(exp.steering_no.completion_percentage)
                    no_text = f"NO → YES: {no_progress} ({len(exp.steering_no.completed_alphas)}/{len(exp.steering_no.expected_alphas)} alphas)"
                    no_branch = steering_branch.add(no_text)
                    
                    if args.detailed or exp.steering_no.missing_alphas:
                        alpha_text = format_alpha_list(exp.steering_no.completed_alphas,
                                                     exp.steering_no.missing_alphas if exp.steering_no.missing_alphas else None)
                        no_branch.add(alpha_text)
            else:
                steering_branch.add("Not started")
        
        console.print(model_tree)
        console.print()

def print_summary_plain(experiments: List[ExperimentStatus], args):
    """Print a plain text summary for environments without rich."""
    print("=" * 80)
    print("🧪 POST-HOC REASONING EXPERIMENT PROGRESS")
    print("=" * 80)
    
    # Group by model
    model_groups = defaultdict(list)
    for exp in experiments:
        if not args.missing_only or not (exp.data_generation_complete and exp.probe_training_complete and exp.steering_complete):
            model_groups[exp.model].append(exp)
    
    for model, model_experiments in sorted(model_groups.items()):
        if args.model and model != args.model:
            continue
        
        print(f"\n📊 {model}")
        
        # Group by dataset
        dataset_groups = defaultdict(list)
        for exp in model_experiments:
            dataset_groups[exp.dataset].append(exp)
        
        for dataset, dataset_experiments in sorted(dataset_groups.items()):
            if args.dataset and dataset != args.dataset:
                continue
            
            # For now, just use the first experiment
            exp = dataset_experiments[0]
            
            print(f"├── {dataset}")
            
            # Data generation status
            data_icon = "✅" if exp.data_generation_complete else "❌"
            print(f"│   ├── 📝 Data Generation: {data_icon}")
            
            # Probe training status
            probe_icon = "✅" if exp.probe_training_complete else "❌"
            probe_text = f"│   ├── 🎯 Probe Training: {probe_icon}"
            if exp.best_auc_score:
                probe_text += f" (Best AUC: {exp.best_auc_score:.4f}"
                if exp.best_auc_layer is not None:
                    probe_text += f" @ Layer {exp.best_auc_layer}"
                probe_text += ")"
            print(probe_text)
            
            # Steering progress
            print(f"│   └── 🎮 Steering Progress:")
            
            if exp.steering_yes or exp.steering_no:
                # YES direction
                if exp.steering_yes:
                    yes_progress = create_progress_bar(exp.steering_yes.completion_percentage, width=10)
                    print(f"│       ├── YES → NO: {yes_progress} ({len(exp.steering_yes.completed_alphas)}/{len(exp.steering_yes.expected_alphas)} alphas)")
                    
                    if args.detailed or exp.steering_yes.missing_alphas:
                        alpha_text = format_alpha_list(exp.steering_yes.completed_alphas,
                                                     exp.steering_yes.missing_alphas if exp.steering_yes.missing_alphas else None)
                        print(f"│       │   └── {alpha_text}")
                
                # NO direction
                if exp.steering_no:
                    no_progress = create_progress_bar(exp.steering_no.completion_percentage, width=10)
                    print(f"│       └── NO → YES: {no_progress} ({len(exp.steering_no.completed_alphas)}/{len(exp.steering_no.expected_alphas)} alphas)")
                    
                    if args.detailed or exp.steering_no.missing_alphas:
                        alpha_text = format_alpha_list(exp.steering_no.completed_alphas,
                                                     exp.steering_no.missing_alphas if exp.steering_no.missing_alphas else None)
                        print(f"│           └── {alpha_text}")
            else:
                print(f"│       └── Not started")

def print_summary(experiments: List[ExperimentStatus], args):
    """Print summary using appropriate method."""
    if args.json:
        # Convert to JSON-serializable format
        output = []
        for exp in experiments:
            exp_dict = asdict(exp)
            # Convert None values and handle special cases
            if exp_dict['steering_yes'] is None:
                exp_dict['steering_yes'] = {}
            if exp_dict['steering_no'] is None:
                exp_dict['steering_no'] = {}
            output.append(exp_dict)
        
        print(json.dumps(output, indent=2, default=str))
        return
    
    if HAS_RICH and not args.no_color:
        print_summary_rich(experiments, args)
    else:
        print_summary_plain(experiments, args)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Track experiment progress from cache")
    parser.add_argument("cache_dir", nargs="?", default="cache",
                      help="Cache directory to analyze (default: cache)")
    parser.add_argument("--detailed", action="store_true",
                      help="Show all alpha values and detailed information")
    parser.add_argument("--missing-only", action="store_true",
                      help="Show only incomplete experiments")
    parser.add_argument("--json", action="store_true",
                      help="Output as JSON")
    parser.add_argument("--model", type=str,
                      help="Filter by specific model")
    parser.add_argument("--dataset", type=str,
                      help="Filter by specific dataset")
    parser.add_argument("--no-color", action="store_true",
                      help="Disable colored output")
    parser.add_argument("--output", "-o", type=str,
                      help="Write output to a text file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cache_dir):
        print(f"❌ Cache directory not found: {args.cache_dir}")
        sys.exit(1)
    
    experiments = find_all_experiments(args.cache_dir)
    
    if not experiments:
        print(f"❌ No experiments found in {args.cache_dir}")
        sys.exit(1)
    
    # Handle output redirection
    if args.output:
        # Redirect stdout to file
        import contextlib
        
        # Force plain text output when writing to file (no color codes)
        original_no_color = args.no_color
        args.no_color = True
        
        with open(args.output, 'w') as f:
            with contextlib.redirect_stdout(f):
                print_summary(experiments, args)
        
        # Restore original setting and inform user
        args.no_color = original_no_color
        print(f"✅ Output written to: {args.output}")
    else:
        print_summary(experiments, args)

if __name__ == "__main__":
    main()