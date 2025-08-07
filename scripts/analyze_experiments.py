#!/usr/bin/env python3
"""
Comprehensive analysis of experiment results from cache.

This script analyzes experiments run with run_transformer_lens_experiments.py 
and run_nnsight_experiments.py to extract:
- Train/Test accuracy
- Probe AUCs  
- Steering success rates at different alpha values

Usage:
    python analyze_experiments.py [cache_dir] [--format markdown|csv]
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import argparse

def load_json_safe(filepath: str) -> Optional[Any]:
    """Safely load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def load_pickle_safe(filepath: str) -> Optional[Any]:
    """Safely load pickle file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def calculate_accuracy(results: List[Dict]) -> float:
    """Calculate accuracy from generation results."""
    if not results:
        return 0.0
    
    correct = sum(1 for r in results if r.get('pred_answer') == r.get('correct_answer'))
    return correct / len(results) * 100

def analyze_steering_results(steering_dir: str) -> Dict[float, Dict[str, Dict]]:
    """Analyze all steering results in a directory."""
    steering_data = defaultdict(lambda: {'yes': None, 'no': None})
    
    if not os.path.exists(steering_dir):
        return {}
    
    for filename in os.listdir(steering_dir):
        if filename.startswith('steering_alpha_') and filename.endswith('.pkl'):
            # Parse filename like "steering_alpha_-2.0_yes.pkl" or "steering_alpha_-2_yes.pkl"
            parts = filename.replace('steering_alpha_', '').replace('.pkl', '').rsplit('_', 1)
            if len(parts) == 2:
                try:
                    alpha = float(parts[0])
                    direction = parts[1]
                    
                    results = load_pickle_safe(os.path.join(steering_dir, filename))
                    if results:
                        # Calculate success rate
                        successes = sum(1 for r in results if r.get('success', False))
                        total = len(results)
                        success_rate = successes / total if total > 0 else 0
                        
                        steering_data[alpha][direction] = {
                            'success_rate': success_rate,
                            'successes': successes,
                            'total': total
                        }
                except ValueError:
                    continue
    
    return dict(steering_data)

def analyze_experiment(exp_path: str) -> Dict[str, Any]:
    """Analyze a single experiment directory."""
    result = {
        'path': exp_path,
        'model': 'unknown',
        'dataset': 'unknown',
        'train_accuracy': None,
        'test_accuracy': None,
        'probe_auc': None,
        'best_layer': None,
        'steering_results': {},
        'status': []
    }
    
    # Extract model and dataset from path
    path_parts = Path(exp_path).parts
    for i, part in enumerate(path_parts):
        if part == 'experiments' and i + 2 < len(path_parts):
            result['model'] = path_parts[i + 1]
            result['dataset'] = path_parts[i + 2]
            break
    
    # Load and analyze generations
    train_gen_path = os.path.join(exp_path, 'data', 'train_generations.pkl')
    test_gen_path = os.path.join(exp_path, 'data', 'test_generations.pkl')
    
    train_results = load_pickle_safe(train_gen_path)
    test_results = load_pickle_safe(test_gen_path)
    
    if train_results:
        result['train_accuracy'] = calculate_accuracy(train_results)
        result['status'].append('train_gen')
    
    if test_results:
        result['test_accuracy'] = calculate_accuracy(test_results)
        result['status'].append('test_gen')
    
    # Load probe results - check both old and new paths
    auc_paths = [
        os.path.join(exp_path, 'probes', 'caa-single-layer', 'auc_scores.json'),
        os.path.join(exp_path, 'probes', 'auc_scores.json')
    ]
    
    for auc_path in auc_paths:
        auc_scores = load_json_safe(auc_path)
        if auc_scores:
            if isinstance(auc_scores, dict):
                # Handle dict format {layer: score}
                scores = list(auc_scores.values())
                layers = list(auc_scores.keys())
                if scores:
                    max_idx = np.argmax(scores)
                    result['probe_auc'] = float(scores[max_idx])
                    result['best_layer'] = int(layers[max_idx])
            else:
                # Handle list format [scores]
                result['probe_auc'] = float(max(auc_scores))
                result['best_layer'] = int(np.argmax(auc_scores))
            result['status'].append('probes')
            break
    
    # Analyze steering results
    steering_dir = os.path.join(exp_path, 'steering')
    result['steering_results'] = analyze_steering_results(steering_dir)
    if result['steering_results']:
        result['status'].append('steering')
    
    return result

def print_markdown_table(experiment_results: List[Dict[str, Any]]):
    """Print results as markdown tables."""
    # Group by model and dataset
    grouped = defaultdict(list)
    for exp in experiment_results:
        key = (exp['model'], exp['dataset'])
        grouped[key].append(exp)
    
    for (model, dataset), experiments in grouped.items():
        # Use the most complete experiment
        exp = max(experiments, key=lambda x: len(x['status']))
        
        print(f"\n## Model: {model}")
        print(f"### Dataset: {dataset}")
        print()
        
        # Basic metrics
        if exp['train_accuracy'] is not None:
            print(f"**Train Accuracy:** {exp['train_accuracy']:.1f}%")
        if exp['test_accuracy'] is not None:
            print(f"**Test Accuracy:** {exp['test_accuracy']:.1f}%")
        if exp['probe_auc'] is not None:
            print(f"**Best Probe Layer:** {exp['best_layer']} (AUC: {exp['probe_auc']:.4f})")
        print()
        
        # Steering results table
        if exp['steering_results']:
            print("#### Steering Success Rates")
            print()
            print("| α | Yes→No | No→Yes | Overall |")
            print("|---|--------|--------|---------|")
            
            alphas = sorted(exp['steering_results'].keys())
            for alpha in alphas:
                data = exp['steering_results'][alpha]
                yes_data = data.get('yes', {})
                no_data = data.get('no', {})
                
                # For negative alpha, we steer yes→no
                # For positive alpha, we steer no→yes
                if alpha < 0:
                    yes_to_no = yes_data.get('success_rate', 0) if yes_data else 0
                    no_to_yes = no_data.get('success_rate', 0) if no_data else 0
                else:
                    yes_to_no = yes_data.get('success_rate', 0) if yes_data else 0
                    no_to_yes = no_data.get('success_rate', 0) if no_data else 0
                
                # Calculate overall
                total_trials = 0
                total_successes = 0
                if yes_data:
                    total_trials += yes_data.get('total', 0)
                    total_successes += yes_data.get('successes', 0)
                if no_data:
                    total_trials += no_data.get('total', 0)
                    total_successes += no_data.get('successes', 0)
                
                overall = total_successes / total_trials if total_trials > 0 else 0
                
                print(f"| {alpha:+.1f} | {yes_to_no:.1%} | {no_to_yes:.1%} | {overall:.1%} |")

def print_csv_summary(experiment_results: List[Dict[str, Any]]):
    """Print results as CSV format."""
    # Print header
    print("model,dataset,train_accuracy,test_accuracy,probe_auc,best_layer,alpha,yes_to_no_rate,no_to_yes_rate,overall_rate")
    
    for exp in experiment_results:
        base_values = [
            exp['model'],
            exp['dataset'],
            f"{exp['train_accuracy']:.2f}" if exp['train_accuracy'] is not None else "",
            f"{exp['test_accuracy']:.2f}" if exp['test_accuracy'] is not None else "",
            f"{exp['probe_auc']:.4f}" if exp['probe_auc'] is not None else "",
            str(exp['best_layer']) if exp['best_layer'] is not None else ""
        ]
        
        if exp['steering_results']:
            for alpha, data in exp['steering_results'].items():
                yes_data = data.get('yes', {})
                no_data = data.get('no', {})
                
                yes_rate = yes_data.get('success_rate', None) if yes_data else None
                no_rate = no_data.get('success_rate', None) if no_data else None
                
                # Calculate overall
                total_trials = 0
                total_successes = 0
                if yes_data:
                    total_trials += yes_data.get('total', 0)
                    total_successes += yes_data.get('successes', 0)
                if no_data:
                    total_trials += no_data.get('total', 0)
                    total_successes += no_data.get('successes', 0)
                
                overall_rate = total_successes / total_trials if total_trials > 0 else None
                
                row = base_values + [
                    f"{alpha:.1f}",
                    f"{yes_rate:.4f}" if yes_rate is not None else "",
                    f"{no_rate:.4f}" if no_rate is not None else "",
                    f"{overall_rate:.4f}" if overall_rate is not None else ""
                ]
                print(",".join(row))
        else:
            # Print row without steering data
            row = base_values + ["", "", "", ""]
            print(",".join(row))

def find_experiment_dirs(cache_dir: str) -> List[str]:
    """Find all experiment directories in cache."""
    experiment_dirs = []
    
    base_dir = os.path.join(cache_dir, 'experiments') if not cache_dir.endswith('experiments') else cache_dir
    
    if not os.path.exists(base_dir):
        return []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        # An experiment directory should have at least a data or metadata folder
        if 'data' in dirs or 'metadata' in dirs:
            experiment_dirs.append(root)
    
    return experiment_dirs

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results from cache')
    parser.add_argument('cache_dir', nargs='?', default='cache', help='Cache directory path')
    parser.add_argument('--format', choices=['markdown', 'csv'], default='markdown', 
                        help='Output format (default: markdown)')
    args = parser.parse_args()
    
    print(f"🔍 Analyzing experiments in: {args.cache_dir}")
    
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(args.cache_dir)
    
    if not exp_dirs:
        print("❌ No experiment directories found")
        return
    
    print(f"📊 Found {len(exp_dirs)} experiment directories")
    
    # Analyze each experiment
    all_results = []
    for exp_dir in exp_dirs:
        result = analyze_experiment(exp_dir)
        if result['status']:  # Only include experiments with some data
            all_results.append(result)
    
    # Sort by model and dataset
    all_results.sort(key=lambda x: (x['model'], x['dataset']))
    
    # Print results
    if args.format == 'markdown':
        print_markdown_table(all_results)
    else:
        print_csv_summary(all_results)
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Total experiments analyzed: {len(all_results)}")
    
    complete_experiments = [e for e in all_results if 'train_gen' in e['status'] and 'probes' in e['status']]
    print(f"   Experiments with generations and probes: {len(complete_experiments)}")
    
    steering_experiments = [e for e in all_results if 'steering' in e['status']]
    print(f"   Experiments with steering results: {len(steering_experiments)}")

if __name__ == "__main__":
    main()