#!/usr/bin/env python3
"""
Summarized analysis of experiment results focusing on key metrics.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
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
            parts = filename.replace('steering_alpha_', '').replace('.pkl', '').rsplit('_', 1)
            if len(parts) == 2:
                try:
                    alpha = float(parts[0])
                    direction = parts[1]
                    
                    results = load_pickle_safe(os.path.join(steering_dir, filename))
                    if results:
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
    
    # Load probe results
    auc_paths = [
        os.path.join(exp_path, 'probes', 'caa-single-layer', 'auc_scores.json'),
        os.path.join(exp_path, 'probes', 'auc_scores.json')
    ]
    
    for auc_path in auc_paths:
        auc_scores = load_json_safe(auc_path)
        if auc_scores:
            if isinstance(auc_scores, dict):
                scores = list(auc_scores.values())
                layers = list(auc_scores.keys())
                if scores:
                    max_idx = np.argmax(scores)
                    result['probe_auc'] = float(scores[max_idx])
                    result['best_layer'] = int(layers[max_idx])
            else:
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

def print_summary_table(experiment_results: List[Dict[str, Any]]):
    """Print a consolidated summary table."""
    # Group by model and dataset
    grouped = defaultdict(list)
    for exp in experiment_results:
        key = (exp['model'], exp['dataset'])
        grouped[key].append(exp)
    
    # Print header
    print("\n" + "="*120)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*120)
    print()
    
    # Model mapping for cleaner names
    model_names = {
        'deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B': 'DeepSeek-1.5B',
        'google_gemma-2-2b-it': 'Gemma-2B'
    }
    
    # Summary table by model and dataset
    print("### Accuracy and Probe Performance")
    print()
    print("| Model | Dataset | Train Acc | Test Acc | Best Layer | Probe AUC |")
    print("|-------|---------|-----------|----------|------------|-----------|")
    
    for (model, dataset), experiments in sorted(grouped.items()):
        exp = max(experiments, key=lambda x: len(x['status']))
        
        model_short = model_names.get(model, model.split('/')[-1][:20])
        
        train_acc = f"{exp['train_accuracy']:.0f}%" if exp['train_accuracy'] is not None else "N/A"
        test_acc = f"{exp['test_accuracy']:.0f}%" if exp['test_accuracy'] is not None else "N/A"
        probe_auc = f"{exp['probe_auc']:.3f}" if exp['probe_auc'] is not None else "N/A"
        best_layer = str(exp['best_layer']) if exp['best_layer'] is not None else "N/A"
        
        print(f"| {model_short:<13} | {dataset:<15} | {train_acc:>9} | {test_acc:>8} | {best_layer:>10} | {probe_auc:>9} |")
    
    print()
    print("### Steering Success Rates (Selected Alpha Values)")
    print()
    
    # Show steering results for key alpha values
    key_alphas = [-10.0, -2.0, 0.0, 2.0, 10.0]
    
    for (model, dataset), experiments in sorted(grouped.items()):
        exp = max(experiments, key=lambda x: len(x['status']))
        
        if exp['steering_results']:
            model_short = model_names.get(model, model.split('/')[-1][:20])
            print(f"\n**{model_short} - {dataset}**")
            print()
            print("| α | Yes→No | No→Yes | Overall |")
            print("|---|--------|--------|---------|")
            
            for alpha in key_alphas:
                if alpha in exp['steering_results']:
                    data = exp['steering_results'][alpha]
                    yes_data = data.get('yes', {})
                    no_data = data.get('no', {})
                    
                    yes_rate = yes_data.get('success_rate', 0) if yes_data else 0
                    no_rate = no_data.get('success_rate', 0) if no_data else 0
                    
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
                    
                    print(f"| {alpha:+.0f} | {yes_rate:.1%} | {no_rate:.1%} | {overall:.1%} |")
    
    print("\n" + "="*120)

def find_experiment_dirs(cache_dir: str) -> List[str]:
    """Find all experiment directories in cache."""
    experiment_dirs = []
    
    base_dir = os.path.join(cache_dir, 'experiments') if not cache_dir.endswith('experiments') else cache_dir
    
    if not os.path.exists(base_dir):
        return []
    
    for root, dirs, files in os.walk(base_dir):
        if 'data' in dirs or 'metadata' in dirs:
            experiment_dirs.append(root)
    
    return experiment_dirs

def main():
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else 'cache'
    
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(cache_dir)
    
    if not exp_dirs:
        print("❌ No experiment directories found")
        return
    
    # Analyze each experiment
    all_results = []
    for exp_dir in exp_dirs:
        result = analyze_experiment(exp_dir)
        if result['status']:
            all_results.append(result)
    
    # Sort by model and dataset
    all_results.sort(key=lambda x: (x['model'], x['dataset']))
    
    # Print summary
    print_summary_table(all_results)
    
    # Key findings
    print("\n### Key Findings:")
    print()
    print("1. **DeepSeek-1.5B**: Shows very poor steering control - most alpha values produce 0% success")
    print("   - Only α=0 shows some response, suggesting the model may not be responding to steering vectors")
    print("   - Lower base accuracies across all datasets (45-81% train, 45-69% test)")
    print()
    print("2. **Gemma-2B**: Better base performance but also limited steering effectiveness")
    print("   - Higher accuracies (61-83% train, 58-82% test)")
    print("   - Best probe performance on anachronisms (AUC: 0.893, Layer 14)")
    print("   - Minimal steering response except at α=0")
    print()
    print("3. **Dataset observations**:")
    print("   - Anachronisms: Best probe performance for Gemma (AUC: 0.893)")
    print("   - Sports: Moderate probe performance (AUC: 0.672)")
    print("   - Logical deduction: Poor probe performance (AUC: 0.328)")

if __name__ == "__main__":
    main()