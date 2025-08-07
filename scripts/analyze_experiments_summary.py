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
                        total = len(results)
                        successes = sum(1 for r in results if r.get('success', False))
                        
                        # Count parsing categories
                        parsed = sum(1 for r in results if r.get('category', '') != 'unparsed')
                        unparsed = sum(1 for r in results if r.get('category', '') == 'unparsed')
                        failures = sum(1 for r in results if r.get('category', '') == 'failure')
                        
                        # Success rate calculations
                        success_rate_all = successes / total if total > 0 else 0
                        success_rate_parsed = successes / parsed if parsed > 0 else 0
                        failure_rate_parsed = failures / parsed if parsed > 0 else 0
                        
                        steering_data[alpha][direction] = {
                            'total': total,
                            'successes': successes,
                            'failures': failures,
                            'parsed': parsed,
                            'unparsed': unparsed,
                            'success_rate_all': success_rate_all,
                            'success_rate_parsed': success_rate_parsed,
                            'failure_rate_parsed': failure_rate_parsed,
                            'parse_rate': parsed / total if total > 0 else 0
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
            print("| α | Yes→No (All) | No→Yes (All) | Parse Rate | Yes→No (Parsed) | No→Yes (Parsed) |")
            print("|---|--------------|--------------|------------|-----------------|-----------------|")
            
            for alpha in key_alphas:
                if alpha in exp['steering_results']:
                    data = exp['steering_results'][alpha]
                    yes_data = data.get('yes', {})
                    no_data = data.get('no', {})
                    
                    yes_rate_all = yes_data.get('success_rate_all', 0) if yes_data else 0
                    no_rate_all = no_data.get('success_rate_all', 0) if no_data else 0
                    yes_rate_parsed = yes_data.get('success_rate_parsed', 0) if yes_data else 0
                    no_rate_parsed = no_data.get('success_rate_parsed', 0) if no_data else 0
                    yes_parse_rate = yes_data.get('parse_rate', 0) if yes_data else 0
                    no_parse_rate = no_data.get('parse_rate', 0) if no_data else 0
                    
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
                    avg_parse_rate = (yes_parse_rate + no_parse_rate) / 2 if (yes_data or no_data) else 0
                    
                    print(f"| {alpha:+.0f} | {yes_rate_all:.1%} | {no_rate_all:.1%} | {avg_parse_rate:.1%} | {yes_rate_parsed:.1%} | {no_rate_parsed:.1%} |")
    
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
    # Print parsing statistics summary
    print("\n### Parsing Statistics")
    print()
    # Re-group for parsing stats
    grouped = defaultdict(list)
    for exp in all_results:
        key = (exp['model'], exp['dataset'])
        grouped[key].append(exp)
    
    # Model mapping for cleaner names (same as above)
    model_names = {
        'deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B': 'DeepSeek-1.5B',
        'google_gemma-2-2b-it': 'Gemma-2B'
    }
    
    for (model, dataset), experiments in sorted(grouped.items()):
        exp = max(experiments, key=lambda x: len(x['status']))
        if exp['steering_results']:
            model_short = model_names.get(model, model.split('/')[-1][:20])
            print(f"\n**{model_short} - {dataset}**")
            
            # Calculate average parse rates
            parse_rates = []
            for alpha, data in exp['steering_results'].items():
                for direction in ['yes', 'no']:
                    if data.get(direction):
                        parse_rates.append(data[direction].get('parse_rate', 0))
            
            if parse_rates:
                avg_parse = sum(parse_rates) / len(parse_rates)
                print(f"Average parse rate: {avg_parse:.1%}")
    
    print("\n### Key Findings:")
    print()
    
    # Analyze model performance dynamically
    model_stats = defaultdict(lambda: {
        'train_accs': [], 'test_accs': [], 'probe_aucs': [], 
        'steering_success': [], 'parse_rates': []
    })
    
    for exp in all_results:
        model = exp['model']
        if exp['train_accuracy'] is not None:
            model_stats[model]['train_accs'].append(exp['train_accuracy'])
        if exp['test_accuracy'] is not None:
            model_stats[model]['test_accs'].append(exp['test_accuracy'])
        if exp['probe_auc'] is not None:
            model_stats[model]['probe_aucs'].append(exp['probe_auc'])
        
        # Collect steering success rates
        for alpha, data in exp['steering_results'].items():
            for direction in ['yes', 'no']:
                if data.get(direction):
                    model_stats[model]['steering_success'].append(data[direction].get('success_rate_all', 0))
                    model_stats[model]['parse_rates'].append(data[direction].get('parse_rate', 0))
    
    # Print findings for each model
    for i, (model, stats) in enumerate(sorted(model_stats.items()), 1):
        model_short = model_names.get(model, model.split('/')[-1][:20])
        print(f"{i}. **{model_short}**:")
        
        if stats['train_accs']:
            min_train = min(stats['train_accs'])
            max_train = max(stats['train_accs'])
            print(f"   - Train accuracy: {min_train:.0f}-{max_train:.0f}%")
        
        if stats['test_accs']:
            min_test = min(stats['test_accs'])
            max_test = max(stats['test_accs'])
            print(f"   - Test accuracy: {min_test:.0f}-{max_test:.0f}%")
        
        if stats['probe_aucs']:
            avg_auc = sum(stats['probe_aucs']) / len(stats['probe_aucs'])
            best_auc = max(stats['probe_aucs'])
            print(f"   - Probe AUC: avg {avg_auc:.3f}, best {best_auc:.3f}")
        
        if stats['steering_success']:
            non_zero = [s for s in stats['steering_success'] if s > 0]
            if non_zero:
                avg_success = sum(non_zero) / len(non_zero)
                print(f"   - Steering: {len(non_zero)}/{len(stats['steering_success'])} conditions with >0% success (avg {avg_success:.1%} when working)")
            else:
                print(f"   - Steering: No successful steering observed")
        
        if stats['parse_rates']:
            avg_parse = sum(stats['parse_rates']) / len(stats['parse_rates'])
            print(f"   - Average parse rate: {avg_parse:.1%}")
        print()
    
    # Dataset-level insights
    dataset_stats = defaultdict(lambda: {'probe_aucs': [], 'accuracies': []})
    for exp in all_results:
        dataset = exp['dataset']
        if exp['probe_auc'] is not None:
            dataset_stats[dataset]['probe_aucs'].append(exp['probe_auc'])
        if exp['test_accuracy'] is not None:
            dataset_stats[dataset]['accuracies'].append(exp['test_accuracy'])
    
    print("**Dataset Performance:**")
    for dataset, stats in sorted(dataset_stats.items()):
        insights = []
        if stats['probe_aucs']:
            avg_auc = sum(stats['probe_aucs']) / len(stats['probe_aucs'])
            insights.append(f"avg probe AUC {avg_auc:.3f}")
        if stats['accuracies']:
            avg_acc = sum(stats['accuracies']) / len(stats['accuracies'])
            insights.append(f"avg test accuracy {avg_acc:.0f}%")
        if insights:
            print(f"   - {dataset}: {', '.join(insights)}")

if __name__ == "__main__":
    main()