#!/usr/bin/env python3
"""
Quick script to summarize experiment results from cache.

Usage:
    python summarize_results.py [cache_dir]

If no cache_dir is provided, it will look for 'cache' or 'results_cache' directories.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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

def find_cache_directories() -> List[str]:
    """Find available cache directories."""
    possible_dirs = ['cache', 'results_cache', 'cache/experiments', 'results_cache/experiments']
    found_dirs = []
    
    for dir_name in possible_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            found_dirs.append(dir_name)
    
    return found_dirs

def analyze_experiment_cache(cache_path: str) -> Dict[str, Any]:
    """Analyze a single experiment cache directory."""
    result = {
        'path': cache_path,
        'has_generations': False,
        'has_activations': False,
        'has_probes': False,
        'probe_performance': None,
        'steering_results': {},
        'model_name': 'unknown',
        'dataset_name': 'unknown'
    }
    
    # Try to infer model and dataset from path
    path_parts = cache_path.split('/')
    if len(path_parts) >= 2:
        result['model_name'] = path_parts[-2] if path_parts[-2] != 'experiments' else 'unknown'
        result['dataset_name'] = path_parts[-1]
    
    # Check for generations
    train_gen_path = os.path.join(cache_path, 'data', 'train_generations.pkl')
    test_gen_path = os.path.join(cache_path, 'data', 'test_generations.pkl')
    if os.path.exists(train_gen_path) and os.path.exists(test_gen_path):
        result['has_generations'] = True
    
    # Check for activations
    train_act_path = os.path.join(cache_path, 'data', 'train_activations.pkl')
    test_act_path = os.path.join(cache_path, 'data', 'test_activations.pkl')
    if os.path.exists(train_act_path) and os.path.exists(test_act_path):
        result['has_activations'] = True
    
    # Check for probes and load performance
    auc_scores_path = os.path.join(cache_path, 'probes', 'caa-single-layer', 'auc_scores.json')
    probe_coef_path = os.path.join(cache_path, 'probes', 'caa-single-layer', 'probe_coefficients.pkl')
    
    if os.path.exists(auc_scores_path) and os.path.exists(probe_coef_path):
        result['has_probes'] = True
        auc_scores = load_json_safe(auc_scores_path)
        if auc_scores:
            result['probe_performance'] = {
                'best_score': float(max(auc_scores)),
                'avg_score': float(np.mean(auc_scores)),
                'best_layer': int(np.argmax(auc_scores)),
                'total_layers': len(auc_scores)
            }
    
    # Check for steering results
    steering_dir = os.path.join(cache_path, 'steering')
    if os.path.exists(steering_dir):
        for filename in os.listdir(steering_dir):
            if filename.endswith('.pkl'):
                # Parse filename like "alpha_-2.0_yes.pkl"
                parts = filename.replace('.pkl', '').split('_')
                if len(parts) >= 3 and parts[0] == 'alpha':
                    try:
                        alpha = float(parts[1])
                        direction = parts[2]
                        
                        steering_data = load_pickle_safe(os.path.join(steering_dir, filename))
                        if steering_data:
                            successes = sum(1 for r in steering_data if r.get('success', False))
                            total = len(steering_data)
                            success_rate = successes / total if total > 0 else 0
                            
                            if alpha not in result['steering_results']:
                                result['steering_results'][alpha] = {}
                            result['steering_results'][alpha][direction] = {
                                'success_rate': success_rate,
                                'successes': successes,
                                'total': total
                            }
                    except:
                        continue
    
    return result

def print_summary(cache_dir: str):
    """Print a comprehensive summary of cache results."""
    print(f"{'='*80}")
    print(f"📊 EXPERIMENT RESULTS SUMMARY")
    print(f"Cache Directory: {cache_dir}")
    print(f"{'='*80}")
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    # Find all experiment directories
    experiment_dirs = []
    
    # Check if this is already an experiments directory
    if os.path.basename(cache_dir) == 'experiments':
        base_dir = cache_dir
    else:
        base_dir = os.path.join(cache_dir, 'experiments')
    
    if os.path.exists(base_dir):
        for model_dir in os.listdir(base_dir):
            model_path = os.path.join(base_dir, model_dir)
            if os.path.isdir(model_path):
                for dataset_dir in os.listdir(model_path):
                    dataset_path = os.path.join(model_path, dataset_dir)
                    if os.path.isdir(dataset_path):
                        for split_dir in os.listdir(dataset_path):
                            split_path = os.path.join(dataset_path, split_dir)
                            if os.path.isdir(split_path):
                                for exp_dir in os.listdir(split_path):
                                    exp_path = os.path.join(split_path, exp_dir)
                                    if os.path.isdir(exp_path):
                                        experiment_dirs.append(exp_path)
    
    if not experiment_dirs:
        print("❌ No experiment directories found")
        return
    
    print(f"🔍 Found {len(experiment_dirs)} experiment directories")
    print()
    
    # Analyze each experiment
    all_results = []
    for exp_dir in experiment_dirs:
        result = analyze_experiment_cache(exp_dir)
        all_results.append(result)
    
    # Group by model and dataset
    experiments_by_model = {}
    for result in all_results:
        model = result['model_name']
        if model not in experiments_by_model:
            experiments_by_model[model] = []
        experiments_by_model[model].append(result)
    
    # Print model-by-model summary
    for model_name, model_results in experiments_by_model.items():
        print(f"🤖 MODEL: {model_name}")
        print(f"{'─'*60}")
        
        for result in model_results:
            dataset = result['dataset_name']
            status_icons = []
            
            if result['has_generations']:
                status_icons.append("📝")
            if result['has_activations']:
                status_icons.append("🧠")
            if result['has_probes']:
                status_icons.append("🎯")
            if result['steering_results']:
                status_icons.append("🎮")
            
            status_str = " ".join(status_icons) if status_icons else "❌"
            
            print(f"  📊 {dataset:<20} | {status_str}")
            
            # Print probe performance
            if result['probe_performance']:
                perf = result['probe_performance']
                print(f"     🎯 Best: {perf['best_score']:.4f} (L{perf['best_layer']}) | Avg: {perf['avg_score']:.4f}")
            
            # Print steering results
            if result['steering_results']:
                steering_summary = []
                for alpha in sorted(result['steering_results'].keys()):
                    alpha_results = result['steering_results'][alpha]
                    for direction, data in alpha_results.items():
                        rate = data['success_rate']
                        steering_summary.append(f"α{alpha:+.1f}({direction}): {rate:.1%}")
                
                if steering_summary:
                    print(f"     🎮 {' | '.join(steering_summary[:3])}")  # Show first 3
                    if len(steering_summary) > 3:
                        print(f"          {' | '.join(steering_summary[3:6])}")  # Show next 3
        print()
    
    # Overall statistics
    total_experiments = len(all_results)
    completed_experiments = sum(1 for r in all_results if r['has_generations'] and r['has_probes'])
    
    print(f"📈 OVERALL STATISTICS:")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Completed: {completed_experiments}")
    print(f"   Success rate: {completed_experiments/total_experiments:.1%}" if total_experiments > 0 else "   Success rate: N/A")
    
    # Best performing models
    probe_results = [r for r in all_results if r['probe_performance']]
    if probe_results:
        print(f"\n🏆 TOP PERFORMING MODELS:")
        probe_results.sort(key=lambda x: x['probe_performance']['best_score'], reverse=True)
        
        for i, result in enumerate(probe_results[:5], 1):
            perf = result['probe_performance']
            model_short = result['model_name'].split('/')[-1] if '/' in result['model_name'] else result['model_name']
            print(f"   {i}. {model_short[:25]:<25} | {result['dataset_name']:<15} | {perf['best_score']:.4f}")
    
    print(f"{'='*80}")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]
    else:
        # Auto-detect cache directories
        cache_dirs = find_cache_directories()
        if not cache_dirs:
            print("❌ No cache directories found. Please specify a cache directory.")
            print("Usage: python summarize_results.py [cache_dir]")
            return
        
        cache_dir = cache_dirs[0]  # Use the first found cache directory
        if len(cache_dirs) > 1:
            print(f"🔍 Found multiple cache directories: {cache_dirs}")
            print(f"📁 Using: {cache_dir}")
            print()
    
    print_summary(cache_dir)

if __name__ == "__main__":
    main()