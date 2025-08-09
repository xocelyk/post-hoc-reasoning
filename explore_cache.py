#!/usr/bin/env python3
"""
Interactive cache explorer for post-hoc reasoning experiments.
Works over SSH without requiring a GUI or Jupyter widgets.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime

class CacheExplorer:
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        self.experiments_dir = os.path.join(cache_dir, 'experiments')
        self.current_exp = None
        self.experiments = self._scan_experiments()
        
    def _scan_experiments(self) -> List[Dict[str, Any]]:
        """Scan cache directory for all experiments."""
        experiments = []
        
        if not os.path.exists(self.experiments_dir):
            return experiments
            
        for root, dirs, files in os.walk(self.experiments_dir):
            if 'data' in dirs or 'metadata' in dirs:
                # Extract experiment info
                path_parts = Path(root).parts
                exp_info = {
                    'path': root,
                    'model': 'unknown',
                    'dataset': 'unknown',
                    'id': 'unknown'
                }
                
                # Parse path to get model, dataset, and experiment ID
                for i, part in enumerate(path_parts):
                    if part == 'experiments' and i + 2 < len(path_parts):
                        exp_info['model'] = path_parts[i + 1]
                        exp_info['dataset'] = path_parts[i + 2]
                        if i + 4 < len(path_parts):
                            exp_info['id'] = path_parts[i + 4]
                        break
                
                # Check what data is available
                exp_info['has_train'] = os.path.exists(os.path.join(root, 'data', 'train_generations.pkl'))
                exp_info['has_test'] = os.path.exists(os.path.join(root, 'data', 'test_generations.pkl'))
                exp_info['has_probes'] = os.path.exists(os.path.join(root, 'probes'))
                exp_info['has_steering'] = os.path.exists(os.path.join(root, 'steering'))
                
                # Get modification time
                try:
                    mtime = os.path.getmtime(root)
                    exp_info['modified'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                except:
                    exp_info['modified'] = 'unknown'
                
                experiments.append(exp_info)
        
        return sorted(experiments, key=lambda x: (x['model'], x['dataset']))
    
    def list_experiments(self, filter_model: Optional[str] = None, filter_dataset: Optional[str] = None):
        """List all experiments with optional filtering."""
        filtered = self.experiments
        
        if filter_model:
            filtered = [e for e in filtered if filter_model.lower() in e['model'].lower()]
        if filter_dataset:
            filtered = [e for e in filtered if filter_dataset.lower() in e['dataset'].lower()]
        
        if not filtered:
            print("No experiments found matching criteria.")
            return
        
        # Group by model
        by_model = defaultdict(list)
        for exp in filtered:
            by_model[exp['model']].append(exp)
        
        print(f"\nFound {len(filtered)} experiments:")
        print("=" * 100)
        
        for i, (model, exps) in enumerate(sorted(by_model.items())):
            print(f"\n{i+1}. {model}")
            for j, exp in enumerate(exps):
                status = []
                if exp['has_train']: status.append('train')
                if exp['has_test']: status.append('test')
                if exp['has_probes']: status.append('probes')
                if exp['has_steering']: status.append('steering')
                
                print(f"   {i+1}.{j+1} {exp['dataset']:<20} [{', '.join(status)}] {exp['modified']} ({exp['id'][:8]}...)")
    
    def select_experiment(self, selection: str) -> Optional[Dict[str, Any]]:
        """Select an experiment by number (e.g., '1.2' for model 1, experiment 2)."""
        try:
            if '.' in selection:
                model_idx, exp_idx = map(int, selection.split('.'))
            else:
                # If just a number, assume first model
                model_idx, exp_idx = 1, int(selection)
            
            by_model = defaultdict(list)
            for exp in self.experiments:
                by_model[exp['model']].append(exp)
            
            models = sorted(by_model.keys())
            if 1 <= model_idx <= len(models):
                model = models[model_idx - 1]
                exps = by_model[model]
                if 1 <= exp_idx <= len(exps):
                    self.current_exp = exps[exp_idx - 1]
                    return self.current_exp
        except:
            pass
        
        print("Invalid selection. Use format '1.2' for model 1, experiment 2.")
        return None
    
    def show_experiment_info(self, exp: Optional[Dict[str, Any]] = None):
        """Show detailed information about an experiment."""
        if exp is None:
            exp = self.current_exp
        if exp is None:
            print("No experiment selected.")
            return
        
        print(f"\n{'='*80}")
        print(f"Experiment: {exp['model']} - {exp['dataset']}")
        print(f"Path: {exp['path']}")
        print(f"ID: {exp['id']}")
        print(f"Modified: {exp['modified']}")
        print(f"{'='*80}")
        
        # Load and show accuracies
        if exp['has_train'] or exp['has_test']:
            print("\nAccuracy Results:")
            for split in ['train', 'test']:
                if exp[f'has_{split}']:
                    gen_path = os.path.join(exp['path'], 'data', f'{split}_generations.pkl')
                    try:
                        with open(gen_path, 'rb') as f:
                            gens = pickle.load(f)
                        correct = sum(1 for g in gens if g.get('pred_answer') == g.get('correct_answer'))
                        acc = correct / len(gens) * 100 if gens else 0
                        print(f"  {split.capitalize()}: {acc:.1f}% ({correct}/{len(gens)})")
                        
                        # Show label distribution
                        labels = Counter(g.get('correct_answer', 'N/A') for g in gens)
                        print(f"    Labels: {dict(labels)}")
                    except Exception as e:
                        print(f"  Error loading {split} data: {e}")
        
        # Show probe results
        if exp['has_probes']:
            print("\nProbe Results:")
            auc_paths = [
                os.path.join(exp['path'], 'probes', 'caa-single-layer', 'auc_scores.json'),
                os.path.join(exp['path'], 'probes', 'auc_scores.json')
            ]
            
            for auc_path in auc_paths:
                if os.path.exists(auc_path):
                    try:
                        with open(auc_path, 'r') as f:
                            scores = json.load(f)
                        if isinstance(scores, dict):
                            scores = list(scores.values())
                        if scores:
                            print(f"  Max AUC: {max(scores):.4f}")
                            print(f"  Mean AUC: {np.mean(scores):.4f}")
                            print(f"  Best layer: {np.argmax(scores)}")
                            break
                    except Exception as e:
                        print(f"  Error loading probe results: {e}")
        
        # Show steering results summary
        if exp['has_steering']:
            print("\nSteering Results:")
            steering_dir = os.path.join(exp['path'], 'steering')
            steering_files = [f for f in os.listdir(steering_dir) if f.endswith('.pkl')]
            alphas = set()
            for f in steering_files:
                if f.startswith('steering_alpha_'):
                    try:
                        alpha = float(f.split('_')[2])
                        alphas.add(alpha)
                    except:
                        pass
            if alphas:
                print(f"  Alpha values tested: {sorted(alphas)}")
                print(f"  Total files: {len(steering_files)}")
                print(f"  Use 'steer <alpha> [direction] [n]' to view examples")
    
    def explore_generations(self, exp: Optional[Dict[str, Any]] = None, split: str = 'train', 
                           num_examples: int = 5, filter_incorrect: bool = False):
        """Explore actual generation examples."""
        if exp is None:
            exp = self.current_exp
        if exp is None:
            print("No experiment selected.")
            return
        
        gen_path = os.path.join(exp['path'], 'data', f'{split}_generations.pkl')
        if not os.path.exists(gen_path):
            print(f"No {split} generations found.")
            return
        
        try:
            with open(gen_path, 'rb') as f:
                gens = pickle.load(f)
        except Exception as e:
            print(f"Error loading generations: {e}")
            return
        
        if filter_incorrect:
            gens = [g for g in gens if g.get('pred_answer') != g.get('correct_answer')]
            if not gens:
                print("No incorrect predictions found.")
                return
        
        print(f"\n{split.upper()} GENERATIONS ({len(gens)} total, showing {min(num_examples, len(gens))})")
        print("=" * 80)
        
        for i, gen in enumerate(gens[:num_examples]):
            correct = gen.get('correct_answer', 'N/A')
            pred = gen.get('pred_answer', 'N/A')
            is_correct = pred == correct
            
            print(f"\nExample {i+1}: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print("-" * 40)
            
            # Show input/question
            if 'question' in gen:
                q = gen['question']
                if len(q) > 200:
                    q = q[:200] + "..."
                print(f"Question: {q}")
            elif 'input' in gen:
                inp = gen['input']
                if len(inp) > 200:
                    inp = inp[:200] + "..."
                print(f"Input: {inp}")
            
            print(f"Correct: {correct}")
            print(f"Predicted: {pred}")
            
            # Show response
            response = gen.get('response', gen.get('generated_text', 'N/A'))
            if isinstance(response, str):
                print(f"Response (length: {len(response)} chars):")
                print("-" * 40)
                print(response)
                print("-" * 40)
            
            if 'category' in gen:
                print(f"Category: {gen['category']}")
        
        print("\n" + "=" * 80)
    
    def explore_steering_results(self, exp: Optional[Dict[str, Any]] = None, alpha: float = 0.0, 
                               direction: Optional[str] = None, num_examples: int = 5):
        """Explore steering results for a specific alpha value."""
        if exp is None:
            exp = self.current_exp
        if exp is None:
            print("No experiment selected.")
            return
        
        if not exp['has_steering']:
            print("No steering results available for this experiment.")
            return
        
        steering_dir = os.path.join(exp['path'], 'steering')
        
        # Find matching files - be flexible with float/int formatting
        matching_files = []
        
        # Try both integer and float formats in filename
        alpha_strs = []
        if alpha == int(alpha):
            # For whole numbers, try both "4" and "4.0" formats
            alpha_strs.append(str(int(alpha)))
            alpha_strs.append(f"{alpha:.1f}")
        else:
            # For decimals, use the float format
            alpha_strs.append(str(alpha))
        
        for filename in os.listdir(steering_dir):
            if filename.startswith('steering_alpha_') and filename.endswith('.pkl'):
                # Check if this file matches our alpha value
                for alpha_str in alpha_strs:
                    expected_prefix = f'steering_alpha_{alpha_str}_'
                    if filename.startswith(expected_prefix):
                        file_direction = filename.replace(expected_prefix, '').replace('.pkl', '')
                        if direction is None or direction == file_direction:
                            matching_files.append(filename)
                            break
        
        if not matching_files:
            print(f"No steering results found for alpha={alpha}" + (f" direction={direction}" if direction else ""))
            # Show available alphas
            available_alphas = set()
            for f in os.listdir(steering_dir):
                if f.startswith('steering_alpha_') and f.endswith('.pkl'):
                    try:
                        # Extract the part between 'steering_alpha_' and the last underscore
                        parts = f.replace('steering_alpha_', '').rsplit('_', 1)
                        if len(parts) >= 1:
                            alpha_val = float(parts[0])
                            available_alphas.add(alpha_val)
                    except Exception as e:
                        print(f"Debug: Failed to parse {f}: {e}")
            if available_alphas:
                print(f"Available alpha values: {sorted(available_alphas)}")
                print(f"Try: steer {sorted(available_alphas)[0]}")
            return
        
        # Load and display results
        for filename in matching_files:
            file_parts = filename.replace('steering_alpha_', '').replace('.pkl', '').rsplit('_', 1)
            if len(file_parts) == 2:
                file_alpha = float(file_parts[0])
                file_direction = file_parts[1]
                
                print(f"\nSTEERING RESULTS: alpha={file_alpha}, direction={file_direction}")
                print("=" * 80)
                
                try:
                    with open(os.path.join(steering_dir, filename), 'rb') as f:
                        results = pickle.load(f)
                    
                    if not results:
                        print("No results in file.")
                        continue
                    
                    # Calculate statistics
                    total = len(results)
                    successes = sum(1 for r in results if r.get('success', False))
                    failures = sum(1 for r in results if r.get('category') == 'failure')
                    unparsed = sum(1 for r in results if r.get('category') == 'unparsed')
                    parsed = total - unparsed
                    
                    success_rate = successes / total * 100 if total > 0 else 0
                    success_rate_parsed = successes / parsed * 100 if parsed > 0 else 0
                    
                    print(f"Total examples: {total}")
                    print(f"Success rate (all): {success_rate:.1f}% ({successes}/{total})")
                    print(f"Success rate (parsed only): {success_rate_parsed:.1f}% ({successes}/{parsed})")
                    print(f"Parsed: {parsed} ({parsed/total*100:.1f}%)")
                    print(f"Unparsed: {unparsed} ({unparsed/total*100:.1f}%)")
                    print(f"Failures: {failures}")
                    
                    # Check for extreme steering effects
                    if unparsed > parsed and abs(file_alpha) > 5:
                        print(f"\n⚠️  High unparsed rate at extreme alpha={file_alpha}")
                        print("   Model may be producing corrupted output due to strong steering")
                    
                    print(f"\nShowing {min(num_examples, len(results))} examples:")
                    print("-" * 80)
                    
                    # Show examples
                    for i, result in enumerate(results[:num_examples]):
                        success = result.get('success', False)
                        category = result.get('category', 'unknown')
                        
                        print(f"\nExample {i+1}: {'✓ SUCCESS' if success else '✗ FAILURE'} (category: {category})")
                        
                        # Show question/input if available
                        if 'question' in result:
                            q = result['question']
                            if len(q) > 200:
                                q = q[:200] + "..."
                            print(f"Question: {q}")
                        elif 'input' in result:
                            inp = result['input']
                            if len(inp) > 200:
                                inp = inp[:200] + "..."
                            print(f"Input: {inp}")
                        elif 'original_prompt' in result:
                            prompt = result['original_prompt']
                            if isinstance(prompt, str) and len(prompt) > 200:
                                prompt = "..." + prompt[-200:]
                            print(f"Prompt (end): {prompt}")
                        
                        # Show original and steered answers
                        if 'original_answer' in result:
                            print(f"Original answer: {result['original_answer']}")
                        if 'new_answer' in result:
                            print(f"Steered answer: {result['new_answer']}")
                        elif 'steered_answer' in result:
                            print(f"Steered answer: {result['steered_answer']}")
                        if 'target_answer' in result:
                            print(f"Target answer: {result['target_answer']}")
                        if 'correct_answer' in result:
                            print(f"Correct answer: {result['correct_answer']}")
                        
                        # Show response - try multiple possible keys
                        response = result.get('steered_generation', 
                                  result.get('response', 
                                  result.get('steered_response', 'N/A')))
                        
                        # Handle case where steered_generation is a list
                        if isinstance(response, list) and response:
                            response = response[0] if isinstance(response[0], str) else str(response[0])
                        
                        if isinstance(response, str):
                            # Show full response for steered results
                            print(f"Response (length: {len(response)} chars):")
                            print("-" * 80)
                            print(response)
                            print("-" * 80)
                        
                        print("-" * 40)
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print("\n" + "=" * 80)
    
    def analyze_probe_training_data(self, exp: Optional[Dict[str, Any]] = None):
        """Analyze what data the probes were trained on."""
        if exp is None:
            exp = self.current_exp
        if exp is None:
            print("No experiment selected.")
            return
        
        train_path = os.path.join(exp['path'], 'data', 'train_generations.pkl')
        if not os.path.exists(train_path):
            print("No training data found.")
            return
        
        try:
            with open(train_path, 'rb') as f:
                train_gens = pickle.load(f)
        except Exception as e:
            print(f"Error loading training data: {e}")
            return
        
        # Probes are trained only on correct predictions
        correct_preds = [g for g in train_gens if g.get('pred_answer') == g.get('correct_answer')]
        
        print(f"\nPROBE TRAINING DATA ANALYSIS")
        print("=" * 80)
        print(f"Total training examples: {len(train_gens)}")
        print(f"Correct predictions (used for probe training): {len(correct_preds)} ({len(correct_preds)/len(train_gens)*100:.1f}%)")
        
        # Analyze label distribution
        all_labels = Counter(g.get('correct_answer', 'N/A') for g in train_gens)
        correct_labels = Counter(g.get('correct_answer', 'N/A') for g in correct_preds)
        
        print(f"\nLabel distribution (all examples): {dict(all_labels)}")
        print(f"Label distribution (correct only): {dict(correct_labels)}")
        
        # Check for issues
        if len(correct_labels) < 2:
            print("\n⚠️  WARNING: Probe training data has fewer than 2 classes!")
            print("   This will cause probe training to fail or produce meaningless results.")
        
        # Show per-class accuracy
        print("\nPer-class accuracy:")
        for label in all_labels:
            total = sum(1 for g in train_gens if g.get('correct_answer') == label)
            correct = sum(1 for g in train_gens if g.get('correct_answer') == label and g.get('pred_answer') == label)
            acc = correct / total * 100 if total > 0 else 0
            print(f"  '{label}': {correct}/{total} ({acc:.1f}%)")
    
    def compare_experiments(self, indices: List[str]):
        """Compare multiple experiments side by side."""
        selected_exps = []
        for idx in indices:
            exp = self.select_experiment(idx)
            if exp:
                selected_exps.append(exp)
        
        if len(selected_exps) < 2:
            print("Please select at least 2 experiments to compare.")
            return
        
        print(f"\nCOMPARING {len(selected_exps)} EXPERIMENTS")
        print("=" * 80)
        
        # Create comparison table
        for exp in selected_exps:
            print(f"\n{exp['model']} - {exp['dataset']}:")
            
            # Load accuracies
            for split in ['train', 'test']:
                if exp[f'has_{split}']:
                    gen_path = os.path.join(exp['path'], 'data', f'{split}_generations.pkl')
                    try:
                        with open(gen_path, 'rb') as f:
                            gens = pickle.load(f)
                        correct = sum(1 for g in gens if g.get('pred_answer') == g.get('correct_answer'))
                        acc = correct / len(gens) * 100 if gens else 0
                        print(f"  {split}: {acc:.1f}%")
                    except:
                        pass
            
            # Load probe AUC
            if exp['has_probes']:
                auc_paths = [
                    os.path.join(exp['path'], 'probes', 'caa-single-layer', 'auc_scores.json'),
                    os.path.join(exp['path'], 'probes', 'auc_scores.json')
                ]
                
                for auc_path in auc_paths:
                    if os.path.exists(auc_path):
                        try:
                            with open(auc_path, 'r') as f:
                                scores = json.load(f)
                            if isinstance(scores, dict):
                                scores = list(scores.values())
                            if scores:
                                print(f"  Max AUC: {max(scores):.4f}")
                                break
                        except:
                            pass

def main():
    parser = argparse.ArgumentParser(description='Explore post-hoc reasoning experiment cache')
    parser.add_argument('cache_dir', nargs='?', default='cache', help='Cache directory path')
    parser.add_argument('--model', help='Filter experiments by model name')
    parser.add_argument('--dataset', help='Filter experiments by dataset name')
    args = parser.parse_args()
    
    explorer = CacheExplorer(args.cache_dir)
    
    print(f"\nPost-Hoc Reasoning Cache Explorer")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Found {len(explorer.experiments)} experiments")
    print("\nCommands:")
    print("  list [model] [dataset] - List experiments (optionally filtered)")
    print("  select <number>        - Select experiment (e.g., '1.2' for model 1, exp 2)")
    print("  info                   - Show current experiment details")
    print("  show [n] [incorrect]   - Show n examples (add 'incorrect' to filter)")
    print("  steer <alpha> [dir] [n] - Show steering results for alpha value")
    print("  probe                  - Analyze probe training data")
    print("  compare <n1> <n2> ...  - Compare multiple experiments")
    print("  help                   - Show this help")
    print("  quit                   - Exit")
    
    # Initial listing
    explorer.list_experiments(args.model, args.dataset)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower().split()
            if not cmd:
                continue
            
            if cmd[0] == 'quit' or cmd[0] == 'q':
                break
            
            elif cmd[0] == 'help' or cmd[0] == 'h':
                print("\nCommands:")
                print("  list [model] [dataset] - List experiments")
                print("  select <number>        - Select experiment (e.g., '1.2')")
                print("  info                   - Show current experiment details")
                print("  show [n] [incorrect]   - Show n examples")
                print("  steer <alpha> [dir] [n] - Show steering results for alpha value")
                print("  probe                  - Analyze probe training data")
                print("  compare <n1> <n2> ...  - Compare experiments")
                print("  quit                   - Exit")
            
            elif cmd[0] == 'list' or cmd[0] == 'ls':
                model_filter = cmd[1] if len(cmd) > 1 else None
                dataset_filter = cmd[2] if len(cmd) > 2 else None
                explorer.list_experiments(model_filter, dataset_filter)
            
            elif cmd[0] == 'select' or cmd[0] == 'sel':
                if len(cmd) > 1:
                    exp = explorer.select_experiment(cmd[1])
                    if exp:
                        print(f"Selected: {exp['model']} - {exp['dataset']}")
                        explorer.show_experiment_info(exp)
                else:
                    print("Usage: select <number>")
            
            elif cmd[0] == 'info' or cmd[0] == 'i':
                explorer.show_experiment_info()
            
            elif cmd[0] == 'show' or cmd[0] == 's':
                num = 5
                incorrect_only = False
                if len(cmd) > 1:
                    try:
                        num = int(cmd[1])
                    except:
                        if cmd[1] == 'incorrect':
                            incorrect_only = True
                if len(cmd) > 2 and cmd[2] == 'incorrect':
                    incorrect_only = True
                
                explorer.explore_generations(num_examples=num, filter_incorrect=incorrect_only)
            
            elif cmd[0] == 'steer' or cmd[0] == 'st':
                if len(cmd) > 1:
                    try:
                        alpha = float(cmd[1])
                        direction = cmd[2] if len(cmd) > 2 and cmd[2] in ['yes', 'no'] else None
                        num = 5
                        
                        # Check if there's a number at position 2 or 3
                        if len(cmd) > 2 and cmd[2].isdigit():
                            num = int(cmd[2])
                        elif len(cmd) > 3 and cmd[3].isdigit():
                            num = int(cmd[3])
                        
                        explorer.explore_steering_results(alpha=alpha, direction=direction, num_examples=num)
                    except ValueError:
                        print("Usage: steer <alpha> [yes/no] [num_examples]")
                        print("Example: steer 2.0        # Show all directions for alpha=2.0")
                        print("         steer -2 yes 10  # Show 10 'yes' examples for alpha=-2")
                else:
                    print("Usage: steer <alpha> [yes/no] [num_examples]")
            
            elif cmd[0] == 'probe' or cmd[0] == 'p':
                explorer.analyze_probe_training_data()
            
            elif cmd[0] == 'compare' or cmd[0] == 'cmp':
                if len(cmd) > 1:
                    explorer.compare_experiments(cmd[1:])
                else:
                    print("Usage: compare <exp1> <exp2> ...")
            
            else:
                print(f"Unknown command: {cmd[0]}. Type 'help' for commands.")
        
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()