#!/usr/bin/env python3
"""
Interactive experiment playground for post-hoc reasoning experiments.

This notebook breaks down the experiment pipeline into separate, runnable steps:
1. Setup and configuration
2. Data loading and model initialization  
3. Data generation (prompts + activations)
4. Probe training (separate from steering)
5. Steering experiments (separate from probes)
6. Results analysis

This gives you a playground to understand each component and run them independently.
"""

#%% Setup and Imports
import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

print("📦 Importing modules...")
from config import ConfigLoader, ExperimentRunConfig, create_experiment_configs
from cache_manager import ExperimentManager, ExperimentCache
from data_loading import load_all_datasets
from models import ChatModel
from nnsight_models import NNsightChatModel
from parsing_utils import parse_response
from utils import generate_with_hooks
from nnsight_utils import batch_get_resid_activations
from nnsight_steering import generate_with_nnsight_steering

print("✓ All modules imported successfully")

#%% Configuration and Setup
print("📖 Setting up configuration...")

config_path = "configs/mini_gemma_experiment.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

# Load the experiment configuration
run_config = ConfigLoader.load_experiment_config(config_path)

# Apply the equivalent of --no-cache --no-interactive
run_config.cache_dir = "cache"
run_config.use_cache = False    # Set to True if you want caching
run_config.interactive = False

print("✓ Configuration loaded")
print(f"  Models: {[m.name for m in run_config.models]}")
print(f"  Datasets: {[d.name for d in run_config.datasets]}")
print(f"  Alpha range: {run_config.steering.alpha_range}")
print(f"  Use cache: {run_config.use_cache}")

# Create individual experiment configurations
experiment_configs = create_experiment_configs(run_config)
exp_manager = ExperimentManager(run_config.cache_dir)

print(f"✓ Created {len(experiment_configs)} experiment configurations")
for i, config in enumerate(experiment_configs):
    print(f"  {i+1}. {config.model_name} + {config.dataset_name}")

#%% Select and Load Model
print("\n🤖 Model Loading...")

# For this playground, let's work with the first experiment config
exp_config = experiment_configs[0]
print(f"Working with: {exp_config.model_name} + {exp_config.dataset_name}")

# Determine backend and load model
model_name = exp_config.model_name
use_nnsight = (exp_config.model_backend == "nnsight" or 
               (exp_config.model_backend == "auto" and "deepseek" in model_name.lower()))

print(f"🔧 Using {'nnsight' if use_nnsight else 'transformer_lens'} backend")

if use_nnsight:
    model = NNsightChatModel(model_name)
    print(f"✓ Loaded NNsight model: {model_name}")
else:
    model = ChatModel(model_name)
    print(f"✓ Loaded TransformerLens model: {model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden size: {model.cfg.d_model}")

# Initialize cache for this experiment
cache = exp_manager.add_experiment(exp_config)

#%% Load Dataset
print("\n📊 Dataset Loading...")

# Load the specific dataset for our experiment
dataset_name = exp_config.dataset_name
print(f"Loading dataset: {dataset_name}")

datasets = load_all_datasets()
if dataset_name not in datasets:
    raise ValueError(f"Dataset {dataset_name} not found. Available: {list(datasets.keys())}")

dataset = datasets[dataset_name]
print(f"✓ Loaded dataset: {dataset_name}")
print(f"  Total samples: {len(dataset)}")

# Create train/test split
train_size = exp_config.train_size
test_size = exp_config.test_size
split_seed = exp_config.split_seed

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(
    dataset, 
    train_size=train_size, 
    test_size=test_size, 
    random_state=split_seed
)

print(f"✓ Data split: {len(train_data)} train, {len(test_data)} test samples")
print(f"📝 Sample train example:")
sample = train_data[0]
print(f"  Question: {sample['prompt'][1]['content'][:100]}...")
print(f"  Correct answer: {sample['correct_answer']}")

#%% PHASE 1: Data Generation (Prompts + Activations)
print("\n" + "="*60)
print("📝 PHASE 1: GENERATING MODEL DATA")
print("="*60)
print("This phase generates model responses and extracts activations")
print("for both training and test data.")

# Check if we already have cached data
has_cached_data = (run_config.use_cache and 
                   cache.has_generations() and 
                   cache.has_activations())

if has_cached_data:
    print("📁 Using cached data...")
    train_results = cache.load_pickle(cache.get_train_generations_path())
    test_results = cache.load_pickle(cache.get_test_generations_path())
    train_activations = cache.load_pickle(cache.get_train_activations_path())
    test_activations = cache.load_pickle(cache.get_test_activations_path())
    print("✓ Loaded cached data")
else:
    print("🔄 Generating fresh data...")
    
    # Generate model responses for training data
    print(f"\n📝 Generating responses for {len(train_data)} training samples...")
    train_results = []
    for i, item in enumerate(train_data):
        prompt_messages = item["prompt"]
        if use_nnsight:
            prompt_string = model.apply_chat_template(prompt_messages)
            response = model.generate(prompt_string, max_new_tokens=exp_config.max_new_tokens)
        else:
            prompt_string = model.apply_chat_template(prompt_messages)
            response = model.generate(prompt_string, max_new_tokens=exp_config.max_new_tokens)
        
        pred_answer, pred_letter = parse_response(response, thinking=True)
        
        train_results.append({
            "prompt": prompt_messages,
            "response": response,
            "pred_answer": pred_answer,
            "pred_letter": pred_letter,
            "correct_answer": item["correct_answer"],
            "correct_letter": item["correct_letter"]
        })
        
        if (i + 1) % 2 == 0:
            print(f"  ✓ Generated {i + 1}/{len(train_data)} training responses")
    
    # Generate model responses for test data
    print(f"\n📝 Generating responses for {len(test_data)} test samples...")
    test_results = []
    for i, item in enumerate(test_data):
        prompt_messages = item["prompt"]
        if use_nnsight:
            prompt_string = model.apply_chat_template(prompt_messages)
            response = model.generate(prompt_string, max_new_tokens=exp_config.max_new_tokens)
        else:
            prompt_string = model.apply_chat_template(prompt_messages)
            response = model.generate(prompt_string, max_new_tokens=exp_config.max_new_tokens)
        
        pred_answer, pred_letter = parse_response(response, thinking=True)
        
        test_results.append({
            "prompt": prompt_messages,
            "response": response,
            "pred_answer": pred_answer,
            "pred_letter": pred_letter,
            "correct_answer": item["correct_answer"],
            "correct_letter": item["correct_letter"]
        })
        
        if (i + 1) % 2 == 0:
            print(f"  ✓ Generated {i + 1}/{len(test_data)} test responses")
    
    # Extract activations
    print(f"\n🧠 Extracting activations...")
    
    # Prepare prompts for activation extraction
    train_prompts = [model.apply_chat_template(r["prompt"]) for r in train_results]
    test_prompts = [model.apply_chat_template(r["prompt"]) for r in test_results]
    
    if use_nnsight:
        print("  Using NNsight for activation extraction...")
        train_activations = batch_get_resid_activations(train_prompts, model)
        test_activations = batch_get_resid_activations(test_prompts, model)
    else:
        print("  Using TransformerLens for activation extraction...")
        # Get activations using transformer lens
        layers = list(range(model.cfg.n_layers))
        train_tokens = model.to_tokens(train_prompts, prepend_bos=True)
        test_tokens = model.to_tokens(test_prompts, prepend_bos=True)
        
        _, train_cache = model.run_with_cache(train_tokens, pos_slice=-1)
        _, test_cache = model.run_with_cache(test_tokens, pos_slice=-1)
        
        train_activations = {}
        test_activations = {}
        for layer in layers:
            train_activations[layer] = train_cache[f"blocks.{layer}.hook_resid_post"].cpu().numpy()
            test_activations[layer] = test_cache[f"blocks.{layer}.hook_resid_post"].cpu().numpy()
    
    print("✓ Activations extracted")
    
    # Cache the results if enabled
    if run_config.use_cache:
        print("💾 Caching generated data...")
        cache.save_pickle(train_results, cache.get_train_generations_path())
        cache.save_pickle(test_results, cache.get_test_generations_path())
        cache.save_pickle(train_activations, cache.get_train_activations_path())
        cache.save_pickle(test_activations, cache.get_test_activations_path())
        print("✓ Data cached")

# Display generation results
train_accuracy = sum(1 for r in train_results if r["pred_answer"] == r["correct_answer"]) / len(train_results)
test_accuracy = sum(1 for r in test_results if r["pred_answer"] == r["correct_answer"]) / len(test_results)

print(f"\n📊 Generation Results:")
print(f"  Training accuracy: {train_accuracy:.3f} ({sum(1 for r in train_results if r['pred_answer'] == r['correct_answer'])}/{len(train_results)})")
print(f"  Test accuracy: {test_accuracy:.3f} ({sum(1 for r in test_results if r['pred_answer'] == r['correct_answer'])}/{len(test_results)})")

# Show a sample result
print(f"\n📝 Sample result:")
sample_result = train_results[0]
print(f"  Question: {sample_result['prompt'][1]['content'][:100]}...")
print(f"  Model response: {sample_result['response'][:100]}...")
print(f"  Predicted: {sample_result['pred_answer']} | Correct: {sample_result['correct_answer']}")

#%% PHASE 2: Probe Training (Independent Step)
print("\n" + "="*60)
print("🎯 PHASE 2: TRAINING PROBES")
print("="*60)
print("This phase trains binary classifiers (probes) on model activations")
print("to predict the correct answer at each layer.")

# Check if we have cached probes
has_cached_probes = run_config.use_cache and cache.has_probes()

if has_cached_probes:
    print("📁 Using cached probes...")
    probe_results = cache.load_pickle(cache.get_probe_results_path())
    probe_coefficients = cache.load_pickle(cache.get_probe_coefficients_path())
    print("✓ Loaded cached probes")
else:
    print("🔄 Training fresh probes...")
    
    # Prepare training data for probes
    train_labels = [1 if r["correct_answer"] == "yes" else 0 for r in train_results]
    test_labels = [1 if r["correct_answer"] == "yes" else 0 for r in test_results]
    
    print(f"📊 Probe training data:")
    print(f"  Train samples: {len(train_labels)} ({'yes': {sum(train_labels)}, 'no': {len(train_labels) - sum(train_labels)}})")
    print(f"  Test samples: {len(test_labels)} ({'yes': {sum(test_labels)}, 'no': {len(test_labels) - sum(test_labels)}})")
    
    # Get layer information
    if use_nnsight:
        # For nnsight, activations are indexed by layer number
        layers = sorted(train_activations.keys())
    else:
        layers = list(range(model.cfg.n_layers))
    
    print(f"  Training probes for {len(layers)} layers: {layers}")
    
    probe_results = {}
    probe_coefficients = {}
    
    for layer in layers:
        print(f"\n🔍 Training probe for layer {layer}...")
        
        # Get activations for this layer
        X_train = train_activations[layer].squeeze()  # Remove batch dimension if present
        X_test = test_activations[layer].squeeze()
        
        # Handle dimension mismatches
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        print(f"  Activation shape: {X_train.shape}")
        
        # Train logistic regression probe
        probe = LogisticRegression(random_state=42, max_iter=1000)
        probe.fit(X_train, train_labels)
        
        # Evaluate probe
        train_probs = probe.predict_proba(X_train)[:, 1]
        test_probs = probe.predict_proba(X_test)[:, 1]
        
        train_auc = roc_auc_score(train_labels, train_probs)
        test_auc = roc_auc_score(test_labels, test_probs)
        
        probe_results[layer] = {
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_accuracy": (probe.predict(X_train) == train_labels).mean(),
            "test_accuracy": (probe.predict(X_test) == test_labels).mean()
        }
        
        # Store probe coefficients (these will be used for steering)
        probe_coefficients[layer] = probe.coef_[0]
        
        print(f"  ✓ Layer {layer}: Train AUC={train_auc:.3f}, Test AUC={test_auc:.3f}")
    
    # Cache the results if enabled
    if run_config.use_cache:
        print("\n💾 Caching probe results...")
        cache.save_pickle(probe_results, cache.get_probe_results_path())
        cache.save_pickle(probe_coefficients, cache.get_probe_coefficients_path())
        print("✓ Probes cached")

# Display probe training results
print(f"\n📊 Probe Training Summary:")
print("Layer | Train AUC | Test AUC | Train Acc | Test Acc")
print("-" * 50)
for layer in sorted(probe_results.keys()):
    results = probe_results[layer]
    print(f"{layer:5d} | {results['train_auc']:9.3f} | {results['test_auc']:8.3f} | {results['train_accuracy']:9.3f} | {results['test_accuracy']:8.3f}")

# Find best performing layer
best_layer = max(probe_results.keys(), key=lambda l: probe_results[l]['test_auc'])
best_auc = probe_results[best_layer]['test_auc']
print(f"\n🏆 Best performing layer: {best_layer} (Test AUC: {best_auc:.3f})")

#%% PHASE 3: Steering Experiments (Independent Step)
print("\n" + "="*60)
print("🎮 PHASE 3: STEERING EXPERIMENTS")
print("="*60)
print("This phase uses the trained probe coefficients to steer model")
print("behavior by adding activation vectors during generation.")

# Get steering parameters
alpha_range = exp_config.alpha_range
print(f"🎯 Testing alpha values: {alpha_range}")

# Create test subsets for steering
print("\n📊 Preparing steering test data...")
yes_test_data = [r for r in test_results if r["pred_answer"] == "yes" and r["correct_answer"] == "yes"]
no_test_data = [r for r in test_results if r["pred_answer"] == "no" and r["correct_answer"] == "no"]

print(f"  'Yes' examples (to steer to 'No'): {len(yes_test_data)}")
print(f"  'No' examples (to steer to 'Yes'): {len(no_test_data)}")

if len(yes_test_data) == 0 or len(no_test_data) == 0:
    print("⚠️  Warning: No examples available for steering in one direction")
    print("    This means the model got everything wrong or right on the test set")

# Steering results storage
steering_results = {}

for alpha in alpha_range:
    print(f"\n🎮 Testing steering with alpha = {alpha}")
    
    # Alpha for steering "yes" to "no" (negative)
    alpha_yes_to_no = -abs(alpha)
    # Alpha for steering "no" to "yes" (positive)  
    alpha_no_to_yes = abs(alpha)
    
    steering_results[alpha] = {"yes_to_no": [], "no_to_yes": []}
    
    # Steer "yes" answers to "no"
    if yes_test_data:
        print(f"  🔄 Steering {len(yes_test_data)} 'yes' examples to 'no' (alpha={alpha_yes_to_no})")
        
        for i, result in enumerate(yes_test_data[:3]):  # Limit for demo
            prompt_string = model.apply_chat_template(result["prompt"])
            
            if use_nnsight:
                # Use nnsight steering
                layers_to_steer = sorted(probe_coefficients.keys())
                steered_response = generate_with_nnsight_steering(
                    model, prompt_string, probe_coefficients, layers_to_steer, 
                    alpha_yes_to_no, max_new_tokens=exp_config.max_new_tokens
                )
            else:
                # Use transformer lens steering
                layers_to_steer = list(range(model.cfg.n_layers))
                steered_response = generate_with_hooks(
                    model, prompt_string, probe_coefficients, layers_to_steer,
                    alpha_yes_to_no, max_new_tokens=exp_config.max_new_tokens
                )
            
            steered_answer, steered_letter = parse_response(steered_response, thinking=True)
            success = steered_answer == "no"
            
            steering_results[alpha]["yes_to_no"].append({
                "original_answer": result["pred_answer"],
                "steered_answer": steered_answer,
                "target_answer": "no",
                "success": success,
                "original_response": result["response"][:100] + "...",
                "steered_response": steered_response[:100] + "..."
            })
            
            if i < 2:  # Show first few examples
                print(f"    Example {i+1}: '{result['pred_answer']}' → '{steered_answer}' {'✓' if success else '✗'}")
    
    # Steer "no" answers to "yes"
    if no_test_data:
        print(f"  🔄 Steering {len(no_test_data)} 'no' examples to 'yes' (alpha={alpha_no_to_yes})")
        
        for i, result in enumerate(no_test_data[:3]):  # Limit for demo
            prompt_string = model.apply_chat_template(result["prompt"])
            
            if use_nnsight:
                # Use nnsight steering
                layers_to_steer = sorted(probe_coefficients.keys())
                steered_response = generate_with_nnsight_steering(
                    model, prompt_string, probe_coefficients, layers_to_steer,
                    alpha_no_to_yes, max_new_tokens=exp_config.max_new_tokens
                )
            else:
                # Use transformer lens steering
                layers_to_steer = list(range(model.cfg.n_layers))
                steered_response = generate_with_hooks(
                    model, prompt_string, probe_coefficients, layers_to_steer,
                    alpha_no_to_yes, max_new_tokens=exp_config.max_new_tokens
                )
            
            steered_answer, steered_letter = parse_response(steered_response, thinking=True)
            success = steered_answer == "yes"
            
            steering_results[alpha]["no_to_yes"].append({
                "original_answer": result["pred_answer"],
                "steered_answer": steered_answer,
                "target_answer": "yes",
                "success": success,
                "original_response": result["response"][:100] + "...",
                "steered_response": steered_response[:100] + "..."
            })
            
            if i < 2:  # Show first few examples  
                print(f"    Example {i+1}: '{result['pred_answer']}' → '{steered_answer}' {'✓' if success else '✗'}")

# Calculate steering success rates
print(f"\n📊 Steering Results Summary:")
print("Alpha | Yes→No Success | No→Yes Success | Overall Success")
print("-" * 55)

for alpha in alpha_range:
    yes_to_no_results = steering_results[alpha]["yes_to_no"]
    no_to_yes_results = steering_results[alpha]["no_to_yes"]
    
    yes_to_no_success = sum(r["success"] for r in yes_to_no_results) / len(yes_to_no_results) if yes_to_no_results else 0
    no_to_yes_success = sum(r["success"] for r in no_to_yes_results) / len(no_to_yes_results) if no_to_yes_results else 0
    
    all_results = yes_to_no_results + no_to_yes_results
    overall_success = sum(r["success"] for r in all_results) / len(all_results) if all_results else 0
    
    print(f"{alpha:5.1f} | {yes_to_no_success:13.1%} | {no_to_yes_success:13.1%} | {overall_success:13.1%}")

#%% Results Analysis and Exploration
print("\n" + "="*60)
print("📈 RESULTS ANALYSIS")
print("="*60)
print("Explore the results from your experiment")

# Create summary dataframe
summary_data = {
    "model": [exp_config.model_name],
    "dataset": [exp_config.dataset_name],
    "train_size": [len(train_data)],
    "test_size": [len(test_data)],
    "model_train_accuracy": [train_accuracy],
    "model_test_accuracy": [test_accuracy],
    "best_probe_layer": [best_layer],
    "best_probe_auc": [best_auc],
}

# Add steering results
for alpha in alpha_range:
    yes_to_no_results = steering_results[alpha]["yes_to_no"]
    no_to_yes_results = steering_results[alpha]["no_to_yes"]
    all_results = yes_to_no_results + no_to_yes_results
    overall_success = sum(r["success"] for r in all_results) / len(all_results) if all_results else 0
    summary_data[f"steering_alpha_{alpha}_success"] = [overall_success]

summary_df = pd.DataFrame(summary_data)

print("📊 Experiment Summary:")
for col, val in summary_data.items():
    print(f"  {col}: {val[0]}")

print(f"\n🎯 Available variables for exploration:")
print(f"  - train_data, test_data: Original dataset splits")
print(f"  - train_results, test_results: Model generations + parsed answers")
print(f"  - train_activations, test_activations: Model internal activations")
print(f"  - probe_results: Performance of probes at each layer")
print(f"  - probe_coefficients: Learned probe weights (steering vectors)")
print(f"  - steering_results: Results of steering experiments")
print(f"  - summary_df: High-level summary dataframe")

#%% Optional: Save and Export Results
print("\n💾 Optional: Save Results")
print("You can save any of the results to files:")
print("# Example saves:")
print("# summary_df.to_csv('experiment_summary.csv', index=False)")
print("# import pickle")
print("# with open('probe_results.pkl', 'wb') as f:")
print("#     pickle.dump(probe_results, f)")
print("# with open('steering_results.pkl', 'wb') as f:")
print("#     pickle.dump(steering_results, f)")

print("\n✅ Experiment playground complete!")
print("🎮 You can now:")
print("  - Modify alpha values and re-run steering")
print("  - Try different layers for steering")  
print("  - Analyze individual examples")
print("  - Experiment with probe training parameters")

# Clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()