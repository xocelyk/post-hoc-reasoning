#!/usr/bin/env python3
"""
Focused notebook for testing CAA Single-Layer steering on Anachronisms dataset.

Configuration:
- Model: Gemma-2-9b-it  
- Dataset: Anachronisms
- Train size: 200
- Steering method: CAA Single-Layer
- Optimized to use existing cached generations

Usage:
    python test_caa_anachronisms.py

Requirements:
    - Existing cached generations from main experiment
    - Either cached activations OR the notebook will guide you
"""

#%% Setup and Imports
import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

print("📦 Loading modules...")
from config import ExperimentConfig
from cache_manager import ExperimentCache
from data_loading import create_cot_dataset, create_dataset
from models import ChatModel
from nnsight_models import NNsightChatModel
from parsing_utils import parse_response
from utils import generate_with_hooks
from nnsight_utils import batch_get_resid_activations
from nnsight_steering import generate_with_nnsight_steering
from steering_methods import CAASingleLayerSteering
from sklearn.metrics import roc_auc_score

print("✅ Modules loaded")

#%% Configuration
print("\n" + "="*60)
print("🎯 EXPERIMENT CONFIGURATION")
print("="*60)

# Fixed configuration for this specific test
MODEL_NAME = "google/gemma-2-9b-it"
DATASET_NAME = "anachronisms"
TRAIN_SIZE = 200
TEST_SIZE = 100
SPLIT_SEED = 42
CACHE_DIR = "cache"

# Steering parameters
ALPHA_RANGE = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 100

print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Train size: {TRAIN_SIZE}")
print(f"Test size: {TEST_SIZE}")
print(f"Steering method: CAA Single-Layer")
print(f"Alpha range: {ALPHA_RANGE}")

#%% Initialize Cache Manager
print("\n📁 Setting up cache manager...")

# Create experiment config
exp_config = ExperimentConfig(
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    split_seed=SPLIT_SEED,
    alpha_range=ALPHA_RANGE,
    temperature=TEMPERATURE,
    max_new_tokens=MAX_NEW_TOKENS
)

# Set steering method for cache key
exp_config.steering_method = "caa-single-layer"

# Initialize cache
cache = ExperimentCache(exp_config, CACHE_DIR)
print(f"✅ Cache initialized at: {cache.cache_dir}")

# Check cache status
print("\n📊 Checking cache status...")
cache_status = cache.get_experiment_status()

print("\nCache contents:")
for component, is_cached in cache_status.items():
    status = "✅ Available" if is_cached else "❌ Not found"
    print(f"  {component}: {status}")

#%% Load Cached Generations
print("\n" + "="*60)
print("📥 LOADING CACHED DATA")
print("="*60)

# Check what we have cached
has_generations = cache.has_generations()
has_activations = cache.has_activations()

if not has_generations:
    print("❌ No cached generations found!")
    print("   You need to run the main experiment first to generate data")
    print("   Run: python run_experiments.py configs/your_config.yaml")
    sys.exit(1)

print("✅ Found cached generations, loading...")

# Load train/test generations
train_results = cache.load_pickle(cache.get_train_generations_path())
test_results = cache.load_pickle(cache.get_test_generations_path())

print(f"  Loaded {len(train_results)} training examples")
print(f"  Loaded {len(test_results)} test examples")

# Show sample
if train_results:
    sample = train_results[0]
    print(f"\n📝 Sample generation:")
    print(f"  Question: {sample['prompt'][1]['content'][:80]}...")
    print(f"  Response: {sample['response'][:80]}...")
    print(f"  Parsed answer: {sample['pred_answer']}")
    print(f"  Correct answer: {sample['correct_answer']}")

# Calculate accuracies
train_acc = sum(1 for r in train_results if r['pred_answer'] == r['correct_answer']) / len(train_results)
test_acc = sum(1 for r in test_results if r['pred_answer'] == r['correct_answer']) / len(test_results)
print(f"\n📊 Model Performance:")
print(f"  Train accuracy: {train_acc:.1%}")
print(f"  Test accuracy: {test_acc:.1%}")

#%% Load or Compute Activations
print("\n" + "="*60)
print("🧠 ACTIVATIONS")
print("="*60)

if has_activations:
    print("✅ Found cached activations, loading...")
    train_activations = cache.load_pickle(cache.get_train_activations_path())
    test_activations = cache.load_pickle(cache.get_test_activations_path())
    print(f"  Loaded activations for {len(train_activations)} layers")
else:
    print("❌ No cached activations found!")
    print("\n💡 To get activations, you have two options:")
    print("   1. Run the full experiment: python run_experiments.py configs/your_config.yaml")
    print("   2. Use the main notebook: python run_experiment_notebook.py")
    print("\n   The full experiment will generate and cache everything you need.")
    print("   This focused notebook is designed to work with existing cache data.")
    
    # Check if we can find any cache directories to help the user
    base_cache = "cache/experiments"
    if os.path.exists(base_cache):
        print(f"\n🔍 Found cache directories:")
        for model_dir in os.listdir(base_cache):
            if "gemma" in model_dir.lower():
                model_path = os.path.join(base_cache, model_dir)
                if os.path.isdir(model_path):
                    for dataset_dir in os.listdir(model_path):
                        if dataset_dir == "anachronisms":
                            print(f"   ✅ {model_dir}/{dataset_dir}")
                            dataset_path = os.path.join(model_path, dataset_dir)
                            for split_dir in os.listdir(dataset_path):
                                split_path = os.path.join(dataset_path, split_dir)
                                if os.path.isdir(split_path):
                                    for exp_dir in os.listdir(split_path):
                                        exp_path = os.path.join(split_path, exp_dir)
                                        data_path = os.path.join(exp_path, "data")
                                        if os.path.exists(data_path):
                                            files = os.listdir(data_path)
                                            print(f"      {split_dir}/{exp_dir}: {files}")
    
    print("\n❌ Exiting - need cached activations to proceed")
    sys.exit(1)

#%% Compute CAA Steering Vectors
print("\n" + "="*60)
print("🎯 COMPUTING CAA STEERING VECTORS")
print("="*60)

# Check for cached steering vectors
method = "caa-single-layer"
probe_coefficients_path = cache.get_probe_coefficients_path(method)
auc_scores_path = cache.get_auc_scores_path(method)

USE_CACHED_VECTORS = True  # Set to False to recompute

if USE_CACHED_VECTORS and os.path.exists(probe_coefficients_path):
    print("✅ Found cached CAA steering vectors, loading...")
    probe_coefficients = cache.load_pickle(probe_coefficients_path)
    
    if os.path.exists(auc_scores_path):
        auc_data = cache.load_json(auc_scores_path)
        if isinstance(auc_data, dict):
            similarity_scores = {int(k): v for k, v in auc_data.items()}
        else:
            similarity_scores = {i: score for i, score in enumerate(auc_data)}
    
    layers = sorted(probe_coefficients.keys())
    print(f"  Loaded steering vectors for {len(layers)} layers")
    
    # Find best layer
    best_layer = max(similarity_scores.keys(), key=lambda k: similarity_scores[k])
    print(f"  Best layer: {best_layer} (similarity score: {similarity_scores[best_layer]:.3f})")
    
else:
    print("🔄 Computing fresh CAA steering vectors...")
    
    # Prepare labels
    train_labels = [1 if r["correct_answer"] == "yes" else 0 for r in train_results]
    test_labels = [1 if r["correct_answer"] == "yes" else 0 for r in test_results]
    
    print(f"\n📊 Label distribution:")
    print(f"  Train: {sum(train_labels)} yes, {len(train_labels)-sum(train_labels)} no")
    print(f"  Test: {sum(test_labels)} yes, {len(test_labels)-sum(test_labels)} no")
    
    # Get layers
    layers = sorted(train_activations.keys())
    print(f"\n🔬 Computing contrastive vectors for {len(layers)} layers...")
    
    contrastive_vectors = {}
    similarity_scores = {}
    
    for layer in layers:
        # Get activations
        X_train = train_activations[layer].squeeze()
        X_test = test_activations[layer].squeeze()
        
        # Reshape if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Compute contrastive vector: mean(yes) - mean(no)
        train_labels_array = np.array(train_labels)
        yes_mask = train_labels_array == 1
        no_mask = train_labels_array == 0
        
        mean_yes = np.mean(X_train[yes_mask], axis=0)
        mean_no = np.mean(X_train[no_mask], axis=0)
        
        # Raw difference vector (no normalization for CAA)
        contrastive_vectors[layer] = mean_yes - mean_no
        
        # Compute similarity scores for evaluation
        test_labels_array = np.array(test_labels)
        similarities = X_test @ contrastive_vectors[layer]
        auc_score = roc_auc_score(test_labels_array, similarities)
        similarity_scores[layer] = auc_score
        
        if layer % 5 == 0:  # Print every 5th layer
            print(f"  Layer {layer}: similarity AUC = {auc_score:.3f}")
    
    # Apply CAA single-layer method
    print("\n🎯 Applying CAA Single-Layer selection...")
    steering_method = CAASingleLayerSteering(list(similarity_scores.values()))
    layer_vectors = [contrastive_vectors[layer] for layer in layers]
    steering_vectors = steering_method.compute_steering_vectors(layer_vectors)
    
    # Convert to dictionary format
    probe_coefficients = {}
    for i, layer in enumerate(layers):
        probe_coefficients[layer] = steering_vectors[i]
    
    # Find best layer
    best_layer = layers[np.argmax(list(similarity_scores.values()))]
    print(f"\n✅ Selected layer {best_layer} with similarity score {similarity_scores[best_layer]:.3f}")
    
    # Save to cache
    print("\n💾 Saving steering vectors to cache...")
    cache.save_pickle(probe_coefficients, probe_coefficients_path)
    cache.save_json({str(k): v for k, v in similarity_scores.items()}, auc_scores_path)
    print("✅ Saved!")

#%% Prepare for Steering Experiments
print("\n" + "="*60)
print("🎮 STEERING EXPERIMENTS")
print("="*60)

# Filter test data for steering
yes_test = [r for r in test_results if r["pred_answer"] == "yes" and r["correct_answer"] == "yes"]
no_test = [r for r in test_results if r["pred_answer"] == "no" and r["correct_answer"] == "no"]

print(f"Test examples available for steering:")
print(f"  Yes → No: {len(yes_test)} examples")
print(f"  No → Yes: {len(no_test)} examples")

if len(yes_test) == 0 or len(no_test) == 0:
    print("\n⚠️  Warning: Not enough examples for bidirectional steering")
    print("   The model may have gotten everything wrong/right on the test set")

# Limit examples for faster testing
MAX_EXAMPLES = 5  # Adjust this for more thorough testing
yes_test = yes_test[:MAX_EXAMPLES]
no_test = no_test[:MAX_EXAMPLES]

print(f"\n🎯 Testing with {MAX_EXAMPLES} examples per direction")

#%% Run Steering Tests
print("\n🔄 Starting steering experiments...")

# Initialize model if not already loaded
if 'model' not in locals():
    print("\n🤖 Loading model for steering...")
    try:
        from nnsight import LanguageModel
        model = NNsightChatModel(MODEL_NAME)
        use_nnsight = True
        print("✅ Using nnsight backend")
    except:
        import transformer_lens as tl
        tl_model = tl.HookedTransformer.from_pretrained(MODEL_NAME)
        model = ChatModel(tl_model, MODEL_NAME)
        use_nnsight = False
        print("✅ Using transformer_lens backend")

# Prepare steering vectors array
layers_to_steer = sorted(probe_coefficients.keys())
steering_vectors_array = np.array([probe_coefficients[layer] for layer in layers_to_steer])

steering_results = {}

for alpha in ALPHA_RANGE:
    print(f"\n📊 Testing alpha = {alpha}")
    steering_results[alpha] = {"yes_to_no": [], "no_to_yes": []}
    
    # Test Yes → No steering
    if yes_test:
        print(f"  Testing Yes → No (alpha = {-alpha})")
        for i, example in enumerate(yes_test):
            # Apply chat template
            prompt_string = model.apply_chat_template(example["prompt"])
            
            # Tokenize
            tokens = model.to_tokens(prompt_string, prepend_bos=False)
            
            # Generate with steering
            if use_nnsight:
                steered_response = generate_with_nnsight_steering(
                    model=model,
                    tokens=tokens,
                    steering_vectors=steering_vectors_array,
                    alpha=-alpha,  # Negative for yes->no
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    layers=layers_to_steer
                )
            else:
                steered_response = generate_with_hooks(
                    model=model,
                    tokens=tokens,
                    steering_vectors=steering_vectors_array,
                    alpha=-alpha,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    layers=layers_to_steer,
                    verbose=False
                )
            
            # Parse response
            steered_letter, steered_answer = parse_response(steered_response, thinking=True)
            success = steered_answer == "no"
            
            steering_results[alpha]["yes_to_no"].append({
                "original": "yes",
                "steered": steered_answer,
                "success": success
            })
            
            if i == 0:  # Show first example
                print(f"    Example: yes → {steered_answer} {'✅' if success else '❌'}")
    
    # Test No → Yes steering  
    if no_test:
        print(f"  Testing No → Yes (alpha = {alpha})")
        for i, example in enumerate(no_test):
            # Apply chat template
            prompt_string = model.apply_chat_template(example["prompt"])
            
            # Tokenize
            tokens = model.to_tokens(prompt_string, prepend_bos=False)
            
            # Generate with steering
            if use_nnsight:
                steered_response = generate_with_nnsight_steering(
                    model=model,
                    tokens=tokens,
                    steering_vectors=steering_vectors_array,
                    alpha=alpha,  # Positive for no->yes
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    layers=layers_to_steer
                )
            else:
                steered_response = generate_with_hooks(
                    model=model,
                    tokens=tokens,
                    steering_vectors=steering_vectors_array,
                    alpha=alpha,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    layers=layers_to_steer,
                    verbose=False
                )
            
            # Parse response
            steered_letter, steered_answer = parse_response(steered_response, thinking=True)
            success = steered_answer == "yes"
            
            steering_results[alpha]["no_to_yes"].append({
                "original": "no",
                "steered": steered_answer,
                "success": success
            })
            
            if i == 0:  # Show first example
                print(f"    Example: no → {steered_answer} {'✅' if success else '❌'}")

#%% Results Summary
print("\n" + "="*60)
print("📈 RESULTS SUMMARY")
print("="*60)

print("\nSteering Success Rates:")
print("Alpha | Yes→No | No→Yes | Overall")
print("-" * 40)

for alpha in ALPHA_RANGE:
    yes_to_no = steering_results[alpha]["yes_to_no"]
    no_to_yes = steering_results[alpha]["no_to_yes"]
    
    yes_to_no_rate = sum(r["success"] for r in yes_to_no) / len(yes_to_no) if yes_to_no else 0
    no_to_yes_rate = sum(r["success"] for r in no_to_yes) / len(no_to_yes) if no_to_yes else 0
    
    all_results = yes_to_no + no_to_yes
    overall_rate = sum(r["success"] for r in all_results) / len(all_results) if all_results else 0
    
    print(f"{alpha:5.2f} | {yes_to_no_rate:5.1%} | {no_to_yes_rate:5.1%} | {overall_rate:6.1%}")

# Find best alpha
best_alpha = max(ALPHA_RANGE, key=lambda a: sum(r["success"] for r in steering_results[a]["yes_to_no"] + steering_results[a]["no_to_yes"]) / max(1, len(steering_results[a]["yes_to_no"] + steering_results[a]["no_to_yes"])))
print(f"\n🏆 Best alpha: {best_alpha}")

print("\n✅ Experiment complete!")
print(f"\n💾 Results available in steering_results dictionary")
print("   You can save these for further analysis")