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
from utils import generate_with_steering as generate_with_steering_utils
from nnsight_utils import extract_activations, generate_with_steering, ProbeResult, train_probes, batch_generate_text
from steering_methods import (
    CAASingleLayerSteering, 
    CAALayerIncrementalSteering, 
    LogisticRegressionSteering
)

print("✓ All modules imported successfully")

#%% Configuration and Setup
print("📖 Setting up configuration...")

config_path = "configs/transformer_lens_test.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

# Steering method selection
# Options: "logistic-regression", "caa-single-layer", "caa-layer-incremental"
STEERING_METHOD = "caa-single-layer"

# Cache control - set to False to force recomputation even if cache exists
USE_CACHED_DATASET = True  # Set to False to recompute dataset
USE_CACHED_GENERATIONS = True  # Set to False to recompute generations
USE_CACHED_ACTIVATIONS = False  # Set to False to recompute activations
USE_CACHED_PROBES = False  # Set to False to recompute steering vectors  

print("\n" + "="*60)
print("🎯 STEERING METHOD CONFIGURATION")
print("="*60)
print(f"Selected method: {STEERING_METHOD}")
print(f"Use cached probes: {USE_CACHED_PROBES}")
print("\nAvailable methods:")
print("  - logistic-regression: Traditional probe-based steering using sklearn")
print("  - caa-single-layer: CAA method selecting best single layer")
print("  - caa-layer-incremental: CAA method with incremental layer differences")
print("\nTo change method, modify STEERING_METHOD variable above")
print("To force recomputation, set USE_CACHED_PROBES = False")
print("="*60 + "\n")

# Load the experiment configuration
run_config = ConfigLoader.load_experiment_config(config_path)

# Configure caching behavior
run_config.cache_dir = "cache"
run_config.use_cache = True    # Enable caching for probe results
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

# Get the backend from the original run config for this model
model_config = next((m for m in run_config.models if m.name == model_name), None)
if model_config is None:
    raise ValueError(f"Model {model_name} not found in run config")

backend = model_config.backend
use_nnsight = (backend == "nnsight" or 
               (backend == "auto" and "deepseek" in model_name.lower()))

print(f"🔧 Using {'nnsight' if use_nnsight else 'transformer_lens'} backend")

# %%

if use_nnsight:
    # Pass device and dtype configuration to avoid meta tensor issues
    model = NNsightChatModel(
        model_name,
        device_map=model_config.device,
        dtype=model_config.dtype
    )
    print(f"✓ Loaded NNsight model: {model_name}")
else:
    model = ChatModel(model_name)
    print(f"✓ Loaded TransformerLens model: {model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden size: {model.cfg.d_model}")

#%%

# Initialize cache for this experiment
cache = exp_manager.add_experiment(exp_config)
print(f"✓ Cache manager initialized")
print(f"  Cache directory: {cache.cache_dir}")

# Show cache status
print("\n💾 Cache Status:")
cache_status = cache.get_experiment_status()
for component, is_cached in cache_status.items():
    status_icon = "✅" if is_cached else "❌"
    print(f"  {status_icon} {component}: {'Cached' if is_cached else 'Not cached'}")


#%%

# Show available probe methods
cached_methods = cache.list_cached_probe_methods()
if cached_methods:
    print(f"\n🎯 Cached probe methods: {', '.join(cached_methods)}")
    if STEERING_METHOD in cached_methods:
        print(f"  ✅ {STEERING_METHOD} is cached and ready to use")
    else:
        print(f"  ❌ {STEERING_METHOD} not cached - will compute fresh")

if cache_status['dataset'] and cache_status['train_test_split']:
    print("\n💡 Tip: Dataset and splits are cached. Phase 1 will load from cache.")
if cache_status['generations'] and cache_status['activations']:
    print("💡 Tip: Generations and activations are cached. You can skip to Phase 2.")

#%% Load Dataset
print("\n📊 Dataset Loading...")

# Load the specific dataset for our experiment
dataset_name = exp_config.dataset_name
print(f"Loading dataset: {dataset_name}")

if USE_CACHED_DATASET and cache_status['dataset']:
    print("📁 Using cached dataset...")
    dataset = cache.load_pickle(cache.get_dataset_path())
else:
    print("🔄 Loading dataset...")
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
# sample = train_data[0]
# print(f"  Question: {sample['prompt'][1]['content'][:100]}...")
# print(f"  Correct answer: {sample['correct_answer']}")

#%% PHASE 1: Data Generation (Prompts + Activations)
print("\n" + "="*60)
print("📝 PHASE 1: GENERATING MODEL DATA")
print("="*60)
print("This phase generates model responses and extracts activations")
print("for both training and test data.")

# Check if we already have cached data (disabled with --no-cache equivalent)
if USE_CACHED_GENERATIONS and cache_status['generations']:
    print("📁 Using cached data...")
    train_results = cache.load_pickle(cache.get_train_generations_path())
    test_results = cache.load_pickle(cache.get_test_generations_path())
    
    train_activations = cache.load_pickle(cache.get_train_activations_path())
    test_activations = cache.load_pickle(cache.get_test_activations_path())
else:
    print("🔄 Generating fresh data...")
    
    def generate_batch_responses(
        data: list[dict],
        data_type: str = "train",
        batch_size: int = 4,
    ):
        """
        Simple batched generation that relies on HookedTransformer.generate
        to handle all tokenisation, padding, and decoding.

        Each item in `data` must contain:
            • "prompt"          – list[dict]  (chat messages)
            • "correct_answer"
            • "correct_letter"
        """
        batch_size = 1
        print(
            f"\n📝 Generating responses for {len(data)} {data_type} samples "
            f"(batch_size={batch_size})..."
        )

        results = []

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            # 1) build prompt strings
            prompt_strs, metas = [], []
            for item in batch:
                prompt_strs.append(model.apply_chat_template(item["prompt"]))
                metas.append(
                    {
                        "prompt": item["prompt"],
                        "correct_answer": item["correct_answer"],
                        "correct_letter": item["correct_letter"],
                    }
                )

            # 2) batched generation
            if use_nnsight:
                # Use proper batch generation for nnsight
                responses = batch_generate_text(
                    model=model,
                    prompts=prompt_strs,
                    max_new_tokens=exp_config.max_new_tokens,
                    temperature=exp_config.temperature,
                    do_sample=True
                )
            else:
                # TransformerLens handles batch generation directly
                responses = model.generate(
                    prompt_strs,
                    max_new_tokens=exp_config.max_new_tokens,
                    temperature=exp_config.temperature,
                    do_sample=True,
                )
            # generate() returns a single str if batch_size==1
            if isinstance(responses, str):
                responses = [responses]

            # 3) parse + collect results
            for meta, resp in zip(metas, responses):
                pred_let, pred_ans = parse_response(resp, thinking=True)
                results.append(
                    {
                        "prompt": meta["prompt"],
                        "response": resp,
                        "pred_answer": pred_ans,
                        "pred_letter": pred_let,
                        "correct_answer": meta["correct_answer"],
                        "correct_letter": meta["correct_letter"],
                    }
                )

            print(
                f"  ✓ Generated {min(i + batch_size, len(data))}/{len(data)} "
                f"{data_type} responses"
            )

        return results



    
    # Set batch size based on available memory and model size
    # BATCH_SIZE = 2 if use_nnsight else 4  # NNsight uses more memory, smaller batches
    BATCH_SIZE = 1
    
    # Generate responses in batches
    train_results = generate_batch_responses(train_data, "train", BATCH_SIZE)
    test_results = generate_batch_responses(test_data, "test", BATCH_SIZE)
    
    # Extract activations
    print(f"\n🧠 Extracting activations...")
    
    if USE_CACHED_ACTIVATIONS and cache_status['activations']:
        print("📁 Using cached activations...")
        train_activations = cache.load_pickle(cache.get_train_activations_path())
        test_activations = cache.load_pickle(cache.get_test_activations_path())
    else:
        print("🔄 Extracting activations...")
    
        # Prepare prompts for activation extraction
        train_prompts = [model.apply_chat_template(r["prompt"]) for r in train_results]
        test_prompts = [model.apply_chat_template(r["prompt"]) for r in test_results]
        
        if use_nnsight:
            print("  Using NNsight for activation extraction...")
            train_activations = extract_activations(model, train_prompts)
            test_activations = extract_activations(model, test_prompts)
        else:
            print("  Using TransformerLens for activation extraction...")
            # Get activations using transformer lens with batched processing
            layers = list(range(model.cfg.n_layers))
            
            # Process in smaller batches to avoid OOM
            def get_activations_batched(prompts, model, batch_size=1):
                """Extract activations in batches to avoid OOM"""
                all_activations = {layer: [] for layer in layers}
                
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i+batch_size]
                    print(f"    Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
                    
                    with torch.no_grad():
                        tokens = model.to_tokens(batch_prompts, prepend_bos=False)
                        _, cache = model.run_with_cache(tokens, pos_slice=-1)
                        
                        for layer in layers:
                            layer_acts = cache[f"blocks.{layer}.hook_resid_post"].cpu().float().numpy()
                            all_activations[layer].append(layer_acts)
                        
                        # Clean up immediately
                        del cache, tokens
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                
                # Concatenate all batches
                for layer in layers:
                    all_activations[layer] = np.vstack(all_activations[layer])
                
                return all_activations
            
            print("  Processing training activations...")
            train_activations = get_activations_batched(train_prompts, model, batch_size=1)
            
            print("  Processing test activations...")
            test_activations = get_activations_batched(test_prompts, model, batch_size=1)
        
        print("✓ Activations extracted")
    
    # Cache the results
    print("\n💾 Caching extracted activations...")
    cache.save_pickle(train_activations, cache.get_train_activations_path())
    cache.save_pickle(test_activations, cache.get_test_activations_path())
    print("✓ Activations cached")
    
#%%
# new_train_results = []
# for result in train_results:
#     pred_letter, pred_answer = parse_response(result["response"], thinking=True)
#     result["pred_answer"] = pred_answer
#     result["pred_letter"] = pred_letter
#     print(result["pred_answer"], result["correct_answer"])
#     new_train_results.append(result)

# new_test_results = []
# for result in test_results:
#     pred_letter, pred_answer = parse_response(result["response"], thinking=True)
#     result["pred_answer"] = pred_answer
#     result["pred_letter"] = pred_letter
#     new_test_results.append(result)

# train_results = new_train_results
# test_results = new_test_results

#%%
# Display generation results
train_accuracy = sum(1 for r in train_results if r["pred_answer"] == r["correct_answer"]) / len(train_results)
test_accuracy = sum(1 for r in test_results if r["pred_answer"] == r["correct_answer"]) / len(test_results)

print(f"\n📊 Generation Results:")
print(f"  Training accuracy: {train_accuracy:.3f} ({sum(1 for r in train_results if r['pred_answer'] == r['correct_answer'])}/{len(train_results)})")
print(f"  Test accuracy: {test_accuracy:.3f} ({sum(1 for r in test_results if r['pred_answer'] == r['correct_answer'])}/{len(test_results)})")

# # Show a sample result
# print(f"\n📝 Sample result:")
# sample_result = train_results[0]
# print(f"  Question: {sample_result['prompt'][1]['content'][:100]}...")
# print(f"  Model response: {sample_result['response'][:100]}...")
# print(f"  Predicted: {sample_result['pred_answer']} | Correct: {sample_result['correct_answer']}")

#%% PHASE 2: Probe Training and Steering Vector Computation
print("\n" + "="*60)
print("🎯 PHASE 2: TRAINING PROBES AND COMPUTING STEERING VECTORS")
print("="*60)
print(f"Using steering method: {STEERING_METHOD}")
if STEERING_METHOD == "logistic-regression":
    print("This phase trains binary classifiers (probes) on model activations")
    print("and uses the coefficients as steering vectors.")
else:
    print("This phase computes contrastive activation vectors")
    print("and applies the selected CAA steering method.")

# Check if we have cached probes/steering vectors for this method
probe_coefficients_path = cache.get_probe_coefficients_path(STEERING_METHOD)
auc_scores_path = cache.get_auc_scores_path(STEERING_METHOD)
probe_metadata_path = cache.get_probe_metadata_path(STEERING_METHOD)

# Try to load from cache
try:
    if USE_CACHED_PROBES and os.path.exists(probe_coefficients_path):
        print("📁 Found cached steering vectors, loading...")
        probe_coefficients = cache.load_pickle(probe_coefficients_path)
        
        # Load AUC scores if available
        if os.path.exists(auc_scores_path):
            try:
                auc_data = cache.load_json(auc_scores_path)
                if isinstance(auc_data, list):
                    # Handle list format (legacy)
                    probe_results = {layer: {"test_auc": score} for layer, score in enumerate(auc_data)}
                else:
                    # Handle dict format  
                    probe_results = {int(k): {"test_auc": v} for k, v in auc_data.items()}
            except:
                probe_results = {layer: {"test_auc": 0.0} for layer in probe_coefficients.keys()}
        else:
            probe_results = {layer: {"test_auc": 0.0} for layer in probe_coefficients.keys()}
            
        print(f"✓ Loaded cached {STEERING_METHOD} steering vectors for {len(probe_coefficients)} layers")
        print("\n⚠️  Note: To force recomputation, delete the cache files or use a different method")
        
        # Quick validation
        layers = sorted(probe_coefficients.keys())
        print(f"\n📊 Cached steering vectors summary:")
        print(f"  Layers: {layers[0]} to {layers[-1]}")
        print(f"  Vector dimension: {probe_coefficients[layers[0]].shape}")
        
        USE_CACHE = True
    else:
        print("🔄 No cache found, computing fresh steering vectors...")
        USE_CACHE = False
except Exception as e:
    print(f"⚠️  Error loading cache: {e}")
    print("🔄 Computing fresh steering vectors...")
    USE_CACHE = False

#%%

STEERING_METHOD = "caa-single-layer"
print(USE_CACHE)
#%%

if not USE_CACHE:
    print(f"\n🎯 Computing {STEERING_METHOD} steering vectors...")
    
    # Prepare training data for probes
    train_labels = [1 if r["correct_answer"] == "yes" else 0 for r in train_results]
    test_labels = [1 if r["correct_answer"] == "yes" else 0 for r in test_results]
    print("Num yes: ", sum(train_labels))
    print("Num no: ", len(train_labels) - sum(train_labels))
    print("Num yes: ", sum(test_labels))
    print("Num no: ", len(test_labels) - sum(test_labels))
    
    print(f"📊 Probe training data:")
    # print(f"  Train samples: {len(train_labels)} ({'yes': {sum(train_labels)}, 'no': {len(train_labels) - sum(train_labels)}})")
    # print(f"  Test samples: {len(test_labels)} ({'yes': {sum(test_labels)}, 'no': {len(test_labels) - sum(test_labels)}})")
    
    # Get layer information
    if use_nnsight:
        # For nnsight, activations are indexed by layer number
        layers = sorted(train_activations.keys())
    else:
        layers = list(range(model.cfg.n_layers))
    
    print(f"  Training probes for {len(layers)} layers: {layers}")
    
    probe_results = {}
    probe_coefficients = {}
    contrastive_vectors = {}  # For CAA methods
    
    # First, collect data in format needed for steering methods
    if STEERING_METHOD in ["caa-single-layer", "caa-layer-incremental"]:
        print("\n🔬 Computing contrastive activation vectors...")
        
        for layer in layers:
            print(f"\n🔍 Computing contrastive vector for layer {layer}...")
            
            # Get activations for this layer
            X_train = train_activations[layer].squeeze()  # Remove batch dimension if present
            X_test = test_activations[layer].squeeze()
            
            # Handle dimension mismatches
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
            
            # Raw difference vector (no normalization for CAA methods)
            contrastive_vectors[layer] = mean_yes - mean_no
            
            # Compute similarity scores for evaluation
            test_labels_array = np.array(test_labels)
            similarities = X_test @ contrastive_vectors[layer]
            auc_score = roc_auc_score(test_labels_array, similarities)
            
            probe_results[layer] = {
                "train_auc": auc_score,  # Using test AUC as proxy
                "test_auc": auc_score,
                "similarity_score": auc_score
            }
            
            if layer % 5 == 0:  # Print every 5th layer
                print(f"  Layer {layer}: similarity AUC = {auc_score:.3f}")
            
            print(f"  ✓ Layer {layer}: Similarity AUC={auc_score:.3f}")
    
    # Apply steering method to get final steering vectors
    if STEERING_METHOD == "caa-single-layer":
        print("\n🎯 Applying CAA Single-Layer steering method...")
        similarity_scores = [probe_results[layer]["similarity_score"] for layer in layers]
        steering_method = CAASingleLayerSteering(similarity_scores)
        layer_vectors = [contrastive_vectors[layer] for layer in layers]
        steering_vectors = steering_method.compute_steering_vectors(layer_vectors)
        
        # For single-layer CAA, only use the best layer
        best_layer_idx = np.argmax(similarity_scores)
        best_layer = layers[best_layer_idx]
        best_score = similarity_scores[best_layer_idx]
        
        print(f"  Selected layer {best_layer} with similarity score {best_score:.3f}")
        
        # Only store the best layer's steering vector
        probe_coefficients = {best_layer: steering_vectors[best_layer_idx]}
        print(f"  Using only layer {best_layer} for steering (single-layer method)")
        
    elif STEERING_METHOD == "caa-layer-incremental":
        print("\n🎯 Applying CAA Layer-Incremental steering method...")
        steering_method = CAALayerIncrementalSteering()
        layer_vectors = [contrastive_vectors[layer] for layer in layers]
        steering_vectors = steering_method.compute_steering_vectors(layer_vectors)
        
        # Convert to dictionary format
        for i, layer in enumerate(layers):
            probe_coefficients[layer] = steering_vectors[i]
            
        print(f"  Computed incremental steering vectors for {len(layers)} layers")
        
    else:  # logistic-regression
        print("\n🎯 Using Logistic Regression steering method...")
        
        for layer in layers:
            print(f"\n🔍 Training probe for layer {layer}...")
            
            # Get activations for this layer
            X_train = train_activations[layer].squeeze()
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
    
    # Cache the results for future use
    print("\n💾 Saving steering vectors to cache...")
    try:
        # Save probe coefficients (steering vectors)
        cache.save_pickle(probe_coefficients, probe_coefficients_path)
        
        # Save AUC scores as dict for better compatibility
        if probe_results:
            auc_scores = {str(layer): probe_results.get(layer, {}).get('test_auc', 0.0) 
                         for layer in sorted(probe_results.keys())}
            cache.save_json(auc_scores, auc_scores_path)
        
        print(f"✓ Cached {STEERING_METHOD} steering vectors to:")
        print(f"  - {probe_coefficients_path}")
        print(f"  - {auc_scores_path}")
    except Exception as e:
        print(f"⚠️  Failed to cache results: {e}")

# Display probe training results
print(f"\n📊 Steering Method Results Summary:")
if STEERING_METHOD in ["caa-single-layer", "caa-layer-incremental"]:
    print("Layer | Similarity AUC")
    print("-" * 25)
    for layer in sorted(probe_results.keys()):
        results = probe_results[layer]
        print(f"{layer:5d} | {results['test_auc']:13.3f}")
else:
    print("Layer | Train AUC | Test AUC | Train Acc | Test Acc")
    print("-" * 50)
    for layer in sorted(probe_results.keys()):
        results = probe_results[layer]
        print(f"{layer:5d} | {results['train_auc']:9.3f} | {results['test_auc']:8.3f} | {results.get('train_accuracy', 0):9.3f} | {results.get('test_accuracy', 0):8.3f}")

# Find best performing layer
best_layer = max(probe_results.keys(), key=lambda l: probe_results[l]['test_auc'])
best_auc = probe_results[best_layer]['test_auc']
print(f"\n🏆 Best performing layer: {best_layer} (Test AUC: {best_auc:.3f})")

#%%
# STEERING HELPERS
import torch
from functools import partial
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint


# ─────────────────────────────────────────────────────────────────────────────
# Steering hook: fires only when generate is in single-token mode (seq_len == 1)
# ─────────────────────────────────────────────────────────────────────────────
def _steer_generated_token(
    resid: torch.Tensor,            # [B, 1, d_model] during generation
    hook: HookPoint,
    *,
    steering_vectors: torch.Tensor, # [n_layers, d_model]
    alpha: float,
) -> torch.Tensor:
    if resid.size(1) == 1:          # skip the prompt pass (seq_len > 1)
        resid += alpha * steering_vectors[hook.layer()][None, None, :]
    return resid


# ─────────────────────────────────────────────────────────────────────────────
# Main convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_with_steering_transformer_lens(
    model,
    prompt_tokens: torch.LongTensor,      # [B, prompt_len]
    steering_vectors,                     # NumPy or torch, [n_layers, d_model]
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    alpha: float = 5.0,
    layers=None,
):
    # 1. Normalise steering_vectors to correct dtype / device
    steering_vectors = torch.as_tensor(
        steering_vectors,
        dtype=model.W_E.dtype,
        device=model.W_E.device,
    )

    # 2. Decide which layers to steer
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    steer_hook = partial(_steer_generated_token,
                         steering_vectors=steering_vectors,
                         alpha=alpha)

    # 3. Register hooks once, generate, then clear hooks
    for l in layers:
        name = utils.get_act_name("resid_post", l)
        model.add_hook(name, steer_hook, dir="fwd")   # returns None on PyPI build

    # run generate (steering active)
    full_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        prepend_bos=False,
    )

    model.reset_hooks()            # ← one call clears every registered hook

    # Return only the freshly generated part
    gen_only = full_tokens[:, prompt_tokens.size(1):]
    return model.to_string(gen_only)


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

print("STEERING METHOD: ", STEERING_METHOD)

alpha_range = [2, 4, 6]

# for alpha in alpha_range:
for alpha in alpha_range:
    print(f"\n🎮 Testing steering with alpha = {alpha}")
    
    # Alpha for steering "yes" to "no" (negative)
    alpha_yes_to_no = -abs(alpha)
    # Alpha for steering "no" to "yes" (positive)  
    alpha_no_to_yes = abs(alpha)
    
    steering_results[alpha] = {"yes_to_no": [], "no_to_yes": []}
    
    max_gen = 10
    # Steer "yes" answers to "no"
    if yes_test_data:
        print(f"  🔄 Steering {len(yes_test_data)} 'yes' examples to 'no' (alpha={alpha_yes_to_no})")
        
        for i, result in enumerate(yes_test_data[:max_gen]):  # Limit for demo
            # Apply chat template to prompt
            prompt_string = model.apply_chat_template(result["prompt"])
            if use_nnsight:
                # Use nnsight steering with ProbeResult
                layers_to_steer = sorted(probe_coefficients.keys())
                # Convert steering vectors to the format expected by ProbeResult
                steering_vectors_dict = probe_coefficients
                
                # Create ProbeResult object 
                probe_result = ProbeResult(
                    method=STEERING_METHOD,
                    vectors=steering_vectors_dict,
                    scores={layer: probe_results.get(layer, {}).get('test_auc', 0.0) for layer in layers_to_steer},
                    best_layer=layers_to_steer[0] if layers_to_steer else None
                )
                
                # Tokenize the prompt
                tokens = model.to_tokens(prompt_string, prepend_bos=False)
                
                steered_response = generate_with_steering(
                    model=model,
                    tokens=tokens,
                    probe_result=probe_result,
                    alpha=alpha_yes_to_no,
                    max_new_tokens=exp_config.max_new_tokens,
                    temperature=exp_config.temperature,
                    do_sample=True
                )
            else:
                # Use transformer lens steering
                layers_to_steer = sorted(probe_coefficients.keys())
                # For single-layer method, we might only have one layer
                if STEERING_METHOD == "caa-single-layer":
                    # Create steering vectors for all layers, but only the selected layer has non-zero values
                    all_layers = list(range(model.cfg.n_layers))
                    steering_vectors_array = np.zeros((len(all_layers), probe_coefficients[layers_to_steer[0]].shape[0]))
                    # Set the selected layer's vector
                    for layer in layers_to_steer:
                        steering_vectors_array[layer] = probe_coefficients[layer]
                    layers_to_steer = all_layers
                else:
                    # Convert steering vectors to numpy array format
                    steering_vectors_array = np.array([probe_coefficients[layer] for layer in layers_to_steer])
                
                # Tokenize the prompt
                
                tokens = model.to_tokens(prompt_string, prepend_bos=False)
                
                # steered_response = generate_with_steering(
                #     model=model,
                #     tokens=tokens,
                #     steering_vectors=steering_vectors_array,
                #     alpha=alpha_yes_to_no,
                #     max_new_tokens=exp_config.max_new_tokens,
                #     temperature=exp_config.temperature,
                #     layers=layers_to_steer,
                #     verbose=False
                # )
                steered_response = generate_with_steering_transformer_lens(
                    model=model,
                    prompt_tokens=tokens,
                    steering_vectors=torch.tensor(steering_vectors_array,
                                                dtype=model.W_E.dtype,
                                                device=model.W_E.device),
                    alpha=alpha_yes_to_no,
                    max_new_tokens=exp_config.max_new_tokens,
                    temperature=exp_config.temperature,
                    layers=layers_to_steer,
                )
            print()
            print("Response: ", steered_response[-200:])
            if use_nnsight:
                # NNsight returns a single string
                steered_letter, steered_answer = parse_response(steered_response, thinking=True)
                print("Pred Letter: ", steered_letter)
                print("Pred Answer: ", steered_answer)
                success = steered_answer == "no"
                
                steering_results[alpha]["yes_to_no"].append({
                    "original_answer": result["pred_answer"],
                    "steered_answer": steered_answer,
                    "target_answer": "no",
                    "success": success,
                    "original_response": result["response"],
                    "steered_response": steered_response
                })
                
                print(f"    Example {i+1}: '{result['pred_answer']}' → '{steered_answer}' {'✓' if success else '✗'}")
            else:
                # TransformerLens returns a list of strings
                for resp in steered_response:           
                    steered_letter, steered_answer = parse_response(resp, thinking=True)
                    print("Pred Letter: ", steered_letter)
                    print("Pred Answer: ", steered_answer)
                    success = steered_answer == "no"
                    
                    steering_results[alpha]["yes_to_no"].append({
                        "original_answer": result["pred_answer"],
                        "steered_answer": steered_answer,
                        "target_answer": "no",
                        "success": success,
                        "original_response": result["response"],
                        "steered_response": resp
                    })
                    
                    print(f"    Example {i+1}: '{result['pred_answer']}' → '{steered_answer}' {'✓' if success else '✗'}")
    
    # Steer "no" answers to "yes"
    if no_test_data:
        print(f"  🔄 Steering {len(no_test_data)} 'no' examples to 'yes' (alpha={alpha_no_to_yes})")
        
        for i, result in enumerate(no_test_data[:max_gen]):  # Limit for demo
            prompt_string = model.apply_chat_template(result["prompt"])
            
            if use_nnsight:
                # Use nnsight steering with ProbeResult
                layers_to_steer = sorted(probe_coefficients.keys())
                # Convert steering vectors to the format expected by ProbeResult
                steering_vectors_dict = probe_coefficients
                
                # Create ProbeResult object 
                probe_result = ProbeResult(
                    method=STEERING_METHOD,
                    vectors=steering_vectors_dict,
                    scores={layer: probe_results.get(layer, {}).get('test_auc', 0.0) for layer in layers_to_steer},
                    best_layer=layers_to_steer[0] if layers_to_steer else None
                )
                
                # Tokenize the prompt
                tokens = model.to_tokens(prompt_string, prepend_bos=False)
                
                steered_response = generate_with_steering(
                    model=model,
                    tokens=tokens,
                    probe_result=probe_result,
                    alpha=alpha_no_to_yes,
                    max_new_tokens=exp_config.max_new_tokens,
                    temperature=exp_config.temperature,
                    do_sample=True
                )
            else:
                # Use transformer lens steering
                layers_to_steer = sorted(probe_coefficients.keys())
                # For single-layer method, we might only have one layer
                if STEERING_METHOD == "caa-single-layer":
                    # Create steering vectors for all layers, but only the selected layer has non-zero values
                    all_layers = list(range(model.cfg.n_layers))
                    steering_vectors_array = np.zeros((len(all_layers), probe_coefficients[layers_to_steer[0]].shape[0]))
                    # Set the selected layer's vector
                    for layer in layers_to_steer:
                        steering_vectors_array[layer] = probe_coefficients[layer]
                    layers_to_steer = all_layers
                else:
                    # Convert steering vectors to numpy array format
                    steering_vectors_array = np.array([probe_coefficients[layer] for layer in layers_to_steer])
                
                # Tokenize the prompt
                tokens = model.to_tokens(prompt_string, prepend_bos=False)
                
                # steered_response = generate_with_steering(
                #     model=model,
                #     tokens=tokens,
                #     steering_vectors=steering_vectors_array,
                #     alpha=alpha_no_to_yes,
                #     max_new_tokens=exp_config.max_new_tokens,
                #     temperature=exp_config.temperature,
                #     layers=layers_to_steer,
                #     verbose=False
                # )
                steered_response = generate_with_steering_transformer_lens(
                    model=model,
                    prompt_tokens=tokens,
                    steering_vectors=steering_vectors_array,
                    alpha=alpha_no_to_yes,
                    max_new_tokens=exp_config.max_new_tokens,
                    temperature=exp_config.temperature,
                    layers=layers_to_steer,
                )
            print()
            print("Response: ", steered_response[-200:])
            if use_nnsight:
                # NNsight returns a single string
                steered_letter, steered_answer = parse_response(steered_response, thinking=True)
                print("Pred Letter: ", steered_letter)
                print("Pred Answer: ", steered_answer)
                success = steered_answer == "yes"
                
                steering_results[alpha]["no_to_yes"].append({
                    "original_answer": result["pred_answer"],
                    "steered_answer": steered_answer,
                    "target_answer": "yes",
                    "success": success,
                    "original_response": result["response"],
                    "steered_response": steered_response
                })
                
                print(f"    Example {i+1}: '{result['pred_answer']}' → '{steered_answer}' {'✓' if success else '✗'}")
            else:
                # TransformerLens returns a list of strings
                for resp in steered_response:           
                    steered_letter, steered_answer = parse_response(resp, thinking=True)
                    print("Pred Letter: ", steered_letter)
                    print("Pred Answer: ", steered_answer)
                    success = steered_answer == "yes"
                
                steering_results[alpha]["no_to_yes"].append({
                    "original_answer": result["pred_answer"],
                    "steered_answer": steered_answer,
                    "target_answer": "yes",
                    "success": success,
                    "original_response": result["response"],
                    "steered_response": resp
                })
                
                print(f"    Example {i+1}: '{result['pred_answer']}' → '{steered_answer}' {'✓' if success else '✗'}")

# Calculate steering success rates
print(f"\n📊 Steering Results Summary:")
print("Alpha | Yes→No Success | No→Yes Success | Overall Success")
print("-" * 55)

# for alpha in alpha_range:
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
if STEERING_METHOD in ["caa-single-layer", "caa-layer-incremental"]:
    print(f"  - contrastive_vectors: Raw contrastive activation vectors per layer")

print(f"\n🌟 Steering Method Analysis:")
print(f"Method: {STEERING_METHOD}")
if STEERING_METHOD == "caa-single-layer":
    print(f"  - Single layer selected for steering")
    print(f"  - Uses raw difference vectors without normalization")
    print(f"  - Most effective when one layer captures the concept well")
elif STEERING_METHOD == "caa-layer-incremental":
    print(f"  - Distributes steering across all layers")
    print(f"  - Uses incremental differences with RMS normalization")
    print(f"  - Better for concepts distributed across layers")
else:
    print(f"  - Traditional logistic regression approach")
    print(f"  - Uses all coefficient vectors from trained probes")
    print(f"  - Well-established baseline method")

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