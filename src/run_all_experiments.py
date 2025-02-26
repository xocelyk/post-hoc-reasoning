#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run experiments on multiple datasets with the Gemma model.
Supports generating examples and running steering experiments.
"""

import os
import argparse
import torch
from models import ChatModel
from main import generate_and_save_generations, run_steering_experiment

def run_experiments(
    model_name="google/gemma-2-9b-it",
    datasets=None,
    train_size=100,
    batch_size=2,
    max_new_tokens=100,
    temperature=0.7,
    alpha_range=None,
    use_cache=True,
    mode="all",
    device="cuda",
    dtype="bfloat16"
):
    """
    Run experiments on specified datasets with the Gemma model.
    
    Args:
        model_name: Name of the model to use
        datasets: List of datasets to run experiments on (if None, use all)
        train_size: Train set size for generation experiments
        batch_size: Batch size for processing
        max_new_tokens: Maximum new tokens to generate
        temperature: Temperature for generation
        alpha_range: Alpha range for steering experiments (if None, use default)
        use_cache: Use cached generations if available
        mode: One of ["generate", "steer", "all"]
        device: Device to run the model on ("cuda" or "cpu")
        dtype: Data type for model weights
    """
    if alpha_range is None:
        alpha_range = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
    if datasets is None:
        datasets = [
            "city_reasoning", 
            "anachronisms", 
            "logical_deduction", 
            "social_chemistry", 
            "quora_question_pairs", 
            "sports_understanding"
        ]
    
    print(f"Loading model: {model_name}")
    model = ChatModel(model_name, device=device, dtype=dtype)
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Running experiments on dataset: {dataset_name}")
        print(f"{'='*50}\n")
        
        if mode in ["generate", "all"]:
            print(f"Generating examples for {dataset_name}...")
            try:
                generate_and_save_generations(
                    model,
                    dataset_name,
                    train_size=train_size,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_cache=use_cache
                )
            except Exception as e:
                print(f"Error generating examples for {dataset_name}: {e}")
                continue
        
        if mode in ["steer", "all"]:
            print(f"Running steering experiments for {dataset_name}...")
            try:
                run_steering_experiment(
                    model,
                    dataset_name,
                    alpha_range=alpha_range,
                    use_cache=use_cache
                )
            except Exception as e:
                print(f"Error running steering experiments for {dataset_name}: {e}")
                continue
        
        # Clear CUDA cache between datasets
        if device == "cuda":
            torch.cuda.empty_cache()
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments on multiple datasets with Gemma model")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it", help="Model name")
    parser.add_argument("--datasets", nargs="+", default=None, help="List of datasets to run experiments on")
    parser.add_argument("--train_size", type=int, default=100, help="Train set size")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--alpha_range", type=int, nargs="+", default=None, help="Alpha range for steering")
    parser.add_argument("--use_cache", action="store_true", help="Use cached generations if available")
    parser.add_argument("--mode", choices=["generate", "steer", "all"], default="all", help="Experiment mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on ('cuda' or 'cpu')")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for model weights")
    
    args = parser.parse_args()
    run_experiments(
        model_name=args.model,
        datasets=args.datasets,
        train_size=args.train_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        alpha_range=args.alpha_range,
        use_cache=args.use_cache,
        mode=args.mode,
        device=args.device,
        dtype=args.dtype
    )