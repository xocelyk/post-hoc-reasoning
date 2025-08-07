#!/usr/bin/env python3
"""Test W&B integration without interactive prompts."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from wandb_integration import WandbExperimentLogger, setup_wandb

def main():
    print("Testing W&B Integration (Non-Interactive)")
    print("="*60)
    
    # Test setup
    print("1. Checking W&B setup...")
    if not setup_wandb():
        print("\nW&B is not configured. Please run:")
        print("  wandb login")
        print("\nThen set environment variables:")
        print("  export WANDB_PROJECT='your-project-name'")
        print("  export WANDB_ENTITY='your-team-name'  # optional")
        return
    
    print("✓ W&B is configured")
    
    # Use environment variables or defaults
    project = os.environ.get("WANDB_PROJECT", "post-hoc-reasoning-test")
    entity = os.environ.get("WANDB_ENTITY", None)
    
    print(f"\n2. Using project: {project}")
    if entity:
        print(f"   Using entity: {entity}")
    else:
        print("   Using personal workspace")
    
    # Test in offline mode first
    print("\n3. Testing in OFFLINE mode first (won't create online run)")
    os.environ["WANDB_MODE"] = "offline"
    
    logger = WandbExperimentLogger(
        project=project,
        entity=entity,
        experiment_config={
            "test": True,
            "model_name": "google/gemma-2-2b-it",
            "dataset_name": "sports_understanding",
            "steering_method": "caa-single-layer",
            "train_size": 500,
            "test_size": 200,
            "split_seed": 42,
            "alpha_range": [-10, -5, -2, -1, 0, 1, 2, 5, 10],
            "temperature": 0.7,
            "max_new_tokens": 300,
            "backend": "transformer_lens"
        }
    )
    
    if logger.disabled:
        print("❌ W&B logging is disabled. Check your setup.")
        return
    
    print("✓ W&B run created in OFFLINE mode")
    
    # Log test data with realistic values
    print("\n4. Logging realistic test data...")
    
    # Simulate probe training across layers
    print("   - Simulating probe training for Gemma-2B (18 layers)...")
    layer_scores = [
        0.523, 0.542, 0.567, 0.589, 0.612, 0.634,  # Early layers
        0.656, 0.678, 0.701, 0.723, 0.745, 0.768,  # Middle layers  
        0.791, 0.813, 0.836, 0.819, 0.802, 0.785   # Late layers
    ]
    
    for layer, score in enumerate(layer_scores):
        logger.log_probe_training(
            layer=layer,
            train_auc=score + 0.02,  # Slightly higher train
            test_auc=score,
            similarity_score=score,
            method="caa-single-layer"
        )
    
    best_layer = 14  # Layer with highest score
    logger.log_best_layer_selection(
        best_layer=best_layer,
        best_score=0.836,
        method="caa-single-layer",
        all_scores=layer_scores
    )
    
    # Simulate realistic steering results
    print("   - Simulating steering experiments...")
    
    test_prompts = [
        "Is basketball played with a ball?",
        "Is soccer played on a field?",
        "Is tennis played with a racket?",
        "Is swimming done in water?",
        "Is golf played with clubs?",
    ]
    
    # Test different alpha values
    for alpha in [-5, -2, 2, 5]:
        direction = "yes_to_no" if alpha < 0 else "no_to_yes"
        
        # Success rate varies with alpha magnitude
        success_rate = min(0.9, abs(alpha) * 0.15)
        
        for i, prompt in enumerate(test_prompts * 2):  # 10 examples per alpha
            # Determine outcome based on success rate
            import random
            random.seed(42 + i + int(alpha * 10))  # Reproducible randomness
            
            if random.random() < success_rate:
                category = "success"
                if direction == "yes_to_no":
                    steered_answer = "no"
                    response_end = "...The answer is no."
                else:
                    steered_answer = "yes"
                    response_end = "...The answer is yes."
            elif random.random() < 0.9:  # 90% of non-success are failures
                category = "failure"
                if direction == "yes_to_no":
                    steered_answer = "yes"
                    response_end = "...The answer is yes."
                else:
                    steered_answer = "no" 
                    response_end = "...The answer is no."
            else:
                category = "unparsed"
                steered_answer = "unparsed"
                response_end = "...[garbled output]"
            
            logger.log_steering_example(
                alpha=alpha,
                direction=direction,
                prompt=prompt,
                original_answer="yes" if direction == "yes_to_no" else "no",
                steered_response=f"Let me think about this {response_end}",
                steered_answer=steered_answer,
                target_answer="no" if direction == "yes_to_no" else "yes",
                category=category,
                example_idx=i,
                model_name="google/gemma-2-2b-it",
                dataset_name="sports_understanding"
            )
        
        # Log summary for this alpha/direction
        success_count = int(10 * success_rate)
        unparsed_count = 1 if success_rate < 0.7 else 0
        failure_count = 10 - success_count - unparsed_count
        
        logger.log_steering_summary(
            alpha=abs(alpha),
            direction=direction.split("_")[0],  # "yes" or "no"
            total_examples=10,
            success_count=success_count,
            failure_count=failure_count,
            unparsed_count=unparsed_count
        )
    
    # Create success rate plot
    print("   - Creating success rate visualization...")
    alpha_values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    yes_to_no_rates = [0.95, 0.75, 0.60, 0.55, 0.50, 0.45, 0.40, 0.25, 0.10]
    no_to_yes_rates = [0.10, 0.25, 0.40, 0.45, 0.50, 0.55, 0.60, 0.75, 0.95]
    
    logger.create_steering_success_plot(
        alpha_values=alpha_values,
        yes_to_no_rates=yes_to_no_rates,
        no_to_yes_rates=no_to_yes_rates
    )
    
    # Log experiment summary
    logger.log_experiment_summary({
        "model": "google/gemma-2-2b-it",
        "dataset": "sports_understanding",
        "train_accuracy": 0.85,
        "test_accuracy": 0.82,
        "best_probe_layer": best_layer,
        "best_probe_score": 0.836,
        "steering_method": "caa-single-layer",
        "best_steering_alpha_yes_to_no": -5.0,
        "best_steering_alpha_no_to_yes": 5.0,
        "best_steering_success_rate": 0.75,
        "avg_unparsed_rate": 0.05
    })
    
    # Finish
    logger.finish()
    
    print("\n✅ Test complete!")
    print("\nSince we ran in OFFLINE mode, the run was saved locally.")
    print("Check the 'wandb/offline-run-*' directory for the saved data.")
    print("\nTo sync this run online later, run:")
    print("  wandb sync wandb/offline-run-*")
    print("\nTo test with real online logging:")
    print("  export WANDB_MODE=online")
    print("  python test_wandb_noninteractive.py")
    print("\nNext steps:")
    print("1. Set your project name: export WANDB_PROJECT='your-actual-project'")
    print("2. Set your team name: export WANDB_ENTITY='your-team-name'")
    print("3. Run experiments with W&B tracking enabled!")

if __name__ == "__main__":
    main()