#!/usr/bin/env python3
"""Test W&B integration before running full experiments."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from wandb_integration import WandbExperimentLogger, setup_wandb
except ImportError:
    print("Error: Could not import wandb_integration")
    print("Make sure wandb_integration.py is in the src/ directory")
    sys.exit(1)

def main():
    print("Testing W&B Integration")
    print("="*60)
    
    # Test setup
    print("1. Checking W&B setup...")
    if not setup_wandb():
        print("\nPlease complete W&B setup:")
        print("1. Run: pip install wandb")
        print("2. Run: wandb login")
        print("3. Get API key from: https://wandb.ai/authorize")
        return
    
    print("✓ W&B is configured")
    
    # Ask for project details
    print("\n2. Project Configuration")
    print("Ask your colleague for these details:")
    project = input("Enter W&B project name (or press Enter for 'post-hoc-reasoning'): ").strip()
    if not project:
        project = "post-hoc-reasoning"
    
    entity = input("Enter W&B entity/team name (or press Enter for personal): ").strip()
    if not entity:
        entity = None
    
    # Test logging
    print(f"\n3. Creating test run in project '{project}'...")
    
    logger = WandbExperimentLogger(
        project=project,
        entity=entity,
        experiment_config={
            "test": True,
            "model": "test-model",
            "dataset": "test-dataset",
            "description": "Testing W&B integration"
        }
    )
    
    if logger.disabled:
        print("❌ W&B logging is disabled. Check your setup.")
        return
    
    print("✓ W&B run created")
    
    # Log some test data
    print("\n4. Logging test data...")
    
    # Simulate probe training results
    print("   - Logging probe training results...")
    for layer in range(5):
        logger.log_probe_training(
            layer=layer,
            train_auc=0.5 + layer * 0.1,
            test_auc=0.5 + layer * 0.08,
            similarity_score=0.5 + layer * 0.09,
            method="caa-single-layer"
        )
    
    logger.log_best_layer_selection(
        best_layer=3,
        best_score=0.82,
        method="caa-single-layer",
        all_scores=[0.5, 0.6, 0.7, 0.82, 0.75]
    )
    
    # Simulate steering results
    print("   - Logging steering examples...")
    
    # Successful steering examples
    for i in range(7):
        logger.log_steering_example(
            alpha=2.0,
            direction="yes_to_no",
            prompt="Is basketball played with a ball?",
            original_answer="yes",
            steered_response="Basketball is played with many things... The answer is no.",
            steered_answer="no",
            target_answer="no",
            category="success",
            example_idx=i,
            model_name="test-model",
            dataset_name="test-dataset"
        )
    
    # Failed steering examples
    for i in range(7, 9):
        logger.log_steering_example(
            alpha=2.0,
            direction="yes_to_no",
            prompt="Is football a sport?",
            original_answer="yes",
            steered_response="Football is definitely a sport... The answer is yes.",
            steered_answer="yes",
            target_answer="no",
            category="failure",
            example_idx=i,
            model_name="test-model",
            dataset_name="test-dataset"
        )
    
    # Unparsed example
    logger.log_steering_example(
        alpha=2.0,
        direction="yes_to_no",
        prompt="Is tennis played on a court?",
        original_answer="yes",
        steered_response="Tennis court surface varies... [garbled output]",
        steered_answer="unparsed",
        target_answer="no",
        category="unparsed",
        example_idx=9,
        model_name="test-model",
        dataset_name="test-dataset"
    )
    
    # Log summary
    print("   - Logging summary statistics...")
    logger.log_steering_summary(
        alpha=2.0,
        direction="yes_to_no",
        total_examples=10,
        success_count=7,
        failure_count=2,
        unparsed_count=1
    )
    
    # Create success rate plot
    print("   - Creating success rate plot...")
    alpha_values = [-5, -2, -1, 0, 1, 2, 5]
    yes_to_no_rates = [0.1, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8]
    no_to_yes_rates = [0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.1]
    
    logger.create_steering_success_plot(
        alpha_values=alpha_values,
        yes_to_no_rates=yes_to_no_rates,
        no_to_yes_rates=no_to_yes_rates
    )
    
    # Log final summary
    logger.log_experiment_summary({
        "model": "test-model",
        "dataset": "test-dataset",
        "train_accuracy": 0.85,
        "test_accuracy": 0.82,
        "best_probe_layer": 3,
        "best_probe_score": 0.82,
        "steering_method": "caa-single-layer",
        "best_steering_alpha": 2.0,
        "best_steering_success_rate": 0.7
    })
    
    # Finish
    logger.finish()
    
    print("\n✅ Test complete!")
    print(f"\nView your results at: https://wandb.ai/{entity or 'YOUR-USERNAME'}/{project}")
    print("\nIf everything looks good, you can now:")
    print("1. Add the wandb_integration.py file to your experiments")
    print("2. Modify experiment_runner.py as shown in add_wandb_to_experiment_runner.py")
    print("3. Run your experiments with W&B tracking enabled!")

if __name__ == "__main__":
    main()