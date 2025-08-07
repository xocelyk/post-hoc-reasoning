import pickle
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter

def check_experiment(exp_path):
    """Check an experiment for potential AUC issues."""
    print(f"\nChecking: {exp_path}")
    
    # Load train/test generations
    train_path = os.path.join(exp_path, 'data', 'train_generations.pkl')
    test_path = os.path.join(exp_path, 'data', 'test_generations.pkl')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("  Missing generation files")
        return
        
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # Check label distributions
    train_correct = [d['correct_answer'] for d in train_data]
    test_correct = [d['correct_answer'] for d in test_data]
    
    print(f"  Train distribution: {dict(Counter(train_correct))}")
    print(f"  Test distribution: {dict(Counter(test_correct))}")
    
    # Filter to only correct predictions for training
    train_correct_preds = [d for d in train_data if d['pred_answer'] == d['correct_answer']]
    train_correct_labels = [d['correct_answer'] for d in train_correct_preds]
    print(f"  Train correct predictions: {len(train_correct_preds)}/{len(train_data)}")
    print(f"  Train correct distribution: {dict(Counter(train_correct_labels))}")
    
    # Check if we have both classes in train correct predictions
    unique_train_correct = set(train_correct_labels)
    if len(unique_train_correct) < 2:
        print("  WARNING: Only one class in correctly predicted training data!")
        print(f"  Classes in correct predictions: {unique_train_correct}")
        
    # Load AUC scores if available
    auc_path = os.path.join(exp_path, 'probes', 'caa-single-layer', 'auc_scores.json')
    if not os.path.exists(auc_path):
        auc_path = os.path.join(exp_path, 'probes', 'auc_scores.json')
    
    if os.path.exists(auc_path):
        with open(auc_path, 'r') as f:
            auc_scores = json.load(f)
            if isinstance(auc_scores, dict):
                max_auc = max(auc_scores.values())
            else:
                max_auc = max(auc_scores)
            print(f"  Max AUC: {max_auc:.4f}")
            
            if max_auc < 0.4:
                print("  POTENTIAL ISSUE: Very low AUC!")

# Check logical deduction experiments
print("Analyzing logical_deduction experiments for AUC issues...")
print("="*70)

cache_dir = 'cache/experiments'
found_issues = []

for root, dirs, files in os.walk(cache_dir):
    if 'logical_deduction' in root and 'data' in dirs:
        check_experiment(root)
        
print("\nSummary:")
print("The issue might be:")
print("1. If the model gets all 'yes' or all 'no' correct, the probe can't learn to discriminate")
print("2. If the test set is imbalanced differently than train, AUC can be affected")
print("3. The probe might be learning the inverse relationship (predicting 'no' when it should predict 'yes')")