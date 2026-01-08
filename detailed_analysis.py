#!/usr/bin/env python3
"""
Detailed analysis of debiasing results with heatmap visualization.
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Path to the cache directory
# cache_dir = "cache_20250818"
cache_dir = "cache"
experiments_dir = os.path.join(cache_dir, "experiments")

# List of datasets to analyze
datasets = [
    "anachronisms",
    "logical_deduction", 
    "social_chemistry",
    "sports_understanding"
]

def find_correct_split_directory(model_experiments_dir):
    """Find the correct split directory with pattern 'split_{x}_500_500'."""
    if not os.path.exists(model_experiments_dir):
        return None
    
    split_dirs = [d for d in os.listdir(model_experiments_dir) 
                  if os.path.isdir(os.path.join(model_experiments_dir, d))]
    
    # Look for the correct split pattern
    correct_splits = [d for d in split_dirs if d.startswith("split_") and "_500_500" in d]
    
    if correct_splits:
        return correct_splits[0]
    else:
        split_dirs = [d for d in split_dirs if d.startswith("split_")]
        return split_dirs[0] if split_dirs else None

def collect_all_results():
    """Collect all debiasing results from the cache."""
    all_results = {}
    
    if not os.path.exists(experiments_dir):
        print(f"❌ Experiments directory not found: {experiments_dir}")
        return all_results
    
    model_dirs = [d for d in os.listdir(experiments_dir) 
                  if os.path.isdir(os.path.join(experiments_dir, d))]
    
    for model_name in model_dirs:
        print(f"Processing {model_name}...")
        model_experiments_dir = os.path.join(experiments_dir, model_name)
        
        for dataset_name in datasets:
            dataset_dir = os.path.join(model_experiments_dir, dataset_name)
            if not os.path.exists(dataset_dir):
                continue
            
            # Find the correct split directory
            split_dir = find_correct_split_directory(dataset_dir)
            if not split_dir:
                continue
            
            split_path = os.path.join(dataset_dir, split_dir)
            
            # Find the first experiment hash directory
            exp_hash_dirs = [d for d in os.listdir(split_path) 
                            if os.path.isdir(os.path.join(split_path, d))]
            if not exp_hash_dirs:
                continue
            
            exp_hash_dir = exp_hash_dirs[0]
            debiasing_results_path = os.path.join(split_path, exp_hash_dir, "debiasing", "debiasing_alpha_0.0.pkl")
            
            if not os.path.exists(debiasing_results_path):
                continue
            
            try:
                with open(debiasing_results_path, "rb") as f:
                    debiasing_results = pickle.load(f)
                
                # Extract evaluation metrics
                eval_dict = debiasing_results.get('evaluation', {})
                original_acc = eval_dict.get('original_accuracy', None)
                debiased_acc = eval_dict.get('debiased_accuracy', None)
                
                if original_acc is not None and debiased_acc is not None:
                    key = (model_name, dataset_name)
                    all_results[key] = {
                        'original_accuracy': original_acc * 100,
                        'debiased_accuracy': debiased_acc * 100,
                        'accuracy_change': (debiased_acc - original_acc) * 100
                    }
                    
            except Exception as e:
                print(f"Error loading {model_name}/{dataset_name}: {e}")
    
    return all_results

def create_heatmap(all_results):
    """Create a heatmap showing accuracy changes."""
    # Prepare data for heatmap
    models = sorted(list(set([key[0] for key in all_results.keys()])))
    datasets = ["anachronisms", "logical_deduction", "social_chemistry", "sports_understanding"]
    
    # Create matrices
    original_matrix = np.full((len(models), len(datasets)), np.nan)
    debiased_matrix = np.full((len(models), len(datasets)), np.nan)
    change_matrix = np.full((len(models), len(datasets)), np.nan)
    
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            key = (model, dataset)
            if key in all_results:
                original_matrix[i, j] = all_results[key]['original_accuracy']
                debiased_matrix[i, j] = all_results[key]['debiased_accuracy']
                change_matrix[i, j] = all_results[key]['accuracy_change']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Clean model names for display
    clean_model_names = [name.replace('google_', '').replace('_', '-') for name in models]
    
    # 1. Original Accuracy Heatmap
    sns.heatmap(original_matrix, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues', 
                xticklabels=datasets, 
                yticklabels=clean_model_names,
                ax=axes[0],
                cbar_kws={'label': 'Original Accuracy (%)'})
    axes[0].set_title('Original Accuracy (%)')
    axes[0].set_xlabel('Datasets')
    axes[0].set_ylabel('Models')
    
    # 2. Debiased Accuracy Heatmap
    sns.heatmap(debiased_matrix, 
                annot=True, 
                fmt='.1f', 
                cmap='Oranges', 
                xticklabels=datasets, 
                yticklabels=clean_model_names,
                ax=axes[1],
                cbar_kws={'label': 'Debiased Accuracy (%)'})
    axes[1].set_title('Debiased Accuracy (%)')
    axes[1].set_xlabel('Datasets')
    axes[1].set_ylabel('Models')
    
    # 3. Accuracy Change Heatmap
    sns.heatmap(change_matrix, 
                annot=True, 
                fmt='+.1f', 
                cmap='RdBu_r', 
                center=0,
                xticklabels=datasets, 
                yticklabels=clean_model_names,
                ax=axes[2],
                cbar_kws={'label': 'Accuracy Change (%)'})
    axes[2].set_title('Accuracy Change (%)')
    axes[2].set_xlabel('Datasets')
    axes[2].set_ylabel('Models')
    
    plt.tight_layout()
    plt.savefig('debiasing_heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_table(all_results):
    """Create a detailed table of all results."""
    print("\n📊 DETAILED RESULTS TABLE")
    print("=" * 120)
    
    # Prepare data for table
    table_data = []
    for (model_name, dataset_name), results in all_results.items():
        table_data.append({
            'Model': model_name.replace('google_', '').replace('_', '-'),
            'Dataset': dataset_name,
            'Original (%)': f"{results['original_accuracy']:.1f}",
            'Debiased (%)': f"{results['debiased_accuracy']:.1f}",
            'Change (%)': f"{results['accuracy_change']:+.1f}",
            'Improvement': '✅' if results['accuracy_change'] > 0 else '❌'
        })
    
    # Sort by model and dataset
    table_data.sort(key=lambda x: (x['Model'], x['Dataset']))
    
    # Create DataFrame and display
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('detailed_debiasing_results.csv', index=False)
    print(f"\n📄 Detailed results saved to 'detailed_debiasing_results.csv'")

def create_model_comparison_plot(all_results):
    """Create a comparison plot showing model performance."""
    # Group by model
    model_stats = {}
    for (model_name, dataset_name), results in all_results.items():
        if model_name not in model_stats:
            model_stats[model_name] = {
                'original_accuracies': [],
                'debiased_accuracies': [],
                'changes': []
            }
        
        model_stats[model_name]['original_accuracies'].append(results['original_accuracy'])
        model_stats[model_name]['debiased_accuracies'].append(results['debiased_accuracy'])
        model_stats[model_name]['changes'].append(results['accuracy_change'])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = list(model_stats.keys())
    clean_model_names = [name.replace('google_', '').replace('_', '-') for name in models]
    
    # Plot 1: Average accuracies
    avg_original = [np.mean(model_stats[model]['original_accuracies']) for model in models]
    avg_debiased = [np.mean(model_stats[model]['debiased_accuracies']) for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, avg_original, width, label='Original', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, avg_debiased, width, label='Debiased', color='orange', alpha=0.8)
    
    # Add change annotations
    for i, (orig, debiased) in enumerate(zip(avg_original, avg_debiased)):
        change = debiased - orig
        if abs(change) > 1:
            color = 'green' if change > 0 else 'red'
            ax1.annotate(f'{change:+.1f}%', 
                        xy=(i, max(orig, debiased) + 2), 
                        ha='center', va='bottom',
                        color=color, fontweight='bold')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Average Accuracy (%)')
    ax1.set_title('Average Original vs Debiased Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy changes distribution
    all_changes = []
    model_labels = []
    for model in models:
        changes = model_stats[model]['changes']
        all_changes.extend(changes)
        model_labels.extend([clean_model_names[models.index(model)]] * len(changes))
    
    # Create box plot
    change_data = []
    change_labels = []
    for model in models:
        changes = model_stats[model]['changes']
        if changes:  # Only include models with data
            change_data.append(changes)
            change_labels.append(clean_model_names[models.index(model)])
    
    if change_data:
        ax2.boxplot(change_data, labels=change_labels)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Change')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Accuracy Change (%)')
        ax2.set_title('Distribution of Accuracy Changes')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    print("🔍 Detailed Analysis of Debiasing Results")
    print("=" * 80)
    
    # Collect all results
    all_results = collect_all_results()
    
    if not all_results:
        print("❌ No results found!")
        return
    
    print(f"\n✅ Found {len(all_results)} model-dataset combinations with debiasing results")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # 1. Heatmap analysis
    create_heatmap(all_results)
    
    # 2. Detailed table
    create_detailed_table(all_results)
    
    # 3. Model comparison plot
    create_model_comparison_plot(all_results)
    
    print("\n✅ Detailed analysis complete!")

if __name__ == "__main__":
    main() 