#!/usr/bin/env python3
"""
Visualize rollout classification results with dynamic alpha bucketing.
For each model-dataset pair, divides available alphas into low/medium/high thirds.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Configuration
CACHE_DIR = Path("final_cache/cache/rollout_classification")
OUTPUT_PATH = "figs/rollout_classification_bucketed_no.png"

# Models and datasets to include
MODELS = [
    "google_gemma-2-2b-it",
    "google_gemma-2-9b-it",
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
]

DATASETS = [
    "logical_deduction",
    "sports_understanding",
    "social_chemistry",
    "anachronisms",
]

# Colors
COLORS = {
    'confabulation': '#e74c3c',      # red
    'non_entailment': '#3498db',     # blue
    'hallucination': '#f39c12',      # orange/yellow
    'sound': '#2ecc71',              # green
}

METRIC_LABELS = {
    'confabulation': 'Confab',
    'non_entailment': 'Non-ent',
    'hallucination': 'Halluc',
    'sound': 'Sound',
}


def load_rollout_classifications(cache_dir: Path):
    """Load all rollout classification results from JSONL files."""
    if not cache_dir.exists():
        print(f"Cache not found at {cache_dir}")
        return {}

    results = defaultdict(list)

    for jsonl_file in cache_dir.rglob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    model = data.get('model', 'unknown')
                    dataset = data.get('dataset', 'unknown')
                    alpha_abs = data.get('alpha_abs', 0)
                    direction = data.get('direction', 'unknown')

                    key = (model, dataset, alpha_abs, direction)
                    results[key].append({
                        'confabulation': data.get('confabulation', False),
                        'non_entailment': data.get('non_entailment', False),
                        'hallucination': data.get('hallucination', False),
                        'sound': data.get('sound', False),
                    })
                except:
                    continue

    return results


def get_available_alphas(data, model, dataset, direction='no'):
    """Get sorted list of available alpha values for a model-dataset pair."""
    alphas = set()
    for (m, d, alpha, dir_), records in data.items():
        if m == model and d == dataset and dir_ == direction and len(records) >= 5:
            alphas.add(alpha)
    return sorted(alphas)


def bucket_alphas(alphas):
    """Divide alphas into low/medium/high thirds."""
    if len(alphas) == 0:
        return {}
    if len(alphas) <= 3:
        # If 3 or fewer, just use what we have
        if len(alphas) == 1:
            return {'All': alphas}
        elif len(alphas) == 2:
            return {'Low': [alphas[0]], 'High': [alphas[1]]}
        else:
            return {'Low': [alphas[0]], 'Med': [alphas[1]], 'High': [alphas[2]]}

    # Split into thirds
    n = len(alphas)
    third = n // 3
    remainder = n % 3

    # Distribute remainder to make splits as even as possible
    if remainder == 0:
        low = alphas[:third]
        med = alphas[third:2*third]
        high = alphas[2*third:]
    elif remainder == 1:
        low = alphas[:third]
        med = alphas[third:2*third+1]
        high = alphas[2*third+1:]
    else:  # remainder == 2
        low = alphas[:third+1]
        med = alphas[third+1:2*third+1]
        high = alphas[2*third+1:]

    buckets = {}
    if low:
        label = f"Low\n({int(min(low))}-{int(max(low))})" if len(low) > 1 else f"Low\n({int(low[0])})"
        buckets[label] = low
    if med:
        label = f"Med\n({int(min(med))}-{int(max(med))})" if len(med) > 1 else f"Med\n({int(med[0])})"
        buckets[label] = med
    if high:
        label = f"High\n({int(min(high))}-{int(max(high))})" if len(high) > 1 else f"High\n({int(high[0])})"
        buckets[label] = high

    return buckets


def calculate_bucket_rates(data, model, dataset, bucket_alphas, direction='no'):
    """Calculate rates for a bucket of alpha values."""
    all_records = []
    for alpha in bucket_alphas:
        key = (model, dataset, float(alpha), direction)
        if key in data:
            all_records.extend(data[key])

    if not all_records:
        return None

    n = len(all_records)
    rates = {
        'n': n,
        'confabulation': sum(1 for r in all_records if r['confabulation']) / n,
        'non_entailment': sum(1 for r in all_records if r['non_entailment']) / n,
        'hallucination': sum(1 for r in all_records if r['hallucination']) / n,
        'sound': sum(1 for r in all_records if r['sound']) / n,
    }

    # Wilson CI
    def wilson_ci(p, n):
        if n == 0:
            return 0, 0
        z = 1.96
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
        return max(0, center - spread), min(1, center + spread)

    for metric in ['confabulation', 'non_entailment', 'hallucination', 'sound']:
        rates[f'{metric}_ci'] = wilson_ci(rates[metric], n)

    return rates


def main():
    print(f"Loading data from {CACHE_DIR}...")
    data = load_rollout_classifications(CACHE_DIR)
    print(f"Loaded {len(data)} unique combinations")
    print(f"Total samples: {sum(len(v) for v in data.values())}")

    # Create figure
    n_models = len(MODELS)
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(n_models, n_datasets, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.25, wspace=0.15, top=0.92, bottom=0.08, left=0.1, right=0.95)

    metrics = ['confabulation', 'non_entailment', 'hallucination', 'sound']
    bar_width = 0.2

    for model_idx, model in enumerate(MODELS):
        for dataset_idx, dataset in enumerate(DATASETS):
            ax = axes[model_idx, dataset_idx]

            # Get available alphas and bucket them
            alphas = get_available_alphas(data, model, dataset, direction='no')
            buckets = bucket_alphas(alphas)

            if not buckets:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue

            bucket_names = list(buckets.keys())
            x = np.arange(len(bucket_names))

            # Calculate rates for each bucket
            for metric_idx, metric in enumerate(metrics):
                rates = []
                errors_low = []
                errors_high = []

                for bucket_name in bucket_names:
                    bucket_rates = calculate_bucket_rates(data, model, dataset, buckets[bucket_name], direction='no')
                    if bucket_rates:
                        rates.append(bucket_rates[metric])
                        ci = bucket_rates[f'{metric}_ci']
                        errors_low.append(bucket_rates[metric] - ci[0])
                        errors_high.append(ci[1] - bucket_rates[metric])
                    else:
                        rates.append(0)
                        errors_low.append(0)
                        errors_high.append(0)

                offset = (metric_idx - 1.5) * bar_width
                bars = ax.bar(x + offset, rates, bar_width,
                             label=METRIC_LABELS[metric] if model_idx == 0 and dataset_idx == 0 else "",
                             color=COLORS[metric], alpha=0.8,
                             yerr=[errors_low, errors_high], capsize=2, error_kw={'linewidth': 1})

            # Formatting
            ax.set_xticks(x)
            ax.set_xticklabels(bucket_names, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, alpha=0.3, axis='y')

            # Labels
            if model_idx == 0:
                ax.set_title(dataset.replace('_', ' ').title(), fontsize=11)
            if dataset_idx == 0:
                model_short = model.split('_')[-1].replace('-it', '').replace('-Instruct', '')
                ax.set_ylabel(model_short, fontsize=10)

    # Legend
    handles = [plt.Rectangle((0,0), 1, 1, color=COLORS[m], alpha=0.8) for m in metrics]
    labels = [METRIC_LABELS[m] for m in metrics]
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle('Rollout Classification Rates by Alpha Bucket (NO direction)', fontsize=14, y=1.0)

    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
