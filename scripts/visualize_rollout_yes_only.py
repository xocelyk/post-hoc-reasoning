#!/usr/bin/env python3
"""
Visualize rollout classification results for YES steering direction only.
Creates a grid plot of confabulation, non-entailment, hallucination rates vs alpha.
Uses sliding window moving average for smoothing.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Configuration
CACHE_DIR = Path("final_cache/cache/rollout_classification_v2")
STEERING_CACHE_DIR = Path("final_cache/cache/experiments")
OUTPUT_PATH = "figs/rollout_classification_v2_combined.png"
WINDOW_SIZE = 200  # Number of examples in sliding window
MIN_WINDOW_SIZE = 20  # Minimum window size at boundaries

# Models and datasets to include
MODELS = [
    "google_gemma-2-2b-it",
    "google_gemma-2-9b-it",
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
]

DATASETS = [
    "anachronisms",
    "logical_deduction",
    "social_chemistry",
    "sports_understanding",
]

# Display name mappings
DATASET_DISPLAY = {
    "logical_deduction": "Logical Deduction",
    "sports_understanding": "Sports Understanding",
    "social_chemistry": "Social Chemistry",
    "anachronisms": "Anachronisms",
}

MODEL_DISPLAY = {
    "google_gemma-2-2b-it": "Gemma 2 2B",
    "google_gemma-2-9b-it": "Gemma 2 9B",
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen 2.5 1.5B",
    "Qwen_Qwen2.5-3B-Instruct": "Qwen 2.5 3B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
}

# Colors (more saturated palette)
COLORS = {
    'confabulation': '#F9A825',      # orange
    'non_entailment': '#2196F3',     # blue
    'hallucination': '#E63946',      # red
    'sound': '#4CAF50',              # green
}


def load_parsed_counts(steering_cache_dir: Path):
    """Load total parsed sample counts from steering experiment pkl files.

    Returns:
        dict: (model, dataset, direction, alpha_abs) -> count of valid parses
    """
    counts = defaultdict(int)

    for model_dir in steering_cache_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            # Find steering pkl files (may be nested in split/hash dirs)
            for pkl_file in dataset_dir.rglob("steering_alpha_*.pkl"):
                # Parse filename: steering_alpha_{alpha}_{direction}.pkl
                name = pkl_file.stem  # steering_alpha_-10_yes
                parts = name.split('_')
                # parts = ['steering', 'alpha', '-10', 'yes']
                try:
                    alpha = int(parts[2])
                    direction = parts[3]
                    alpha_abs = abs(alpha)

                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)

                    valid_count = sum(1 for item in data if item.get('is_valid_parse', False))
                    key = (model, dataset, direction, alpha_abs)
                    counts[key] += valid_count
                except (IndexError, ValueError) as e:
                    continue

    return dict(counts)


def load_rollout_classifications(cache_dir: Path):
    """Load all rollout classification results from JSONL files.

    Returns a dict: (model, dataset, direction) -> list of (alpha, record) tuples
    """
    if not cache_dir.exists():
        print(f"Cache not found at {cache_dir}")
        return {}

    # results[(model, dataset, direction)] = list of (alpha, record) tuples
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

                    # Use new classification_label field if available (v2), otherwise use old fields (v1)
                    if 'classification_label' in data:
                        label = data['classification_label']
                        record = {
                            'confabulation': label == 'confabulation',
                            'non_entailment': label == 'non_entailment',
                            'hallucination': label == 'hallucination',
                            'sound': label == 'sound',
                            'refuse_flag': data.get('refuse_flag', False),
                        }
                    else:
                        record = {
                            'confabulation': data.get('confabulation', False),
                            'non_entailment': data.get('non_entailment', False),
                            'hallucination': data.get('hallucination', False),
                            'sound': data.get('sound', False),
                            'refuse_flag': data.get('refuse_flag', False),
                        }

                    # Include all parsed samples (including sound) for denominator
                    key = (model, dataset, direction)
                    results[key].append((alpha_abs, record))
                except Exception as e:
                    continue

    return results


def wilson_ci(p, n):
    """Compute Wilson score 90% confidence interval."""
    if n == 0:
        return 0, 0
    z = 1.645  # 90% CI
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def compute_sliding_window_rates(data, model, dataset, window_size, min_window_size, direction=None):
    """Compute rates using sliding window over examples sorted by alpha.

    Each example is treated as an individual point. Examples are sorted by alpha,
    then a window slides over them computing relative rates within each window.
    Window can shrink at boundaries down to min_window_size, but not below the
    count of examples at the lowest/highest alpha bucket.

    Args:
        data: dict of (model, dataset, direction) -> list of (alpha, record)
        model: model name
        dataset: dataset name
        window_size: target number of examples in each window
        min_window_size: minimum window size at boundaries
        direction: 'yes', 'no', or None for both

    Returns:
        alphas: array of mean alpha values for each window position
        rates: dict with rate arrays and CI bounds for each category
    """
    # Collect all examples for this model/dataset/direction
    examples = []
    for (m, d, dir_), records in data.items():
        if m == model and d == dataset:
            if direction is None or dir_ == direction:
                examples.extend(records)

    if len(examples) < min_window_size:
        return None, None

    # Sort by alpha
    examples = sorted(examples, key=lambda x: x[0])
    n_examples = len(examples)


    # Phase 1: Start at left, grow window from min to max size
    # Phase 2: Slide full-size window through middle
    # Phase 3: Shrink window from max to min at right edge

    windows = []

    # Phase 1: left anchored at 0, right grows from min_window_size to window_size
    for right in range(min_window_size, min(window_size + 1, n_examples + 1)):
        windows.append((0, right))

    # Phase 2: slide full window (if we have enough examples)
    if n_examples >= window_size:
        for left in range(1, n_examples - window_size + 1):
            windows.append((left, left + window_size))

    # Phase 3: right anchored at end, left shrinks
    start_left = max(1, n_examples - window_size + 1)
    for left in range(start_left, n_examples - min_window_size + 1):
        windows.append((left, n_examples))

    alphas = []
    confab_rates, confab_lo, confab_hi = [], [], []
    non_ent_rates, non_ent_lo, non_ent_hi = [], [], []
    hall_rates, hall_lo, hall_hi = [], [], []

    for left, right in windows:
        window = examples[left:right]
        window_alphas = [ex[0] for ex in window]
        window_records = [ex[1] for ex in window]

        # Mean alpha for this window
        mean_alpha = np.mean(window_alphas)
        alphas.append(mean_alpha)

        # Count each type (excluding sound from numerator)
        confab_count = sum(1 for r in window_records if r['confabulation'])
        non_ent_count = sum(1 for r in window_records if r['non_entailment'])
        hall_count = sum(1 for r in window_records if r['hallucination'])

        # Normalize to relative proportions (sum to 1)
        total_unfaithful = confab_count + non_ent_count + hall_count
        n = total_unfaithful  # Use unfaithful count for CI calculation

        if total_unfaithful > 0:
            confab = confab_count / total_unfaithful
            non_ent = non_ent_count / total_unfaithful
            hall = hall_count / total_unfaithful
        else:
            confab, non_ent, hall = 0, 0, 0
            n = 1  # Avoid division issues

        confab_rates.append(confab)
        ci = wilson_ci(confab, n)
        confab_lo.append(ci[0])
        confab_hi.append(ci[1])

        non_ent_rates.append(non_ent)
        ci = wilson_ci(non_ent, n)
        non_ent_lo.append(ci[0])
        non_ent_hi.append(ci[1])

        hall_rates.append(hall)
        ci = wilson_ci(hall, n)
        hall_lo.append(ci[0])
        hall_hi.append(ci[1])

    return np.array(alphas), {
        'confabulation': np.array(confab_rates),
        'confabulation_lo': np.array(confab_lo),
        'confabulation_hi': np.array(confab_hi),
        'non_entailment': np.array(non_ent_rates),
        'non_entailment_lo': np.array(non_ent_lo),
        'non_entailment_hi': np.array(non_ent_hi),
        'hallucination': np.array(hall_rates),
        'hallucination_lo': np.array(hall_lo),
        'hallucination_hi': np.array(hall_hi),
    }


def setup_style():
    """Set up matplotlib style to match visualize_results.ipynb."""
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = '#EEEEEE'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Lato', 'DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'


def main():
    print(f"Loading classification data from {CACHE_DIR}...")
    data = load_rollout_classifications(CACHE_DIR)
    print(f"Loaded {len(data)} unique (model, dataset, direction) combinations")
    print(f"Total classified samples: {sum(len(v) for v in data.values())}")

    # Set up style
    setup_style()

    # Create figure
    n_models = len(MODELS)
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(n_models, n_datasets, figsize=(16, 12), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.08, top=0.92, bottom=0.08, left=0.08, right=0.95)

    for model_idx, model in enumerate(MODELS):
        for dataset_idx, dataset in enumerate(DATASETS):
            ax = axes[model_idx, dataset_idx]

            # Compute sliding window rates (YES + NO combined)
            alphas, rates = compute_sliding_window_rates(data, model, dataset, WINDOW_SIZE, MIN_WINDOW_SIZE, direction=None)

            if alphas is not None:
                # Plot lines with confidence interval bands
                ax.fill_between(alphas, rates['confabulation_lo'], rates['confabulation_hi'],
                                color=COLORS['confabulation'], alpha=0.2)
                ax.plot(alphas, rates['confabulation'], '-', color=COLORS['confabulation'],
                        label='Confabulation', linewidth=2)

                ax.fill_between(alphas, rates['non_entailment_lo'], rates['non_entailment_hi'],
                                color=COLORS['non_entailment'], alpha=0.2)
                ax.plot(alphas, rates['non_entailment'], '-', color=COLORS['non_entailment'],
                        label='Non-entailment', linewidth=2)

                ax.fill_between(alphas, rates['hallucination_lo'], rates['hallucination_hi'],
                                color=COLORS['hallucination'], alpha=0.2)
                ax.plot(alphas, rates['hallucination'], '-', color=COLORS['hallucination'],
                        label='Hallucination', linewidth=2)

            # Compute and plot raw rates per alpha value
            examples = []
            for (m, d, dir_), records in data.items():
                if m == model and d == dataset:
                    examples.extend(records)

            if examples:
                # Group by alpha
                by_alpha = defaultdict(list)
                for alpha, record in examples:
                    by_alpha[alpha].append(record)

                raw_alphas = []
                raw_confab = []
                raw_non_ent = []
                raw_hall = []
                raw_counts = []

                for alpha in sorted(by_alpha.keys()):
                    records = by_alpha[alpha]
                    confab_count = sum(1 for r in records if r['confabulation'])
                    non_ent_count = sum(1 for r in records if r['non_entailment'])
                    hall_count = sum(1 for r in records if r['hallucination'])
                    total = confab_count + non_ent_count + hall_count

                    if total > 0:
                        raw_alphas.append(alpha)
                        raw_confab.append(confab_count / total)
                        raw_non_ent.append(non_ent_count / total)
                        raw_hall.append(hall_count / total)
                        raw_counts.append(total)

                # Plot scatter points (size proportional to sample count)
                sizes = [c / 2 for c in raw_counts]  # Scale down for reasonable size
                ax.scatter(raw_alphas, raw_confab, color=COLORS['confabulation'], s=sizes, alpha=0.6, zorder=5)
                ax.scatter(raw_alphas, raw_non_ent, color=COLORS['non_entailment'], s=sizes, alpha=0.6, zorder=5)
                ax.scatter(raw_alphas, raw_hall, color=COLORS['hallucination'], s=sizes, alpha=0.6, zorder=5)

            # Formatting
            ax.set_xlim(-0.9, 20.9)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(np.arange(0, 21, 5))
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.grid(True)
            ax.tick_params(axis='y', length=0)

            # Spines
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(1.0)

            # Labels
            if model_idx == 0:
                ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=15, fontweight='bold', pad=4)
            if dataset_idx == 0:
                ax.set_ylabel(MODEL_DISPLAY.get(model, model), fontsize=14, fontweight='bold')
            if model_idx == n_models - 1:
                ax.set_xlabel(r'|$\alpha$|', fontsize=13)
                ax.tick_params(axis='x', labelsize=13)

            # Legend (only first subplot)
            if model_idx == 0 and dataset_idx == 0:
                ax.legend(loc='upper right', fontsize=9, frameon=True, handletextpad=0.5)

    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
