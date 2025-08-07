#!/usr/bin/env python3
"""
Web-based cache viewer for post-hoc reasoning experiments.
Can be accessed via SSH port forwarding: ssh -L 8888:localhost:8888 your-server
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Global cache directory
CACHE_DIR = 'cache'

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Post-Hoc Reasoning Cache Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .clickable { cursor: pointer; color: #0066cc; }
        .clickable:hover { text-decoration: underline; }
        .metric { font-weight: bold; }
        .good { color: green; }
        .bad { color: red; }
        .warning { color: orange; }
        .section { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
        .example { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }
        .correct { background-color: #e6ffe6; }
        .incorrect { background-color: #ffe6e6; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
        .filter-section { background-color: #f0f0f0; padding: 10px; margin: 20px 0; }
        .plot { margin: 20px 0; text-align: center; }
    </style>
    <script>
        function showExperiment(path) {
            window.location.href = '/experiment?path=' + encodeURIComponent(path);
        }
        
        function showGenerations(path, split) {
            window.location.href = '/generations?path=' + encodeURIComponent(path) + '&split=' + split;
        }
        
        function filterTable() {
            var modelFilter = document.getElementById('modelFilter').value.toLowerCase();
            var datasetFilter = document.getElementById('datasetFilter').value.toLowerCase();
            var table = document.getElementById('expTable');
            var rows = table.getElementsByTagName('tr');
            
            for (var i = 1; i < rows.length; i++) {
                var model = rows[i].cells[0].textContent.toLowerCase();
                var dataset = rows[i].cells[1].textContent.toLowerCase();
                
                if ((modelFilter === '' || model.includes(modelFilter)) &&
                    (datasetFilter === '' || dataset.includes(datasetFilter))) {
                    rows[i].style.display = '';
                } else {
                    rows[i].style.display = 'none';
                }
            }
        }
    </script>
</head>
<body>
    <h1>Post-Hoc Reasoning Cache Viewer</h1>
    <p>Cache directory: {{ cache_dir }}</p>
    
    {% if page == 'index' %}
        <div class="filter-section">
            <label>Filter by Model: <input type="text" id="modelFilter" onkeyup="filterTable()" placeholder="Enter model name..."></label>
            <label style="margin-left: 20px;">Filter by Dataset: <input type="text" id="datasetFilter" onkeyup="filterTable()" placeholder="Enter dataset name..."></label>
        </div>
        
        <h2>Experiments ({{ experiments|length }} total)</h2>
        <table id="expTable">
            <tr>
                <th>Model</th>
                <th>Dataset</th>
                <th>Train Acc</th>
                <th>Test Acc</th>
                <th>Probe AUC</th>
                <th>Data</th>
                <th>Modified</th>
                <th>Actions</th>
            </tr>
            {% for exp in experiments %}
            <tr>
                <td>{{ exp.model_short }}</td>
                <td>{{ exp.dataset }}</td>
                <td class="metric {% if exp.train_acc > 80 %}good{% elif exp.train_acc < 60 %}bad{% endif %}">
                    {{ "%.1f%%" % exp.train_acc if exp.train_acc else "N/A" }}
                </td>
                <td class="metric {% if exp.test_acc > 80 %}good{% elif exp.test_acc < 60 %}bad{% endif %}">
                    {{ "%.1f%%" % exp.test_acc if exp.test_acc else "N/A" }}
                </td>
                <td class="metric {% if exp.probe_auc and exp.probe_auc > 0.7 %}good{% elif exp.probe_auc and exp.probe_auc < 0.4 %}bad{% endif %}">
                    {{ "%.3f" % exp.probe_auc if exp.probe_auc else "N/A" }}
                </td>
                <td>
                    {% if exp.has_train %}<span title="Training data">T</span>{% endif %}
                    {% if exp.has_test %}<span title="Test data">E</span>{% endif %}
                    {% if exp.has_probes %}<span title="Probe results">P</span>{% endif %}
                    {% if exp.has_steering %}<span title="Steering results">S</span>{% endif %}
                </td>
                <td>{{ exp.modified }}</td>
                <td><span class="clickable" onclick="showExperiment('{{ exp.path }}')">View</span></td>
            </tr>
            {% endfor %}
        </table>
        
        <div class="plot">
            <h3>Accuracy Overview</h3>
            <img src="/plot/accuracy_overview" alt="Accuracy Overview">
        </div>
        
        <div class="plot">
            <h3>Probe AUC Distribution</h3>
            <img src="/plot/auc_distribution" alt="AUC Distribution">
        </div>
        
    {% elif page == 'experiment' %}
        <p><a href="/">← Back to experiments</a></p>
        
        <h2>{{ experiment.model }} - {{ experiment.dataset }}</h2>
        <div class="section">
            <h3>Experiment Details</h3>
            <p><strong>Path:</strong> {{ experiment.path }}</p>
            <p><strong>ID:</strong> {{ experiment.id }}</p>
            <p><strong>Modified:</strong> {{ experiment.modified }}</p>
        </div>
        
        {% if accuracies %}
        <div class="section">
            <h3>Accuracy Results</h3>
            <table>
                <tr>
                    <th>Split</th>
                    <th>Accuracy</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Label Distribution</th>
                    <th>Actions</th>
                </tr>
                {% for split, data in accuracies.items() %}
                <tr>
                    <td>{{ split.capitalize() }}</td>
                    <td class="metric {% if data.accuracy > 80 %}good{% elif data.accuracy < 60 %}bad{% endif %}">
                        {{ "%.1f%%" % data.accuracy }}
                    </td>
                    <td>{{ data.correct }}</td>
                    <td>{{ data.total }}</td>
                    <td>{{ data.labels }}</td>
                    <td><span class="clickable" onclick="showGenerations('{{ experiment.path }}', '{{ split }}')">View Examples</span></td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        {% if probe_info %}
        <div class="section">
            <h3>Probe Results</h3>
            <p><strong>Max AUC:</strong> <span class="metric {% if probe_info.max_auc > 0.7 %}good{% elif probe_info.max_auc < 0.4 %}bad{% endif %}">{{ "%.4f" % probe_info.max_auc }}</span></p>
            <p><strong>Mean AUC:</strong> {{ "%.4f" % probe_info.mean_auc }}</p>
            <p><strong>Best Layer:</strong> {{ probe_info.best_layer }}</p>
            <p><strong>Number of Layers:</strong> {{ probe_info.num_layers }}</p>
            
            <div class="plot">
                <img src="/plot/probe_layers?path={{ experiment.path|urlencode }}" alt="Probe AUC by Layer">
            </div>
        </div>
        {% endif %}
        
        {% if probe_training_analysis %}
        <div class="section {% if probe_training_analysis.warning %}warning{% endif %}">
            <h3>Probe Training Data Analysis</h3>
            <p><strong>Total training examples:</strong> {{ probe_training_analysis.total }}</p>
            <p><strong>Correct predictions (used for probes):</strong> {{ probe_training_analysis.correct }} ({{ "%.1f%%" % probe_training_analysis.correct_pct }})</p>
            <p><strong>Label distribution (all):</strong> {{ probe_training_analysis.all_labels }}</p>
            <p><strong>Label distribution (correct only):</strong> {{ probe_training_analysis.correct_labels }}</p>
            
            {% if probe_training_analysis.warning %}
            <p class="warning"><strong>⚠️ WARNING:</strong> {{ probe_training_analysis.warning }}</p>
            {% endif %}
            
            <h4>Per-class accuracy:</h4>
            <ul>
            {% for label, stats in probe_training_analysis.per_class.items() %}
                <li>'{{ label }}': {{ stats.correct }}/{{ stats.total }} ({{ "%.1f%%" % stats.accuracy }})</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        {% if steering_info %}
        <div class="section">
            <h3>Steering Results</h3>
            <p><strong>Alpha values tested:</strong> {{ steering_info.alphas }}</p>
            <p><strong>Total conditions:</strong> {{ steering_info.total }}</p>
            
            <div class="plot">
                <img src="/plot/steering_results?path={{ experiment.path|urlencode }}" alt="Steering Success Rates">
            </div>
        </div>
        {% endif %}
        
    {% elif page == 'generations' %}
        <p><a href="/experiment?path={{ exp_path|urlencode }}">← Back to experiment</a></p>
        
        <h2>{{ split.capitalize() }} Generations</h2>
        <p><strong>Total examples:</strong> {{ total }}</p>
        <p><strong>Showing:</strong> {{ showing }} {% if filter_incorrect %}(incorrect only){% endif %}</p>
        
        <div class="filter-section">
            <form method="get" action="/generations">
                <input type="hidden" name="path" value="{{ exp_path }}">
                <input type="hidden" name="split" value="{{ split }}">
                <label>Number of examples: <input type="number" name="num" value="{{ num }}" min="1" max="100"></label>
                <label style="margin-left: 20px;">
                    <input type="checkbox" name="incorrect" {% if filter_incorrect %}checked{% endif %}> Show incorrect only
                </label>
                <button type="submit">Update</button>
            </form>
        </div>
        
        {% for i, gen in examples %}
        <div class="example {% if gen.is_correct %}correct{% else %}incorrect{% endif %}">
            <h4>Example {{ i }} {% if gen.is_correct %}✓{% else %}✗{% endif %}</h4>
            
            {% if gen.question %}
            <p><strong>Question:</strong> {{ gen.question }}</p>
            {% elif gen.input %}
            <p><strong>Input:</strong> {{ gen.input }}</p>
            {% endif %}
            
            <p><strong>Correct Answer:</strong> <span class="metric">{{ gen.correct_answer }}</span></p>
            <p><strong>Predicted Answer:</strong> <span class="metric {% if gen.is_correct %}good{% else %}bad{% endif %}">{{ gen.pred_answer }}</span></p>
            
            {% if gen.response %}
            <details>
                <summary>Show Response</summary>
                <pre>{{ gen.response }}</pre>
            </details>
            {% endif %}
            
            {% if gen.category %}
            <p><strong>Category:</strong> {{ gen.category }}</p>
            {% endif %}
        </div>
        {% endfor %}
        
    {% endif %}
</body>
</html>
'''

def load_experiments(cache_dir):
    """Load all experiments from cache."""
    experiments = []
    experiments_dir = os.path.join(cache_dir, 'experiments')
    
    if not os.path.exists(experiments_dir):
        return experiments
    
    for root, dirs, files in os.walk(experiments_dir):
        if 'data' in dirs or 'metadata' in dirs:
            exp_info = {
                'path': root,
                'model': 'unknown',
                'dataset': 'unknown',
                'id': 'unknown',
                'has_train': os.path.exists(os.path.join(root, 'data', 'train_generations.pkl')),
                'has_test': os.path.exists(os.path.join(root, 'data', 'test_generations.pkl')),
                'has_probes': os.path.exists(os.path.join(root, 'probes')),
                'has_steering': os.path.exists(os.path.join(root, 'steering'))
            }
            
            # Parse path
            path_parts = Path(root).parts
            for i, part in enumerate(path_parts):
                if part == 'experiments' and i + 2 < len(path_parts):
                    exp_info['model'] = path_parts[i + 1]
                    exp_info['dataset'] = path_parts[i + 2]
                    if i + 4 < len(path_parts):
                        exp_info['id'] = path_parts[i + 4]
                    break
            
            # Model short names
            model_names = {
                'deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B': 'DeepSeek-1.5B',
                'google_gemma-2-2b-it': 'Gemma-2B',
                'google_gemma-2-9b-it': 'Gemma-9B'
            }
            exp_info['model_short'] = model_names.get(exp_info['model'], exp_info['model'].split('/')[-1][:20])
            
            # Get modification time
            try:
                mtime = os.path.getmtime(root)
                exp_info['modified'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            except:
                exp_info['modified'] = 'unknown'
            
            # Load accuracies
            for split in ['train', 'test']:
                if exp_info[f'has_{split}']:
                    gen_path = os.path.join(root, 'data', f'{split}_generations.pkl')
                    try:
                        with open(gen_path, 'rb') as f:
                            gens = pickle.load(f)
                        correct = sum(1 for g in gens if g.get('pred_answer') == g.get('correct_answer'))
                        exp_info[f'{split}_acc'] = correct / len(gens) * 100 if gens else 0
                    except:
                        exp_info[f'{split}_acc'] = None
            
            # Load probe AUC
            if exp_info['has_probes']:
                auc_paths = [
                    os.path.join(root, 'probes', 'caa-single-layer', 'auc_scores.json'),
                    os.path.join(root, 'probes', 'auc_scores.json')
                ]
                
                for auc_path in auc_paths:
                    if os.path.exists(auc_path):
                        try:
                            with open(auc_path, 'r') as f:
                                scores = json.load(f)
                            if isinstance(scores, dict):
                                scores = list(scores.values())
                            if scores:
                                exp_info['probe_auc'] = max(scores)
                                break
                        except:
                            pass
            
            experiments.append(exp_info)
    
    return sorted(experiments, key=lambda x: (x['model'], x['dataset']))

@app.route('/')
def index():
    experiments = load_experiments(CACHE_DIR)
    return render_template_string(HTML_TEMPLATE, 
                                page='index',
                                cache_dir=CACHE_DIR,
                                experiments=experiments)

@app.route('/experiment')
def experiment_detail():
    exp_path = request.args.get('path', '')
    if not exp_path or not os.path.exists(exp_path):
        return "Experiment not found", 404
    
    # Parse experiment info
    path_parts = Path(exp_path).parts
    exp_info = {
        'path': exp_path,
        'model': 'unknown',
        'dataset': 'unknown',
        'id': 'unknown'
    }
    
    for i, part in enumerate(path_parts):
        if part == 'experiments' and i + 2 < len(path_parts):
            exp_info['model'] = path_parts[i + 1]
            exp_info['dataset'] = path_parts[i + 2]
            if i + 4 < len(path_parts):
                exp_info['id'] = path_parts[i + 4]
            break
    
    try:
        mtime = os.path.getmtime(exp_path)
        exp_info['modified'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    except:
        exp_info['modified'] = 'unknown'
    
    # Load accuracies
    accuracies = {}
    for split in ['train', 'test']:
        gen_path = os.path.join(exp_path, 'data', f'{split}_generations.pkl')
        if os.path.exists(gen_path):
            try:
                with open(gen_path, 'rb') as f:
                    gens = pickle.load(f)
                correct = sum(1 for g in gens if g.get('pred_answer') == g.get('correct_answer'))
                labels = Counter(g.get('correct_answer', 'N/A') for g in gens)
                accuracies[split] = {
                    'accuracy': correct / len(gens) * 100 if gens else 0,
                    'correct': correct,
                    'total': len(gens),
                    'labels': dict(labels)
                }
            except:
                pass
    
    # Load probe info
    probe_info = None
    auc_paths = [
        os.path.join(exp_path, 'probes', 'caa-single-layer', 'auc_scores.json'),
        os.path.join(exp_path, 'probes', 'auc_scores.json')
    ]
    
    for auc_path in auc_paths:
        if os.path.exists(auc_path):
            try:
                with open(auc_path, 'r') as f:
                    scores = json.load(f)
                if isinstance(scores, dict):
                    scores = list(scores.values())
                if scores:
                    probe_info = {
                        'max_auc': max(scores),
                        'mean_auc': np.mean(scores),
                        'best_layer': np.argmax(scores),
                        'num_layers': len(scores),
                        'scores': scores
                    }
                    break
            except:
                pass
    
    # Analyze probe training data
    probe_training_analysis = None
    train_path = os.path.join(exp_path, 'data', 'train_generations.pkl')
    if os.path.exists(train_path):
        try:
            with open(train_path, 'rb') as f:
                train_gens = pickle.load(f)
            
            correct_preds = [g for g in train_gens if g.get('pred_answer') == g.get('correct_answer')]
            all_labels = Counter(g.get('correct_answer', 'N/A') for g in train_gens)
            correct_labels = Counter(g.get('correct_answer', 'N/A') for g in correct_preds)
            
            per_class = {}
            for label in all_labels:
                total = sum(1 for g in train_gens if g.get('correct_answer') == label)
                correct = sum(1 for g in train_gens if g.get('correct_answer') == label and g.get('pred_answer') == label)
                per_class[label] = {
                    'total': total,
                    'correct': correct,
                    'accuracy': correct / total * 100 if total > 0 else 0
                }
            
            probe_training_analysis = {
                'total': len(train_gens),
                'correct': len(correct_preds),
                'correct_pct': len(correct_preds) / len(train_gens) * 100 if train_gens else 0,
                'all_labels': dict(all_labels),
                'correct_labels': dict(correct_labels),
                'per_class': per_class
            }
            
            if len(correct_labels) < 2:
                probe_training_analysis['warning'] = f"Probe training data has only {len(correct_labels)} class(es)! This will cause probe training to fail."
        except:
            pass
    
    # Load steering info
    steering_info = None
    steering_dir = os.path.join(exp_path, 'steering')
    if os.path.exists(steering_dir):
        try:
            steering_files = [f for f in os.listdir(steering_dir) if f.endswith('.pkl')]
            alphas = set()
            for f in steering_files:
                if f.startswith('steering_alpha_'):
                    try:
                        alpha = float(f.split('_')[2])
                        alphas.add(alpha)
                    except:
                        pass
            
            steering_info = {
                'alphas': sorted(list(alphas)),
                'total': len(steering_files)
            }
        except:
            pass
    
    return render_template_string(HTML_TEMPLATE,
                                page='experiment',
                                experiment=exp_info,
                                accuracies=accuracies,
                                probe_info=probe_info,
                                probe_training_analysis=probe_training_analysis,
                                steering_info=steering_info)

@app.route('/generations')
def view_generations():
    exp_path = request.args.get('path', '')
    split = request.args.get('split', 'train')
    num = int(request.args.get('num', 10))
    filter_incorrect = 'incorrect' in request.args
    
    if not exp_path or not os.path.exists(exp_path):
        return "Experiment not found", 404
    
    gen_path = os.path.join(exp_path, 'data', f'{split}_generations.pkl')
    if not os.path.exists(gen_path):
        return f"No {split} generations found", 404
    
    try:
        with open(gen_path, 'rb') as f:
            gens = pickle.load(f)
    except Exception as e:
        return f"Error loading generations: {e}", 500
    
    if filter_incorrect:
        gens = [g for g in gens if g.get('pred_answer') != g.get('correct_answer')]
    
    examples = []
    for i, gen in enumerate(gens[:num]):
        example = {
            'correct_answer': gen.get('correct_answer', 'N/A'),
            'pred_answer': gen.get('pred_answer', 'N/A'),
            'is_correct': gen.get('pred_answer') == gen.get('correct_answer'),
            'question': gen.get('question', '')[:500],
            'input': gen.get('input', '')[:500],
            'response': gen.get('response', gen.get('generated_text', ''))[:1000],
            'category': gen.get('category', '')
        }
        examples.append((i + 1, example))
    
    return render_template_string(HTML_TEMPLATE,
                                page='generations',
                                exp_path=exp_path,
                                split=split,
                                total=len(gens),
                                showing=len(examples),
                                num=num,
                                filter_incorrect=filter_incorrect,
                                examples=examples)

@app.route('/plot/<plot_type>')
def generate_plot(plot_type):
    """Generate various plots."""
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'accuracy_overview':
        experiments = load_experiments(CACHE_DIR)
        df = pd.DataFrame(experiments)
        
        if 'train_acc' in df.columns and 'test_acc' in df.columns:
            df_acc = df[df['train_acc'].notna() | df['test_acc'].notna()]
            
            x = np.arange(len(df_acc))
            width = 0.35
            
            plt.bar(x - width/2, df_acc['train_acc'].fillna(0), width, label='Train', alpha=0.8)
            plt.bar(x + width/2, df_acc['test_acc'].fillna(0), width, label='Test', alpha=0.8)
            
            plt.xlabel('Experiment')
            plt.ylabel('Accuracy (%)')
            plt.title('Train vs Test Accuracy')
            plt.xticks(x, [f"{row['model_short']}\n{row['dataset'][:10]}" for _, row in df_acc.iterrows()], 
                      rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
    
    elif plot_type == 'auc_distribution':
        experiments = load_experiments(CACHE_DIR)
        aucs = [exp['probe_auc'] for exp in experiments if 'probe_auc' in exp and exp['probe_auc']]
        
        if aucs:
            plt.hist(aucs, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=0.5, color='r', linestyle='--', label='Random baseline')
            plt.xlabel('Probe AUC')
            plt.ylabel('Count')
            plt.title('Distribution of Probe AUC Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    elif plot_type == 'probe_layers':
        exp_path = request.args.get('path', '')
        if exp_path:
            auc_paths = [
                os.path.join(exp_path, 'probes', 'caa-single-layer', 'auc_scores.json'),
                os.path.join(exp_path, 'probes', 'auc_scores.json')
            ]
            
            for auc_path in auc_paths:
                if os.path.exists(auc_path):
                    try:
                        with open(auc_path, 'r') as f:
                            scores = json.load(f)
                        if isinstance(scores, dict):
                            layers = list(map(int, scores.keys()))
                            values = [scores[str(l)] for l in layers]
                        else:
                            layers = list(range(len(scores)))
                            values = scores
                        
                        plt.plot(layers, values, marker='o', markersize=8)
                        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
                        plt.xlabel('Layer')
                        plt.ylabel('AUC Score')
                        plt.title('Probe AUC by Layer')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        break
                    except:
                        pass
    
    elif plot_type == 'steering_results':
        exp_path = request.args.get('path', '')
        if exp_path:
            steering_dir = os.path.join(exp_path, 'steering')
            if os.path.exists(steering_dir):
                # Collect steering data
                steering_data = defaultdict(lambda: {'yes': [], 'no': []})
                
                for filename in os.listdir(steering_dir):
                    if filename.startswith('steering_alpha_') and filename.endswith('.pkl'):
                        parts = filename.replace('steering_alpha_', '').replace('.pkl', '').rsplit('_', 1)
                        if len(parts) == 2:
                            try:
                                alpha = float(parts[0])
                                direction = parts[1]
                                
                                with open(os.path.join(steering_dir, filename), 'rb') as f:
                                    results = pickle.load(f)
                                
                                successes = sum(1 for r in results if r.get('success', False))
                                total = len(results)
                                success_rate = successes / total if total > 0 else 0
                                
                                steering_data[alpha][direction] = success_rate
                            except:
                                pass
                
                if steering_data:
                    alphas = sorted(steering_data.keys())
                    yes_rates = [steering_data[a].get('yes', 0) for a in alphas]
                    no_rates = [steering_data[a].get('no', 0) for a in alphas]
                    
                    plt.plot(alphas, yes_rates, marker='o', label='Yes steering', markersize=8)
                    plt.plot(alphas, no_rates, marker='s', label='No steering', markersize=8)
                    plt.xlabel('Alpha')
                    plt.ylabel('Success Rate')
                    plt.title('Steering Success Rate by Alpha')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.ylim(-0.05, 1.05)
    
    # Convert plot to base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

def main():
    parser = argparse.ArgumentParser(description='Web-based cache viewer for post-hoc reasoning experiments')
    parser.add_argument('--cache', default='cache', help='Cache directory path')
    parser.add_argument('--port', type=int, default=8888, help='Port to run server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (use 0.0.0.0 for external access)')
    args = parser.parse_args()
    
    global CACHE_DIR
    CACHE_DIR = args.cache
    
    print(f"Starting cache viewer on http://{args.host}:{args.port}")
    print(f"Cache directory: {CACHE_DIR}")
    print("\nTo access from your local machine via SSH:")
    print(f"  ssh -L {args.port}:localhost:{args.port} your-server")
    print(f"  Then open http://localhost:{args.port} in your browser")
    
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == '__main__':
    main()