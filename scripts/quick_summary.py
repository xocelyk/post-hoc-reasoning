#!/usr/bin/env python3
"""
Quick one-liner to check cache status.
"""

import os
import sys

def quick_check():
    cache_dirs = []
    for d in ['cache', 'results_cache']:
        if os.path.exists(d):
            cache_dirs.append(d)
    
    if not cache_dirs:
        print("❌ No cache found")
        return
    
    for cache_dir in cache_dirs:
        print(f"📁 {cache_dir}:")
        exp_dir = os.path.join(cache_dir, 'experiments')
        if os.path.exists(exp_dir):
            count = 0
            for root, dirs, files in os.walk(exp_dir):
                if 'data' in dirs:
                    count += 1
            print(f"   🔍 {count} experiment directories")
            
            # Quick check for completed experiments
            completed = 0
            for root, dirs, files in os.walk(exp_dir):
                if 'probes' in dirs and 'steering' in dirs:
                    probe_path = os.path.join(root, 'probes', 'caa-single-layer')
                    if os.path.exists(os.path.join(probe_path, 'auc_scores.json')):
                        completed += 1
            print(f"   ✅ {completed} completed experiments")
        else:
            print("   ❌ No experiments directory")

if __name__ == "__main__":
    quick_check()