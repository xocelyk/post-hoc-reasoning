"""
Parallel experiment runner optimized for RTX A6000 (48GB VRAM).

This runner intelligently schedules experiments based on model size
to maximize GPU utilization without OOM errors.
"""

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from experiment_runner import EnhancedExperimentRunner
from core.config import ExperimentConfig, ExperimentRunConfig


class A6000ParallelRunner(EnhancedExperimentRunner):
    """Parallel runner optimized for RTX A6000 with 48GB VRAM."""
    
    # Model memory estimates in GB (conservative, includes overhead)
    MODEL_MEMORY_GB = {
        "gemma-2-2b": 5,
        "gemma-2-9b": 18,
        "qwen2.5-3b": 6,
        "qwen2.5-7b": 14,
        "phi-3-mini": 8,
        "llama-2-7b": 14,
        "gpt2": 2,  # If using for testing
    }
    
    MAX_VRAM_GB = 45  # Leave 3GB buffer from 48GB total
    
    def __init__(self, run_config: ExperimentRunConfig):
        super().__init__(run_config)
        self.current_vram_usage = 0.0
        self.running_models = set()
        
    def estimate_model_memory_gb(self, model_name: str) -> float:
        """Estimate VRAM usage for a model."""
        model_lower = model_name.lower()
        
        # Check against known patterns
        for pattern, memory_gb in self.MODEL_MEMORY_GB.items():
            if pattern in model_lower:
                return memory_gb
        
        # Default based on size indicators
        if "9b" in model_lower:
            return 18
        elif "7b" in model_lower:
            return 14
        elif "3b" in model_lower:
            return 6
        elif "2b" in model_lower:
            return 5
        else:
            return 8  # Conservative default
    
    def group_experiments_by_size(self) -> Dict[str, List[ExperimentConfig]]:
        """Group experiments by model memory requirements."""
        groups = {"small": [], "medium": [], "large": []}
        
        for config in self.experiment_configs:
            memory_gb = self.estimate_model_memory_gb(config.model_name)
            if memory_gb <= 6:
                groups["small"].append(config)
            elif memory_gb <= 14:
                groups["medium"].append(config)
            else:
                groups["large"].append(config)
        
        return groups
    
    def calculate_optimal_workers(self, configs: List[ExperimentConfig]) -> int:
        """Calculate optimal number of workers for a batch of experiments."""
        if not configs:
            return 1
        
        # Get max memory requirement in this batch
        max_memory = max(self.estimate_model_memory_gb(c.model_name) for c in configs)
        
        # Calculate how many can fit
        workers = int(self.MAX_VRAM_GB / max_memory)
        
        # Apply limits based on config and practical constraints
        workers = min(workers, self.run_config.max_concurrent_models)
        workers = min(workers, len(configs))  # Don't exceed number of experiments
        workers = max(workers, 1)  # At least 1
        
        return workers
    
    def run_all_experiments(self):
        """Run experiments with intelligent parallel scheduling."""
        
        if self.run_config.max_concurrent_models <= 1:
            # Fall back to sequential
            return super().run_all_experiments()
        
        self.logger.info(f"Starting parallel execution on RTX A6000 (48GB VRAM)")
        self.logger.info(f"Total experiments: {len(self.experiment_configs)}")
        
        # Initialize status tracking
        for config in self.experiment_configs:
            exp_key = f"{config.model_name}_{config.dataset_name}"
            cache = self.exp_manager.add_experiment(config)
            self.experiments_status[exp_key] = cache.get_experiment_status()
        
        # Group experiments by size for better scheduling
        size_groups = self.group_experiments_by_size()
        
        # Process each group with appropriate parallelism
        all_results = []
        
        for group_name in ["small", "medium", "large"]:
            configs = size_groups[group_name]
            if not configs:
                continue
            
            workers = self.calculate_optimal_workers(configs)
            self.logger.info(
                f"\nProcessing {len(configs)} {group_name} experiments "
                f"with {workers} parallel workers"
            )
            
            results = self.run_experiment_batch(configs, workers)
            all_results.extend(results)
        
        # Summary
        successful = sum(1 for r in all_results if r.get("success", False))
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Parallel execution complete!")
        self.logger.info(f"Success rate: {successful}/{len(all_results)}")
        
        # Call parent's final summary
        self.print_final_summary()
    
    def run_experiment_batch(
        self, 
        configs: List[ExperimentConfig], 
        max_workers: int
    ) -> List[Dict]:
        """Run a batch of experiments with specified parallelism."""
        
        results = []
        futures_to_config = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments in this batch
            for config in configs:
                future = executor.submit(self.run_single_experiment_safe, config)
                futures_to_config[future] = config
            
            # Collect results
            for future in as_completed(futures_to_config):
                config = futures_to_config[future]
                exp_key = f"{config.model_name}_{config.dataset_name}"
                
                try:
                    result = future.result(timeout=1800)  # 30 min timeout
                    results.append(result)
                    
                    if result["success"]:
                        self.experiments_status[exp_key] = result["status"]
                        self.logger.info(f"✓ Completed: {exp_key}")
                    else:
                        self.logger.error(f"✗ Failed: {exp_key} - {result.get('error')}")
                        
                except Exception as e:
                    self.logger.error(f"✗ Crashed: {exp_key} - {str(e)}")
                    results.append({"success": False, "error": str(e)})
        
        # Force cleanup between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    def run_single_experiment_safe(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Thread-safe wrapper around run_single_experiment."""
        exp_key = f"{config.model_name}_{config.dataset_name}"
        
        try:
            self.logger.info(f"Starting: {exp_key}")
            start_time = time.time()
            
            # Run the experiment
            result = super().run_single_experiment(config)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed {exp_key} in {elapsed:.1f}s")
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"OOM for {exp_key}: {str(e)}")
            # Clear cache and retry once
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(5)
            
            try:
                self.logger.info(f"Retrying {exp_key} after OOM...")
                return super().run_single_experiment(config)
            except Exception as retry_error:
                return {"success": False, "error": f"OOM retry failed: {str(retry_error)}"}
                
        except Exception as e:
            self.logger.error(f"Error in {exp_key}: {str(e)}")
            return {"success": False, "error": str(e)}


def main():
    """Example usage for parallel execution."""
    import argparse
    import yaml
    from core.config import ExperimentRunConfig
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override max concurrent if specified
    if args.max_workers:
        config_dict['max_concurrent_models'] = args.max_workers
    
    # Create config object
    run_config = ExperimentRunConfig.from_dict(config_dict)
    
    # Run with parallel runner
    runner = A6000ParallelRunner(run_config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()