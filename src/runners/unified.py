"""
Unified experiment runner that delegates to appropriate backend implementation.

This runner automatically selects the correct implementation based on the
backend configuration of the models.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from core.config import ExperimentRunConfig
from experiment_runner import EnhancedExperimentRunner
from nnsight_experiment_runner import NNsightExperimentRunner
from backends.nnsight.unified.experiment_runner import UnifiedExperimentRunner as NNsightUnifiedRunner


class UnifiedExperimentRunner:
    """Unified runner that delegates to the appropriate backend implementation."""
    
    def __init__(self, run_config: ExperimentRunConfig):
        """Initialize the unified runner.
        
        Args:
            run_config: Experiment run configuration
        """
        self.run_config = run_config
        
        # Determine which backend to use based on model configurations
        backend = self._determine_backend()
        
        # Create the appropriate runner
        if backend == "nnsight_unified":
            # Use the unified nnsight runner for nnsight models
            self.runner = NNsightUnifiedRunner(run_config)
        elif backend == "nnsight":
            # Use the standard nnsight runner
            self.runner = NNsightExperimentRunner(run_config)
        else:
            # Use the transformer_lens runner
            self.runner = EnhancedExperimentRunner(run_config)
        
        self.backend = backend
        print(f"🔧 Using {backend} backend implementation")
    
    def _determine_backend(self) -> str:
        """Determine which backend to use based on model configurations.
        
        Returns:
            Backend name: "transformer_lens", "nnsight", or "nnsight_unified"
        """
        # Check if any model explicitly requires nnsight
        has_nnsight = False
        has_transformer_lens = False
        
        for model in self.run_config.models:
            if model.backend == "nnsight":
                has_nnsight = True
            elif model.backend == "transformer_lens":
                has_transformer_lens = True
            elif model.backend == "auto":
                # Auto-detect based on model name
                if any(pattern in model.name.lower() for pattern in ["deepseek", "mistral", "mixtral"]):
                    has_nnsight = True
                else:
                    has_transformer_lens = True
        
        # If mixed backends, prefer nnsight for compatibility
        if has_nnsight and has_transformer_lens:
            print("⚠️  Warning: Mixed backends detected. Using NNsight for compatibility.")
            return "nnsight_unified"
        elif has_nnsight:
            # Check if we should use the unified nnsight runner
            # (it has better memory management and features)
            return "nnsight_unified"
        else:
            return "transformer_lens"
    
    def run_all_experiments(self):
        """Run all experiments using the selected backend."""
        return self.runner.run_all_experiments()
    
    def run_single_experiment(self, config) -> Dict[str, Any]:
        """Run a single experiment."""
        return self.runner.run_single_experiment(config)
    
    def resume_experiments(self, experiment_ids: Optional[List[str]] = None):
        """Resume incomplete experiments."""
        return self.runner.resume_experiments(experiment_ids)
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of results."""
        return self.runner.get_results_summary()
    
    def __getattr__(self, name):
        """Delegate any other method calls to the underlying runner."""
        return getattr(self.runner, name)