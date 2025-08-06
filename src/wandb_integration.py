"""
W&B integration for post-hoc reasoning experiments.

This module provides a clean interface for logging experiment results to W&B,
with special focus on steering experiments and real-time monitoring.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb'")


class WandbExperimentLogger:
    """W&B logger for post-hoc reasoning experiments."""
    
    def __init__(self,
                 project: str = None,
                 entity: str = None,
                 experiment_config: Dict[str, Any] = None,
                 disabled: bool = False):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name (defaults to WANDB_PROJECT env var)
            entity: W&B entity/team name (defaults to WANDB_ENTITY env var)
            experiment_config: Experiment configuration to log
            disabled: If True, disables all W&B logging
        """
        self.disabled = disabled or not WANDB_AVAILABLE or os.environ.get("WANDB_DISABLED", "false").lower() == "true"
        self.run = None
        
        if not self.disabled:
            try:
                # Use environment variables as defaults
                project = project or os.environ.get("WANDB_PROJECT", "post-hoc-reasoning")
                entity = entity or os.environ.get("WANDB_ENTITY")
                
                # Generate run name from config
                run_name = self._generate_run_name(experiment_config)
                
                # Generate tags as per guide
                tags = []
                if "model_name" in experiment_config:
                    tags.append(experiment_config["model_name"].split("/")[-1])
                if "dataset_name" in experiment_config:
                    tags.append(experiment_config["dataset_name"])
                if "steering_method" in experiment_config:
                    tags.append(experiment_config["steering_method"])
                if "split_seed" in experiment_config:
                    tags.append(f"split_{experiment_config['split_seed']}")
                if "runner" in experiment_config:
                    tags.append(experiment_config["runner"])
                
                # Initialize W&B run
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    config=experiment_config,
                    name=run_name,
                    tags=tags,
                    reinit=True,
                    settings=wandb.Settings(start_method="thread")
                )
                
                # Define custom metrics for proper grouping
                wandb.define_metric("layer")
                wandb.define_metric("probe/*", step_metric="layer")
                wandb.define_metric("steering/example_idx")
                wandb.define_metric("steering/individual/*", step_metric="steering/example_idx")
                
                print(f"✓ W&B initialized: {self.run.url}")
                
            except Exception as e:
                print(f"Warning: Failed to initialize W&B: {e}")
                self.disabled = True
    
    def _generate_run_name(self, config: Dict[str, Any]) -> str:
        """Generate a descriptive run name from config."""
        if not config:
            return None
            
        parts = []
        if "model_name" in config:
            parts.append(config["model_name"].split("/")[-1])
        if "dataset_name" in config:
            parts.append(config["dataset_name"])
        if "steering_method" in config:
            parts.append(config["steering_method"])
            
        # Add timestamp as per guide
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts.append(timestamp)
            
        if parts:
            return "-".join(parts)
        return None
    
    def log_probe_training(self,
                          layer: int,
                          train_auc: float,
                          test_auc: float,
                          similarity_score: float = None,
                          method: str = "unknown"):
        """Log probe training results for a layer."""
        if self.disabled:
            return
            
        metrics = {
            f"probe/{method}/train_auc": train_auc,
            f"probe/{method}/test_auc": test_auc,
            "layer": layer,
        }
        
        if similarity_score is not None:
            metrics[f"probe/{method}/similarity_score"] = similarity_score
            
        wandb.log(metrics)
    
    def log_best_layer_selection(self,
                                best_layer: int,
                                best_score: float,
                                method: str = "unknown",
                                all_scores: List[float] = None):
        """Log the selected best layer for steering."""
        if self.disabled:
            return
            
        wandb.log({
            f"probe/{method}/best_layer": best_layer,
            f"probe/{method}/best_score": best_score,
        })
        
        if all_scores:
            # Create a plot of scores by layer
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(all_scores)), all_scores, 'o-')
            ax.axvline(x=best_layer, color='r', linestyle='--', label=f'Best layer: {best_layer}')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Score')
            ax.set_title(f'Layer Scores - {method}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            wandb.log({f"probe/{method}/layer_scores_plot": wandb.Image(fig)})
            plt.close()
    
    def log_steering_example(self,
                           alpha: float,
                           direction: str,
                           prompt: str,
                           original_answer: str,
                           steered_response: str,
                           steered_answer: str,
                           target_answer: str,
                           category: str,
                           example_idx: int,
                           model_name: str = None,
                           dataset_name: str = None):
        """Log a single steering example with all details."""
        if self.disabled:
            return
        
        success = (category == "success")
        parsable = (category != "unparsed")
        answer_changed = (steered_answer != original_answer)
        
        # Log metrics
        wandb.log({
            "steering/example_idx": example_idx,
            "steering/individual/alpha": alpha,
            "steering/individual/success": success,
            "steering/individual/parsable": parsable,
            "steering/individual/answer_changed": answer_changed,
        })
        
        # Log detailed example every N examples to avoid overwhelming the UI
        if example_idx % 5 == 0:  # Log every 5th example in detail
            # Truncate response for display
            display_response = steered_response
            if len(display_response) > 500:
                display_response = display_response[-500:] + "..."
            
            # Create HTML for better formatting
            html_content = f"""
            <div style="font-family: monospace; padding: 10px; background: #f5f5f5;">
                <h4>Steering Example {example_idx}</h4>
                <p><strong>Alpha:</strong> {alpha} ({direction})</p>
                <p><strong>Original → Target:</strong> {original_answer} → {target_answer}</p>
                <p><strong>Steered Answer:</strong> <span style="color: {'green' if success else 'red'}">{steered_answer}</span></p>
                <p><strong>Status:</strong> {category}</p>
                <details>
                    <summary>Show Response (last 500 chars)</summary>
                    <pre style="white-space: pre-wrap;">{display_response}</pre>
                </details>
            </div>
            """
            
            wandb.log({
                f"steering/examples/alpha_{alpha}_{direction}": wandb.Html(html_content)
            })
    
    def log_steering_summary(self,
                           alpha: float,
                           direction: str,
                           total_examples: int,
                           success_count: int,
                           failure_count: int,
                           unparsed_count: int):
        """Log summary statistics for a steering condition."""
        if self.disabled:
            return
            
        success_rate = success_count / total_examples if total_examples > 0 else 0
        failure_rate = failure_count / total_examples if total_examples > 0 else 0
        unparsed_rate = unparsed_count / total_examples if total_examples > 0 else 0
        
        wandb.log({
            f"steering/summary/alpha_{alpha}_{direction}/success_rate": success_rate,
            f"steering/summary/alpha_{alpha}_{direction}/failure_rate": failure_rate,
            f"steering/summary/alpha_{alpha}_{direction}/unparsed_rate": unparsed_rate,
            f"steering/summary/alpha_{alpha}_{direction}/total": total_examples,
        })
        
        # Alert if steering completely fails
        if success_rate == 0 and total_examples > 5:
            wandb.alert(
                title=f"Steering Failure at α={alpha}",
                text=f"No successful steering for {direction} at alpha={alpha} ({total_examples} examples)",
                level=wandb.AlertLevel.WARN
            )
    
    def log_steering_results_table(self, 
                                 results: List[Dict[str, Any]], 
                                 alpha: float,
                                 direction: str):
        """Log a table of steering results for easy viewing."""
        if self.disabled or not results:
            return
            
        # Create table data
        table_data = []
        for r in results[:20]:  # Limit to 20 rows
            table_data.append([
                alpha,
                direction,
                r.get("original_answer", "N/A"),
                r.get("new_answer", r.get("steered_answer", "N/A")),
                r.get("target_answer", "N/A"),
                r.get("category", "unknown")
            ])
        
        table = wandb.Table(
            columns=["Alpha", "Direction", "Original", "Steered", "Target", "Status"],
            data=table_data
        )
        
        wandb.log({f"steering/results_table/alpha_{alpha}_{direction}": table})
    
    def create_steering_success_plot(self, 
                                   alpha_values: List[float],
                                   yes_to_no_rates: List[float],
                                   no_to_yes_rates: List[float]):
        """Create and log a plot of steering success rates."""
        if self.disabled:
            return
            
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(alpha_values, yes_to_no_rates, 'o-', label='Yes→No', linewidth=2)
        ax.plot(alpha_values, no_to_yes_rates, 's-', label='No→Yes', linewidth=2)
        
        ax.set_xlabel('Alpha (α)', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Steering Success Rate vs Alpha', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Add horizontal line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        
        wandb.log({"steering/success_rate_plot": wandb.Image(fig)})
        plt.close()
    
    def log_experiment_summary(self, summary: Dict[str, Any]):
        """Log final experiment summary."""
        if self.disabled:
            return
            
        # Log to summary (persists in the UI)
        wandb.summary.update(summary)
        
        # Also log as regular metrics
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                wandb.log({f"summary/{key}": value})
    
    def save_artifact(self, 
                     artifact_path: str,
                     artifact_name: str,
                     artifact_type: str = "experiment_output",
                     metadata: Dict[str, Any] = None):
        """Save a file or directory as a W&B artifact."""
        if self.disabled:
            return
            
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {}
            )
            
            if os.path.isdir(artifact_path):
                artifact.add_dir(artifact_path)
            else:
                artifact.add_file(artifact_path)
                
            wandb.log_artifact(artifact)
            print(f"✓ Saved artifact: {artifact_name}")
            
        except Exception as e:
            print(f"Warning: Failed to save artifact: {e}")
    
    def finish(self):
        """Finish the W&B run."""
        if self.run:
            self.run.finish()
            print("✓ W&B run finished")


# Convenience function for quick setup
def setup_wandb(project: str = None, entity: str = None) -> bool:
    """
    Quick setup function for W&B.
    
    Returns True if setup successful, False otherwise.
    """
    if not WANDB_AVAILABLE:
        print("W&B not installed. Run: pip install wandb")
        return False
        
    # Check if already logged in
    try:
        wandb.ensure_configured()
        return True
    except:
        print("Please login to W&B:")
        print("Run: wandb login")
        print("Get your API key from: https://wandb.ai/authorize")
        return False