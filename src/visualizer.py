import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotext as plt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


class RealTimeVisualizer:
    """Real-time visualization system for experiment progress and results."""

    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )

        # Data storage for plotting
        self.auc_scores: Dict[str, Dict[str, List[float]]] = (
            {}
        )  # {model_dataset: {layer: [auc]}}
        self.steering_results: Dict[str, Dict[float, Dict[str, float]]] = (
            {}
        )  # {model_dataset: {alpha: {label: success_rate}}}

        # Layout for live display
        self.layout = Layout()
        self.setup_layout()

        # Progress tracking
        self.current_tasks = {}

    def setup_layout(self):
        """Setup the main layout for live display."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        self.layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        self.layout["left"].split_column(
            Layout(name="auc_plot", ratio=2), Layout(name="steering_plot", ratio=2)
        )

        self.layout["right"].split_column(
            Layout(name="progress", ratio=1), Layout(name="status", ratio=1)
        )

    def create_auc_plot(self, model_dataset: str, auc_scores: List[float]) -> str:
        """Create ASCII plot for AUC scores across layers."""
        if not auc_scores:
            return "No AUC data available"

        plt.clf()
        plt.theme("dark")
        plt.title(f"Probe AUC Scores - {model_dataset}")
        plt.xlabel("Layer")
        plt.ylabel("AUROC")

        layers = list(range(len(auc_scores)))
        plt.plot(layers, auc_scores, marker="braille", color="cyan")
        plt.ylim(0, 1)

        # Add horizontal line at 0.5 (random performance)
        plt.hline(0.5, color="red")

        # Add horizontal line at 0.9 (good performance threshold)
        plt.hline(0.9, color="green")

        plt.plotsize(50, 15)
        plot_str = plt.build()
        plt.clf()
        return plot_str

    def create_steering_plot(
        self, model_dataset: str, steering_data: Dict[float, Dict[str, float]]
    ) -> str:
        """Create ASCII plot for steering success rates across alpha values."""
        if not steering_data:
            return "No steering data available"

        plt.clf()
        plt.theme("dark")
        plt.title(f"Steering Success Rates - {model_dataset}")
        plt.xlabel("Alpha (Steering Coefficient)")
        plt.ylabel("Success Rate")

        alphas = sorted(steering_data.keys())

        # Plot for 'yes' label (negative alpha values, steering to 'no')
        yes_rates = []
        no_rates = []

        for alpha in alphas:
            alpha_data = steering_data[alpha]
            yes_rates.append(alpha_data.get("yes", 0))
            no_rates.append(alpha_data.get("no", 0))

        if yes_rates:
            plt.plot(alphas, yes_rates, marker="braille", color="blue", label="Yes→No")
        if no_rates:
            plt.plot(alphas, no_rates, marker="braille", color="red", label="No→Yes")

        plt.ylim(0, 1)
        plt.plotsize(50, 15)
        plot_str = plt.build()
        plt.clf()
        return plot_str

    def update_auc_scores(
        self, model_name: str, dataset_name: str, layer_scores: List[float]
    ):
        """Update AUC scores for a model/dataset combination."""
        key = f"{model_name}_{dataset_name}"
        if key not in self.auc_scores:
            self.auc_scores[key] = {}

        # Store scores by layer
        for layer, score in enumerate(layer_scores):
            if f"layer_{layer}" not in self.auc_scores[key]:
                self.auc_scores[key][f"layer_{layer}"] = []
            self.auc_scores[key][f"layer_{layer}"].append(score)

    def update_steering_results(
        self,
        model_name: str,
        dataset_name: str,
        alpha: float,
        label: str,
        success_rate: float,
    ):
        """Update steering results for a model/dataset/alpha/label combination."""
        key = f"{model_name}_{dataset_name}"
        if key not in self.steering_results:
            self.steering_results[key] = {}
        if alpha not in self.steering_results[key]:
            self.steering_results[key][alpha] = {}

        self.steering_results[key][alpha][label] = success_rate

    def create_experiment_status_table(
        self, experiments_status: Dict[str, Dict]
    ) -> Table:
        """Create a table showing status of all experiments."""
        table = Table(
            title="Experiment Status", show_header=True, header_style="bold magenta"
        )

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Dataset", style="green")
        table.add_column("Data", justify="center")
        table.add_column("Probes", justify="center")
        table.add_column("Steering", justify="center")
        table.add_column("Progress", justify="right")

        for exp_key, status in experiments_status.items():
            model, dataset = exp_key.split("_", 1)

            # Status indicators
            data_status = "✓" if status.get("has_data", False) else "○"
            probe_status = "✓" if status.get("has_probes", False) else "○"
            steering_status = "✓" if status.get("steering_complete", False) else "○"

            # Progress calculation
            completed_steps = sum(
                [
                    status.get("has_data", False),
                    status.get("has_probes", False),
                    status.get("steering_complete", False),
                ]
            )
            progress_pct = f"{completed_steps}/3"

            table.add_row(
                model, dataset, data_status, probe_status, steering_status, progress_pct
            )

        return table

    def create_progress_panel(self) -> Panel:
        """Create a panel showing current progress."""
        return Panel(self.progress, title="Current Progress", border_style="blue")

    def update_display(self, experiments_status: Dict[str, Dict]):
        """Update the entire display with current data."""
        # Header
        self.layout["header"].update(
            Panel(
                Text(
                    "Post-Hoc Reasoning Experiments",
                    style="bold white",
                    justify="center",
                ),
                style="blue",
            )
        )

        # AUC Plot (show the first available experiment)
        if self.auc_scores:
            first_exp = list(self.auc_scores.keys())[0]
            if self.auc_scores[first_exp]:
                # Get the latest scores across all layers
                latest_scores = []
                max_layers = (
                    max(int(k.split("_")[1]) for k in self.auc_scores[first_exp].keys())
                    + 1
                )
                for layer in range(max_layers):
                    layer_key = f"layer_{layer}"
                    if layer_key in self.auc_scores[first_exp]:
                        latest_scores.append(self.auc_scores[first_exp][layer_key][-1])
                    else:
                        latest_scores.append(0.5)  # Default to random performance

                auc_plot = self.create_auc_plot(first_exp, latest_scores)
            else:
                auc_plot = "No AUC data available"
        else:
            auc_plot = "No AUC data available"

        self.layout["auc_plot"].update(
            Panel(auc_plot, title="Probe AUC Scores", border_style="cyan")
        )

        # Steering Plot (show the first available experiment)
        if self.steering_results:
            first_exp = list(self.steering_results.keys())[0]
            steering_plot = self.create_steering_plot(
                first_exp, self.steering_results[first_exp]
            )
        else:
            steering_plot = "No steering data available"

        self.layout["steering_plot"].update(
            Panel(steering_plot, title="Steering Results", border_style="yellow")
        )

        # Progress
        self.layout["progress"].update(self.create_progress_panel())

        # Status table
        status_table = self.create_experiment_status_table(experiments_status)
        self.layout["status"].update(
            Panel(status_table, title="Experiments", border_style="green")
        )

        # Footer
        self.layout["footer"].update(
            Panel(
                Text("Press Ctrl+C to interrupt", style="dim white", justify="center"),
                style="red",
            )
        )

    def add_task(self, description: str, total: int) -> int:
        """Add a new progress task and return its ID."""
        task_id = self.progress.add_task(description, total=total)
        self.current_tasks[description] = task_id
        return task_id

    def update_task(self, task_id: int, advance: int = 1, description: str = None):
        """Update progress for a task."""
        self.progress.update(task_id, advance=advance, description=description)

    def complete_task(self, task_id: int):
        """Mark a task as completed."""
        self.progress.update(task_id, completed=True)

    def print_summary(self, experiments_status: Dict[str, Dict]):
        """Print a final summary of all experiments."""
        self.console.print("\n" + "=" * 80)
        self.console.print("EXPERIMENT SUMMARY", style="bold white", justify="center")
        self.console.print("=" * 80)

        for exp_key, status in experiments_status.items():
            model, dataset = exp_key.split("_", 1)

            self.console.print(f"\n[cyan]{model}[/cyan] - [green]{dataset}[/green]")
            self.console.print(
                f"  Data Generation: {'✓' if status.get('has_data') else '✗'}"
            )
            self.console.print(
                f"  Probe Training: {'✓' if status.get('has_probes') else '✗'}"
            )
            self.console.print(
                f"  Steering: {'✓' if status.get('steering_complete') else '✗'}"
            )

            # Print AUC scores if available
            if exp_key in self.auc_scores and self.auc_scores[exp_key]:
                max_layers = (
                    max(int(k.split("_")[1]) for k in self.auc_scores[exp_key].keys())
                    + 1
                )
                latest_scores = []
                for layer in range(max_layers):
                    layer_key = f"layer_{layer}"
                    if layer_key in self.auc_scores[exp_key]:
                        latest_scores.append(self.auc_scores[exp_key][layer_key][-1])

                if latest_scores:
                    best_auc = max(latest_scores)
                    best_layer = latest_scores.index(best_auc)
                    self.console.print(
                        f"  Best AUC: {best_auc:.3f} (layer {best_layer})"
                    )

            # Print steering success rates if available
            if exp_key in self.steering_results:
                steering_data = self.steering_results[exp_key]
                if steering_data:
                    avg_success_rates = []
                    for alpha_data in steering_data.values():
                        for rate in alpha_data.values():
                            avg_success_rates.append(rate)

                    if avg_success_rates:
                        avg_success = np.mean(avg_success_rates)
                        self.console.print(f"  Avg Steering Success: {avg_success:.3f}")


class SimpleProgressTracker:
    """Simple progress tracker for non-interactive use."""

    def __init__(self):
        self.console = Console()
        self.current_step = ""
        self.step_count = 0
        self.total_steps = 0

    def start_experiment(self, model_name: str, dataset_name: str, total_steps: int):
        """Start tracking a new experiment."""
        self.total_steps = total_steps
        self.step_count = 0
        self.console.print(
            f"\n[bold blue]Starting experiment: {model_name} on {dataset_name}[/bold blue]"
        )
        self.console.print(f"Total steps: {total_steps}")

    def update_step(self, step_name: str, step_number: int = None):
        """Update the current step."""
        if step_number is not None:
            self.step_count = step_number
        else:
            self.step_count += 1

        self.current_step = step_name
        progress = f"[{self.step_count}/{self.total_steps}]"
        self.console.print(f"{progress} {step_name}")

    def log_auc_scores(
        self, model_name: str, dataset_name: str, auc_scores: List[float]
    ):
        """Log AUC scores."""
        if auc_scores:
            best_auc = max(auc_scores)
            best_layer = auc_scores.index(best_auc)
            self.console.print(f"  Best AUC: {best_auc:.3f} at layer {best_layer}")

    def log_steering_result(
        self, alpha: float, label: str, success_rate: float, total: int
    ):
        """Log steering result."""
        self.console.print(
            f"  Alpha {alpha:+.1f} ({label}): {success_rate:.2f} success rate ({int(success_rate * total)}/{total})"
        )

    def complete_experiment(self, model_name: str, dataset_name: str):
        """Mark experiment as completed."""
        self.console.print(
            f"[bold green]✓ Completed: {model_name} on {dataset_name}[/bold green]\n"
        )


def create_visualizer(
    interactive: bool = True,
) -> RealTimeVisualizer | SimpleProgressTracker:
    """Factory function to create appropriate visualizer based on environment."""
    if interactive:
        try:
            # Try to create RealTimeVisualizer, fall back to simple if it fails
            return RealTimeVisualizer()
        except Exception as e:
            print(f"Warning: Could not create interactive visualizer ({e}), using simple tracker")
            return SimpleProgressTracker()
    else:
        return SimpleProgressTracker()
