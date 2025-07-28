#!/usr/bin/env python3
"""
Command-line interface for running post-hoc reasoning experiments.

This script provides a comprehensive CLI for running probe and steering experiments
with advanced caching, visualization, and resume functionality.
"""

import argparse
import os
import sys
from typing import List, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import ConfigLoader, ExperimentRunConfig, save_default_configs
from experiment_runner import EnhancedExperimentRunner


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run post-hoc reasoning experiments with probe training and steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a configuration file
  python run_experiments.py --config configs/basic.yaml

  # Resume incomplete experiments
  python run_experiments.py --resume

  # Create example configuration files
  python run_experiments.py --create-configs

  # Run with specific models and datasets (override config)
  python run_experiments.py --config configs/basic.yaml --models google/gemma-2-9b-it --datasets sports_understanding

  # Run in non-interactive mode
  python run_experiments.py --config configs/basic.yaml --no-interactive

  # List existing experiments
  python run_experiments.py --list-experiments
        """,
    )

    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete experiments",
    )
    group.add_argument(
        "--create-configs",
        action="store_true",
        help="Create example configuration files",
    )
    group.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all existing experiments",
    )

    # Configuration overrides
    parser.add_argument(
        "--models",
        nargs="+",
        help="Override models from config (space-separated list)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Override datasets from config (space-separated list)",
    )
    parser.add_argument(
        "--alpha-range",
        nargs="+",
        type=float,
        help="Override alpha range for steering (space-separated floats)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        help="Override train size for all datasets",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        help="Override test size for all datasets",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Override split seed for all datasets",
    )

    # Runtime options
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for caching results (default: cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (not recommended)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive visualization",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Maximum number of concurrent model experiments",
    )
    parser.add_argument(
        "--resume-ids",
        nargs="+",
        help="Specific experiment IDs to resume (space-separated)",
    )

    # Output options
    parser.add_argument(
        "--output-summary",
        type=str,
        help="Save experiment summary to CSV file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def apply_overrides(
    config: ExperimentRunConfig, args: argparse.Namespace
) -> ExperimentRunConfig:
    """Apply command-line overrides to configuration."""
    # Model overrides
    if args.models:
        from config import ModelConfig

        config.models = [ModelConfig(name=model) for model in args.models]

    # Dataset overrides
    if args.datasets:
        from config import DatasetConfig

        new_datasets = []
        for dataset in args.datasets:
            dataset_config = DatasetConfig(name=dataset)
            if args.train_size:
                dataset_config.train_size = args.train_size
            if args.test_size:
                dataset_config.test_size = args.test_size
            if args.split_seed:
                dataset_config.split_seed = args.split_seed
            new_datasets.append(dataset_config)
        config.datasets = new_datasets
    else:
        # Apply size/seed overrides to existing datasets
        if args.train_size or args.test_size or args.split_seed:
            for dataset in config.datasets:
                if args.train_size:
                    dataset.train_size = args.train_size
                if args.test_size:
                    dataset.test_size = args.test_size
                if args.split_seed:
                    dataset.split_seed = args.split_seed

    # Steering overrides
    if args.alpha_range:
        config.steering.alpha_range = args.alpha_range

    # Runtime overrides
    config.cache_dir = args.cache_dir
    config.use_cache = not args.no_cache
    config.interactive = not args.no_interactive

    if args.max_concurrent:
        config.max_concurrent_models = args.max_concurrent

    return config


def list_experiments(cache_dir: str = "cache"):
    """List all existing experiments."""
    from rich.console import Console
    from rich.table import Table

    from cache_manager import ExperimentManager

    console = Console()
    exp_manager = ExperimentManager(cache_dir)

    try:
        summary_df = exp_manager.get_experiments_summary()

        if summary_df.empty:
            console.print("[yellow]No experiments found.[/yellow]")
            return

        # Create Rich table
        table = Table(
            title="Existing Experiments", show_header=True, header_style="bold magenta"
        )

        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("Dataset", style="blue")
        table.add_column("Train/Test", justify="center")
        table.add_column("Data", justify="center")
        table.add_column("Probes", justify="center")
        table.add_column("Steering", justify="center")
        table.add_column("Size (MB)", justify="right")

        for _, row in summary_df.iterrows():
            data_status = "✓" if row["has_data"] else "○"
            probe_status = "✓" if row["has_probes"] else "○"

            table.add_row(
                row["experiment_id"][:8] + "...",  # Truncate ID
                row["model"].split("/")[-1],  # Show only model name
                row["dataset"],
                f"{row['train_size']}/{row['test_size']}",
                data_status,
                probe_status,
                row["steering_progress"],
                f"{row['cache_size_mb']:.1f}",
            )

        console.print(table)
        console.print(f"\nTotal experiments: {len(summary_df)}")

    except Exception as e:
        console.print(f"[red]Error listing experiments: {str(e)}[/red]")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle special modes
    if args.create_configs:
        save_default_configs()
        print("✓ Example configuration files created in 'configs/' directory")
        return

    if args.list_experiments:
        list_experiments(args.cache_dir)
        return

    # Resume mode
    if args.resume:
        print("🔄 Resuming incomplete experiments...")

        # Create a minimal config for resume functionality
        from config import create_default_config

        config = create_default_config()
        config.cache_dir = args.cache_dir
        config.interactive = not args.no_interactive

        runner = EnhancedExperimentRunner(config)
        runner.resume_experiments(args.resume_ids)

        # Save summary if requested
        if args.output_summary:
            summary_df = runner.get_results_summary()
            summary_df.to_csv(args.output_summary, index=False)
            print(f"✓ Summary saved to {args.output_summary}")

        return

    # Config mode
    if not args.config:
        parser.error(
            "Configuration file is required when not using --resume, --create-configs, or --list-experiments"
        )

    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("Use --create-configs to create example configuration files")
        return

    try:
        # Load configuration
        print(f"📖 Loading configuration from {args.config}")
        config = ConfigLoader.load_experiment_config(args.config)

        # Apply command-line overrides
        config = apply_overrides(config, args)

        print("✓ Configuration loaded and validated")
        print(f"  Models: {[m.name for m in config.models]}")
        print(f"  Datasets: {[d.name for d in config.datasets]}")
        print(f"  Alpha range: {config.steering.alpha_range}")
        print(f"  Cache directory: {config.cache_dir}")

        # Create and run experiments
        print("\n🚀 Starting experiments...")
        runner = EnhancedExperimentRunner(config)
        runner.run_all_experiments()

        # Save summary if requested
        if args.output_summary:
            summary_df = runner.get_results_summary()
            summary_df.to_csv(args.output_summary, index=False)
            print(f"\n✓ Summary saved to {args.output_summary}")

        print("\n🎉 All experiments completed!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user")
        print("Use --resume to continue from where you left off")

    except Exception as e:
        print(f"\n❌ Error running experiments: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
