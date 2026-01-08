"""
Logging utilities for improved console output with color support.
"""
import logging
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.text import Text


# Define custom theme for steering results
custom_theme = Theme({
    "success": "green bold",
    "failure": "red bold", 
    "unparsed": "magenta bold",
    "info": "cyan",
    "warning": "yellow",
    "header": "bold blue",
    "progress": "dim white",
})

# Global console instance
console = Console(theme=custom_theme)


class SteeringLogFormatter(logging.Formatter):
    """Custom formatter with simplified timestamp."""
    
    def __init__(self):
        super().__init__()
        self.datefmt = "%H:%M:%S"
        
    def format(self, record):
        # Use simple time format
        record.asctime = datetime.fromtimestamp(record.created).strftime(self.datefmt)
        
        # Simplified format without logger name
        if record.levelname == "INFO":
            return f"[{record.asctime}] {record.getMessage()}"
        else:
            return f"[{record.asctime}] {record.levelname}: {record.getMessage()}"


def setup_logging(log_file: Optional[str] = None, verbose: bool = True):
    """
    Setup logging with rich console handler and optional file handler.
    
    Args:
        log_file: Optional path to log file
        verbose: Whether to show verbose output
    """
    # Remove existing handlers
    logger = logging.getLogger()
    logger.handlers = []
    
    # Set level
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    # Console handler with rich
    console_handler = RichHandler(
        console=console,
        show_time=True,
        omit_repeated_times=False,
        show_level=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_width=100,
        tracebacks_extra_lines=3,
    )
    console_handler.setFormatter(SteeringLogFormatter())
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    
    return logger


def log_steering_result(
    example_idx: int,
    total_examples: int,
    alpha: float,
    direction: str,
    original_answer: str,
    target_answer: str,
    steered_answer: str,
    response_text: str,
    category: str,
    model_name: str,
    dataset_name: str,
    success_count: int = 0,
    failure_count: int = 0,
    unparsed_count: int = 0,
):
    """
    Log a steering result with nice formatting and color coding.
    """
    # Clear previous line for cleaner output
    console.print()
    
    # Header with model/dataset info
    console.rule(
        f"[header]{model_name.split('/')[-1]} on {dataset_name}[/header]",
        style="dim",
    )
    
    # Alpha and direction info
    direction_display = "yes→no" if direction == "yes_to_no" else "no→yes"
    console.print(
        f"[header]Alpha:[/header] {alpha:+.1f} | "
        f"[header]Direction:[/header] {direction_display} | "
        f"[header]Example:[/header] {example_idx}/{total_examples}"
    )
    console.print()
    
    # Original vs target
    console.print(f"[info]Original:[/info] {original_answer} → [info]Target:[/info] {target_answer}")
    
    # Clean up response text - handle list display
    if isinstance(response_text, list):
        if len(response_text) == 1:
            response_text = response_text[0]
        else:
            response_text = str(response_text)
    
    # Truncate long responses
    # if len(response_text) > 200:
    #     response_text = response_text[:200] + "..."
    
    console.print(f"[dim]Response:[/dim] {response_text}")
    console.print()
    
    # Result with color coding
    if category == "success":
        console.print(f"[success]✓ SUCCESS[/success] → Parsed: {steered_answer}")
    elif category == "failure":
        console.print(f"[failure]✗ FAILURE[/failure] → Parsed: {steered_answer}")
    else:
        console.print("[unparsed]⚠ UNPARSED[/unparsed]")
    
    # Progress summary
    total_processed = success_count + failure_count + unparsed_count
    if total_processed > 0:
        success_rate = success_count / total_processed
        console.print(
            f"\n[progress]Progress: {total_processed}/{total_examples} | "
            f"Success: {success_count} ({success_rate:.0%}) | "
            f"Failed: {failure_count} | "
            f"Unparsed: {unparsed_count}[/progress]"
        )
    
    console.rule(style="dim")


def log_phase_start(phase: str, details: str = ""):
    """Log the start of a new phase with styling."""
    console.print()
    console.rule(f"[header]{phase}[/header]", style="blue")
    if details:
        console.print(f"[info]{details}[/info]")
    console.print()


def log_batch_progress(current: int, total: int, phase: str = "Processing"):
    """Log simple batch progress without verbose output."""
    console.print(f"[progress]{phase}: {current}/{total}[/progress]", end="\r")