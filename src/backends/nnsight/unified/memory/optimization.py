"""
Memory optimization utilities for NNsight operations.

This module provides functions for managing GPU memory and optimizing
batch processing to prevent OOM errors.
"""

import gc
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch


def smart_empty_cache():
    """
    Intelligently empty GPU cache based on available backends.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Also run garbage collection
    gc.collect()


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory statistics
    """
    info = {"backend": "cpu", "allocated": 0, "cached": 0, "total": 0}
    
    if torch.cuda.is_available():
        info["backend"] = "cuda"
        info["allocated"] = torch.cuda.memory_allocated()
        info["cached"] = torch.cuda.memory_reserved()
        try:
            info["total"] = torch.cuda.get_device_properties(0).total_memory
        except:
            info["total"] = 0
    elif torch.backends.mps.is_available():
        info["backend"] = "mps"
        info["allocated"] = torch.mps.current_allocated_memory()
        try:
            info["cached"] = torch.mps.driver_allocated_memory()
        except:
            info["cached"] = 0
    
    return info


@contextmanager
def memory_cleanup_context(
    initial_cleanup: bool = True,
    final_cleanup: bool = True,
    monitor: bool = False
) -> Iterator[None]:
    """
    Context manager for automatic memory cleanup.
    
    Args:
        initial_cleanup: Whether to clean up at start
        final_cleanup: Whether to clean up at end
        monitor: Whether to print memory usage
    """
    if initial_cleanup:
        smart_empty_cache()
    
    if monitor:
        initial_info = get_gpu_memory_info()
        print(f"Initial GPU memory: {initial_info['allocated'] / 1e9:.2f}GB allocated")
    
    try:
        yield
    finally:
        if final_cleanup:
            smart_empty_cache()
        
        if monitor:
            final_info = get_gpu_memory_info()
            print(f"Final GPU memory: {final_info['allocated'] / 1e9:.2f}GB allocated")


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    dtype: torch.dtype = torch.float32
) -> float:
    """
    Estimate memory usage for activation extraction.
    
    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length
        d_model: Model dimension
        n_layers: Number of layers
        dtype: Tensor data type
        
    Returns:
        Estimated memory usage in bytes
    """
    # Bytes per element based on dtype
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 4)
    
    # Estimate: activations + gradients + overhead
    base_usage = batch_size * seq_len * d_model * n_layers * bytes_per_element
    overhead_factor = 3  # Account for intermediate tensors, gradients, etc.
    
    return base_usage * overhead_factor


def suggest_batch_size(
    seq_len: int,
    d_model: int,
    n_layers: int,
    max_memory_gb: float = 8.0,
    safety_factor: float = 0.7
) -> int:
    """
    Suggest appropriate batch size based on memory constraints.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        n_layers: Number of layers
        max_memory_gb: Maximum memory to use in GB
        safety_factor: Safety factor (0.7 = use 70% of available memory)
        
    Returns:
        Suggested batch size
    """
    max_memory_bytes = max_memory_gb * 1e9 * safety_factor
    
    # Try different batch sizes
    for batch_size in [32, 16, 8, 4, 2, 1]:
        estimated_usage = estimate_memory_usage(
            batch_size, seq_len, d_model, n_layers
        )
        if estimated_usage <= max_memory_bytes:
            return batch_size
    
    return 1  # Fallback to batch size 1


def batch_with_memory_limit(
    items: list,
    batch_size: Optional[int] = None,
    max_memory_gb: float = 8.0,
    seq_len: Optional[int] = None,
    d_model: Optional[int] = None,
    n_layers: Optional[int] = None
) -> Iterator[list]:
    """
    Yield batches with automatic memory-based sizing.
    
    Args:
        items: List of items to batch
        batch_size: Fixed batch size (None = auto-determine)
        max_memory_gb: Maximum memory to use
        seq_len: Sequence length (for auto-sizing)
        d_model: Model dimension (for auto-sizing)  
        n_layers: Number of layers (for auto-sizing)
        
    Yields:
        Batches of items
    """
    if batch_size is None:
        if all(x is not None for x in [seq_len, d_model, n_layers]):
            batch_size = suggest_batch_size(
                seq_len, d_model, n_layers, max_memory_gb
            )
        else:
            batch_size = 4  # Default fallback
    
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage of a function.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function that prints memory usage
    """
    def wrapper(*args, **kwargs):
        initial_info = get_gpu_memory_info()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            final_info = get_gpu_memory_info()
            
            allocated_diff = (final_info['allocated'] - initial_info['allocated']) / 1e9
            print(f"{func.__name__} memory change: {allocated_diff:+.2f}GB")
    
    return wrapper


def cleanup_tensors(*tensors):
    """
    Explicitly clean up tensors and free memory.
    
    Args:
        *tensors: Tensors to clean up
    """
    for tensor in tensors:
        if tensor is not None:
            del tensor
    
    smart_empty_cache()


@contextmanager
def temporary_tensors(*tensors) -> Iterator[None]:
    """
    Context manager to automatically clean up temporary tensors.
    
    Args:
        *tensors: Tensors to clean up when done
    """
    try:
        yield
    finally:
        cleanup_tensors(*tensors)


def optimize_for_memory():
    """
    Apply global settings to optimize for memory usage.
    """
    # Disable benchmarking to save memory
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        pass
    
    # Set memory fraction if using CUDA
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)
        except:
            pass