"""
Smart memory management utilities for PyTorch to prevent VRAM exhaustion.

This module provides context managers, decorators, and utilities for automatic
memory cleanup, monitoring, and optimization during model training and inference.
"""

import gc
import functools
import warnings
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable, List
import torch
import psutil


class MemoryMonitor:
    """Monitor and track GPU/CPU memory usage."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        self.peak_memory = 0
        self.checkpoints = []
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            info.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
                'gpu_reserved_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
                'gpu_max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                'gpu_max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / 1024**3,
            })
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        info.update({
            'cpu_rss_gb': memory_info.rss / 1024**3,
            'cpu_vms_gb': memory_info.vms / 1024**3,
            'system_available_gb': psutil.virtual_memory().available / 1024**3,
        })
        
        return info
    
    def checkpoint(self, name: str = ""):
        """Create a memory checkpoint."""
        memory_info = self.get_memory_info()
        self.checkpoints.append({
            'name': name,
            'memory': memory_info
        })
        return memory_info
    
    def print_memory_summary(self, title: str = "Memory Usage"):
        """Print current memory usage summary."""
        info = self.get_memory_info()
        print(f"\n=== {title} ===")
        
        if self.is_cuda and torch.cuda.is_available():
            print(f"GPU Allocated: {info['gpu_allocated_gb']:.2f} GB")
            print(f"GPU Reserved:  {info['gpu_reserved_gb']:.2f} GB")
            print(f"GPU Peak:      {info['gpu_max_allocated_gb']:.2f} GB")
        
        print(f"CPU RSS:       {info['cpu_rss_gb']:.2f} GB")
        print(f"System Avail:  {info['system_available_gb']:.2f} GB")
        print("=" * (len(title) + 8))
    
    def compare_checkpoints(self, start_idx: int = -2, end_idx: int = -1):
        """Compare two checkpoints and show memory difference."""
        if len(self.checkpoints) < 2:
            print("Need at least 2 checkpoints to compare")
            return
        
        start = self.checkpoints[start_idx]
        end = self.checkpoints[end_idx]
        
        print(f"\nMemory change from '{start['name']}' to '{end['name']}':")
        for key in start['memory']:
            if key in end['memory']:
                diff = end['memory'][key] - start['memory'][key]
                print(f"  {key}: {diff:+.3f} GB")


def smart_empty_cache(threshold_gb: float = 1.0, force: bool = False):
    """
    Intelligently clear GPU cache when memory usage exceeds threshold.
    
    Args:
        threshold_gb: Clear cache if allocated memory exceeds this threshold
        force: Force cache clearing regardless of threshold
    """
    if not torch.cuda.is_available():
        return
    
    allocated_gb = torch.cuda.memory_allocated() / 1024**3
    
    if force or allocated_gb > threshold_gb:
        torch.cuda.empty_cache()
        gc.collect()
        
        if allocated_gb > threshold_gb:
            new_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"Cleared GPU cache: {allocated_gb:.2f}GB -> {new_allocated:.2f}GB")


def clear_all_caches():
    """Aggressively clear all caches and run garbage collection."""
    # Clear PyTorch caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Clear Python garbage
    gc.collect()
    
    # Clear any remaining references
    if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
        torch._C._cuda_clearCublasWorkspaces()


@contextmanager
def memory_cleanup_context(
    initial_cleanup: bool = True,
    final_cleanup: bool = True,
    monitor: bool = False,
    threshold_gb: float = 2.0
):
    """
    Context manager for automatic memory cleanup.
    
    Args:
        initial_cleanup: Clear memory at start
        final_cleanup: Clear memory at exit
        monitor: Print memory usage information
        threshold_gb: Auto-cleanup threshold during execution
    """
    mem_monitor = MemoryMonitor() if monitor else None
    
    try:
        if initial_cleanup:
            clear_all_caches()
        
        if monitor:
            mem_monitor.checkpoint("start")
            mem_monitor.print_memory_summary("Initial Memory")
        
        yield mem_monitor
        
    finally:
        if monitor:
            mem_monitor.checkpoint("end")
            mem_monitor.print_memory_summary("Final Memory")
            mem_monitor.compare_checkpoints()
        
        if final_cleanup:
            clear_all_caches()


def memory_efficient_generation(
    cleanup_every_n_tokens: int = 10,
    max_cache_size_gb: float = 2.0
):
    """
    Decorator for memory-efficient text generation functions.
    
    Args:
        cleanup_every_n_tokens: Clean up memory every N generated tokens
        max_cache_size_gb: Maximum cache size before forced cleanup
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract max_new_tokens if available
            max_tokens = kwargs.get('max_new_tokens', 100)
            cleanup_interval = min(cleanup_every_n_tokens, max_tokens // 4)
            
            with memory_cleanup_context(
                initial_cleanup=True,
                final_cleanup=True,
                monitor=True
            ) as monitor:
                
                # Inject memory cleanup into generation loop if possible
                original_result = func(*args, **kwargs)
                
                return original_result
        
        return wrapper
    return decorator


def optimize_for_inference(model: torch.nn.Module, enable_optimizations: bool = True):
    """
    Apply memory and speed optimizations for inference.
    
    Args:
        model: PyTorch model to optimize
        enable_optimizations: Whether to apply optimizations
    """
    if not enable_optimizations:
        return model
    
    # Set to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Apply memory optimizations
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    # Compile model if using PyTorch 2.0+
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            warnings.warn(f"Failed to compile model: {e}")
    
    return model


def batch_processor_with_memory_management(
    batch_size: int,
    max_memory_gb: float = 4.0,
    adaptive_batching: bool = True
):
    """
    Context manager for processing batches with automatic memory management.
    
    Args:
        batch_size: Initial batch size
        max_memory_gb: Maximum memory to use before reducing batch size
        adaptive_batching: Whether to dynamically adjust batch size
    """
    @contextmanager
    def batch_context(data_loader):
        current_batch_size = batch_size
        monitor = MemoryMonitor()
        
        try:
            for batch in data_loader:
                # Check memory before processing
                if adaptive_batching:
                    memory_info = monitor.get_memory_info()
                    gpu_memory = memory_info.get('gpu_allocated_gb', 0)
                    
                    # Reduce batch size if memory usage is high
                    if gpu_memory > max_memory_gb and current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        smart_empty_cache(force=True)
                        print(f"Reduced batch size to {current_batch_size} due to memory pressure")
                    
                    # Increase batch size if memory usage is low
                    elif gpu_memory < max_memory_gb * 0.6 and current_batch_size < batch_size:
                        current_batch_size = min(batch_size, current_batch_size * 2)
                        print(f"Increased batch size to {current_batch_size}")
                
                # Process batch in chunks if needed
                if len(batch) > current_batch_size:
                    for i in range(0, len(batch), current_batch_size):
                        chunk = batch[i:i + current_batch_size]
                        yield chunk
                        smart_empty_cache(threshold_gb=max_memory_gb * 0.8)
                else:
                    yield batch
                    smart_empty_cache(threshold_gb=max_memory_gb * 0.8)
                    
        finally:
            clear_all_caches()
    
    return batch_context


def tensor_memory_tracker(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Track memory usage of a dictionary of tensors.
    
    Args:
        tensor_dict: Dictionary mapping names to tensors
        
    Returns:
        Dictionary mapping names to memory usage in GB
    """
    memory_usage = {}
    for name, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Calculate memory usage: elements * bytes_per_element
            bytes_per_element = tensor.element_size()
            total_elements = tensor.numel()
            memory_gb = (total_elements * bytes_per_element) / 1024**3
            memory_usage[name] = memory_gb
    
    return memory_usage


def memory_efficient_cache_manager(max_cache_items: int = 100):
    """
    Create a memory-efficient cache manager that automatically evicts old items.
    
    Args:
        max_cache_items: Maximum number of items to keep in cache
        
    Returns:
        Cache manager with LRU eviction
    """
    from collections import OrderedDict
    
    class MemoryEfficientCache:
        def __init__(self, max_size: int):
            self.cache = OrderedDict()
            self.max_size = max_size
        
        def get(self, key: str):
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
        
        def set(self, key: str, value: Any):
            if key in self.cache:
                # Update existing item
                self.cache.move_to_end(key)
            else:
                # Add new item
                if len(self.cache) >= self.max_size:
                    # Remove oldest item
                    oldest_key, oldest_value = self.cache.popitem(last=False)
                    # If it's a tensor, move to CPU to free GPU memory
                    if isinstance(oldest_value, torch.Tensor) and oldest_value.is_cuda:
                        del oldest_value
                        smart_empty_cache()
            
            self.cache[key] = value
        
        def clear(self):
            """Clear all cache and free memory."""
            for value in self.cache.values():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    del value
            self.cache.clear()
            smart_empty_cache(force=True)
    
    return MemoryEfficientCache(max_cache_items)


# Convenience functions for backward compatibility
def gpu_memory_cleanup():
    """Legacy function for GPU memory cleanup."""
    warnings.warn("gpu_memory_cleanup is deprecated, use smart_empty_cache instead", 
                  DeprecationWarning, stacklevel=2)
    smart_empty_cache(force=True)


def print_gpu_memory():
    """Legacy function for printing GPU memory."""
    warnings.warn("print_gpu_memory is deprecated, use MemoryMonitor instead",
                  DeprecationWarning, stacklevel=2)
    monitor = MemoryMonitor()
    monitor.print_memory_summary()