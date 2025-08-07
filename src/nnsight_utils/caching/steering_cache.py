"""
Steering vector caching for performance optimization.

This module provides caching for pre-computed steering vectors to avoid
repeated tensor conversions and device transfers.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)


class SteeringVectorCache:
    """
    Cache for pre-converted steering vectors to avoid repeated tensor operations.
    
    This cache stores steering vectors as torch tensors on the appropriate device,
    eliminating the need for repeated numpy→torch conversion and device transfers.
    """
    
    def __init__(self, max_cache_size: int = 200):
        """
        Initialize steering vector cache.
        
        Args:
            max_cache_size: Maximum number of cached vector sets
        """
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict[int, torch.Tensor]] = {}
        self.access_order: List[str] = []
        
    def _make_cache_key(
        self,
        vectors: Dict[int, np.ndarray],
        alpha: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> str:
        """Create cache key from steering parameters."""
        # Create hash from vector data, alpha, device, and dtype
        vector_data = []
        for layer in sorted(vectors.keys()):
            vector_data.append(vectors[layer].tobytes())
        
        vector_hash = hashlib.sha256(b''.join(vector_data)).hexdigest()[:16]
        params_str = f"{alpha}_{device}_{dtype}"
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        
        return f"{vector_hash}_{params_hash}"
    
    def get(
        self,
        vectors: Dict[int, np.ndarray],
        alpha: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Get cached steering tensors.
        
        Args:
            vectors: Dictionary mapping layer indices to numpy steering vectors
            alpha: Steering strength (affects caching)
            device: Target device
            dtype: Target dtype
            
        Returns:
            Dictionary mapping layer indices to torch tensors, or None if not cached
        """
        cache_key = self._make_cache_key(vectors, alpha, device, dtype)
        
        if cache_key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            
            logger.debug(f"Steering vector cache hit for key {cache_key}")
            return self.cache[cache_key]
        
        logger.debug(f"Steering vector cache miss for key {cache_key}")
        return None
    
    def put(
        self,
        vectors: Dict[int, np.ndarray],
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
        tensors: Dict[int, torch.Tensor]
    ):
        """
        Store steering tensors in cache.
        
        Args:
            vectors: Original numpy vectors (for key generation)
            alpha: Steering strength
            device: Device tensors are on
            dtype: Tensor dtype
            tensors: Pre-converted torch tensors to cache
        """
        cache_key = self._make_cache_key(vectors, alpha, device, dtype)
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_cache_size and cache_key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            logger.debug(f"Evicted steering cache entry {oldest_key}")
        
        self.cache[cache_key] = tensors
        
        # Update access order
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
        
        logger.debug(f"Cached steering tensors for key {cache_key}")
    
    def prepare_steering_tensors(
        self,
        vectors: Dict[int, np.ndarray],
        alpha: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> Dict[int, torch.Tensor]:
        """
        Get or create cached steering tensors.
        
        Args:
            vectors: Dictionary mapping layer indices to numpy steering vectors
            alpha: Steering strength (for scaling)
            device: Target device
            dtype: Target dtype
            
        Returns:
            Dictionary mapping layer indices to prepared torch tensors
        """
        # Try to get from cache first
        cached_tensors = self.get(vectors, alpha, device, dtype)
        if cached_tensors is not None:
            return cached_tensors
        
        # Convert and cache
        tensors = {}
        for layer, vector in vectors.items():
            tensor = torch.tensor(
                alpha * vector,  # Apply alpha scaling
                device=device,
                dtype=dtype
            )
            tensors[layer] = tensor
        
        # Cache the results
        self.put(vectors, alpha, device, dtype, tensors)
        
        return tensors
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cleared steering vector cache")
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        total_bytes = 0
        for tensor_dict in self.cache.values():
            for tensor in tensor_dict.values():
                total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes


# Global cache instance
_global_steering_cache = SteeringVectorCache()


def get_cached_steering_tensors(
    vectors: Dict[int, np.ndarray],
    alpha: float,
    device: torch.device,
    dtype: torch.dtype
) -> Dict[int, torch.Tensor]:
    """
    Get cached steering tensors using the global cache.
    
    Args:
        vectors: Dictionary mapping layer indices to numpy steering vectors
        alpha: Steering strength
        device: Target device
        dtype: Target dtype
        
    Returns:
        Dictionary mapping layer indices to prepared torch tensors
    """
    return _global_steering_cache.prepare_steering_tensors(vectors, alpha, device, dtype)


def clear_global_steering_cache():
    """Clear the global steering vector cache."""
    _global_steering_cache.clear()


def get_steering_cache_stats() -> Dict:
    """Get statistics about the global steering cache."""
    return {
        'size': _global_steering_cache.size(),
        'memory_usage_bytes': _global_steering_cache.memory_usage(),
        'memory_usage_mb': _global_steering_cache.memory_usage() / (1024 * 1024)
    }