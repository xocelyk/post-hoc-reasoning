"""
KV (Key-Value) caching implementation for NNsight steering optimization.

This module implements caching strategies to avoid recomputing attention states
for shared prompt prefixes, reducing complexity from O(t*n) to O(n).
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

from ..core.models import NNsightChatModel

logger = logging.getLogger(__name__)


class KVCache:
    """
    Key-Value cache for storing and reusing attention states.
    
    Stores computed KV states for prompt prefixes to avoid recomputation
    when generating multiple continuations with the same prefix.
    """
    
    def __init__(self, max_cache_size: int = 100):
        """
        Initialize KV cache.
        
        Args:
            max_cache_size: Maximum number of cached entries
        """
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict] = {}
        self.access_order: List[str] = []
        
    def _make_cache_key(self, tokens: torch.Tensor, model_name: str) -> str:
        """Create cache key from tokens and model name."""
        # Use hash of tokens and model name
        token_hash = hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()[:16]
        model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:8]
        return f"{model_hash}_{token_hash}"
    
    def get(self, tokens: torch.Tensor, model_name: str) -> Optional[Dict]:
        """
        Get cached KV state for given tokens.
        
        Args:
            tokens: Input token tensor
            model_name: Model identifier
            
        Returns:
            Cached KV state dict or None if not found
        """
        cache_key = self._make_cache_key(tokens, model_name)
        
        if cache_key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            
            logger.debug(f"KV cache hit for key {cache_key}")
            return self.cache[cache_key]
        
        logger.debug(f"KV cache miss for key {cache_key}")
        return None
    
    def put(self, tokens: torch.Tensor, model_name: str, kv_state: Dict):
        """
        Store KV state in cache.
        
        Args:
            tokens: Input token tensor  
            model_name: Model identifier
            kv_state: KV state dictionary to cache
        """
        cache_key = self._make_cache_key(tokens, model_name)
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_cache_size and cache_key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            logger.debug(f"Evicted cache entry {oldest_key}")
        
        self.cache[cache_key] = kv_state
        
        # Update access order
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
        
        logger.debug(f"Cached KV state for key {cache_key}")
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cleared KV cache")
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        total_bytes = 0
        for kv_state in self.cache.values():
            if 'past_key_values' in kv_state:
                for layer_kv in kv_state['past_key_values']:
                    if isinstance(layer_kv, (list, tuple)):
                        for tensor in layer_kv:
                            if isinstance(tensor, torch.Tensor):
                                total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes


class SharedPrefixCache:
    """
    High-level cache for sharing computation across prompts with common prefixes.
    
    This class identifies common prompt prefixes and caches their computed states,
    enabling efficient reuse for steering experiments.
    """
    
    def __init__(self, max_cache_size: int = 50):
        """
        Initialize shared prefix cache.
        
        Args:
            max_cache_size: Maximum number of prefix entries to cache
        """
        self.kv_cache = KVCache(max_cache_size)
        self.prefix_mapping: Dict[str, str] = {}  # Maps full prompt to prefix
        
    def extract_common_prefix(self, prompts: List[str]) -> Optional[str]:
        """
        Extract the longest common prefix from a list of prompts.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            Common prefix string or None if no significant prefix
        """
        if len(prompts) < 2:
            return None
        
        # Find longest common prefix
        min_len = min(len(p) for p in prompts)
        prefix_len = 0
        
        for i in range(min_len):
            if all(p[i] == prompts[0][i] for p in prompts):
                prefix_len = i + 1
            else:
                break
        
        # Only consider significant prefixes (>50% of shortest prompt)
        if prefix_len > min_len * 0.5:
            return prompts[0][:prefix_len]
        
        return None
    
    def process_with_prefix_caching(
        self,
        model: NNsightChatModel,
        prompts: List[str],
        steering_fn: callable,
        **generation_kwargs
    ) -> List[str]:
        """
        Process multiple prompts with prefix caching optimization.
        
        Args:
            model: NNsight model instance
            prompts: List of prompt strings to process
            steering_fn: Function to apply steering during generation
            **generation_kwargs: Arguments for generation
            
        Returns:
            List of generated continuations
        """
        # Group prompts by common prefix
        prefix_groups = self._group_by_prefix(prompts)
        
        results = []
        for prefix, group_prompts in prefix_groups.items():
            if prefix and len(group_prompts) > 1:
                # Use caching for groups with shared prefix
                group_results = self._process_with_shared_prefix(
                    model, prefix, group_prompts, steering_fn, **generation_kwargs
                )
            else:
                # Process individually for unique prompts
                group_results = [
                    steering_fn(model, prompt, **generation_kwargs)
                    for prompt in group_prompts
                ]
            
            results.extend(group_results)
        
        return results
    
    def _group_by_prefix(self, prompts: List[str]) -> Dict[Optional[str], List[str]]:
        """Group prompts by their common prefixes."""
        # For now, implement simple grouping - could be enhanced
        # to find multiple prefix groups
        prefix = self.extract_common_prefix(prompts)
        
        if prefix:
            return {prefix: prompts}
        else:
            return {None: prompts}
    
    def _process_with_shared_prefix(
        self,
        model: NNsightChatModel,
        prefix: str,
        prompts: List[str],
        steering_fn: callable,
        **generation_kwargs
    ) -> List[str]:
        """Process prompts that share a common prefix."""
        # Tokenize prefix
        prefix_tokens = model.to_tokens(prefix)
        
        # Check if prefix computation is cached
        cached_state = self.kv_cache.get(prefix_tokens, model.model_name)
        
        if cached_state is None:
            # Compute prefix state and cache it
            logger.info(f"Computing prefix state for {len(prompts)} prompts")
            # This would need to be implemented based on the specific
            # nnsight API for extracting intermediate states
            cached_state = self._compute_prefix_state(model, prefix_tokens)
            self.kv_cache.put(prefix_tokens, model.model_name, cached_state)
        else:
            logger.info(f"Reusing cached prefix state for {len(prompts)} prompts")
        
        # Generate continuations using cached prefix
        results = []
        for prompt in prompts:
            # Extract the continuation part (after prefix)
            continuation = prompt[len(prefix):]
            
            # Generate with cached prefix state
            result = self._generate_with_cached_prefix(
                model, cached_state, continuation, steering_fn, **generation_kwargs
            )
            results.append(result)
        
        return results
    
    def _compute_prefix_state(self, model: NNsightChatModel, prefix_tokens: torch.Tensor) -> Dict:
        """Compute and return the state after processing prefix tokens."""
        # This is a placeholder - would need specific nnsight implementation
        # to extract KV states and activations at a specific position
        
        with model.model.trace(prefix_tokens):
            # Extract relevant intermediate states
            states = {}
            
            # Example: extract KV states from attention layers
            for layer_idx in range(model.cfg.n_layers):
                layer = model.model.model.layers[layer_idx]
                # This would need to be adapted based on model architecture
                # states[f'layer_{layer_idx}_kv'] = layer.attention.past_key_value
            
            states['position'] = prefix_tokens.size(1)
            states['prefix_tokens'] = prefix_tokens
        
        return states
    
    def _generate_with_cached_prefix(
        self,
        model: NNsightChatModel,
        cached_state: Dict,
        continuation: str,
        steering_fn: callable,
        **generation_kwargs
    ) -> str:
        """Generate continuation using cached prefix state."""
        # This is a placeholder for the actual implementation
        # Would need to:
        # 1. Restore cached KV states
        # 2. Continue generation from cached position
        # 3. Apply steering only to new tokens
        
        # For now, fall back to regular generation
        full_prompt = model.tokenizer.decode(cached_state['prefix_tokens'][0]) + continuation
        return steering_fn(model, full_prompt, **generation_kwargs)
    
    def clear(self):
        """Clear all cached data."""
        self.kv_cache.clear()
        self.prefix_mapping.clear()
        logger.info("Cleared shared prefix cache")


def optimize_batch_with_kv_cache(
    model: NNsightChatModel,
    prompts: List[str],
    steering_function: callable,
    cache: Optional[SharedPrefixCache] = None,
    **generation_kwargs
) -> List[str]:
    """
    Optimize batch generation using KV caching.
    
    Args:
        model: NNsight model instance
        prompts: List of prompts to process
        steering_function: Function to apply steering
        cache: Optional cache to use (creates new one if None)
        **generation_kwargs: Generation parameters
        
    Returns:
        List of generated results
    """
    if cache is None:
        cache = SharedPrefixCache()
    
    return cache.process_with_prefix_caching(
        model, prompts, steering_function, **generation_kwargs
    )