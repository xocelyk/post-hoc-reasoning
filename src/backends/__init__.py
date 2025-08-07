"""
Backend implementations for post-hoc reasoning experiments.

This package contains backend-specific implementations for:
- TransformerLens: For models supported by the TransformerLens library
- NNsight: For broader model support including DeepSeek and others
"""

from .base import BaseModel, BaseActivationExtractor, BaseSteering

__all__ = ["BaseModel", "BaseActivationExtractor", "BaseSteering"]