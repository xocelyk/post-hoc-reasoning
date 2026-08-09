"""
Apple Silicon MLX loader and generation helpers.

Provides a minimal, cache-backed path to run Qwen2.5 3B using MLX 4-bit
quantization on M1/M2 Macs. Intended to be called as an early-exit in
notebook/local generation utilities before falling back to Hugging Face
transformers.
"""

from __future__ import annotations

import platform
from typing import Dict, Optional, Tuple


_MLX_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}


def is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon (arm64) macOS."""
    try:
        return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}
    except Exception:
        return False


def can_use_mlx_for_model(hf_model_name: str) -> bool:
    """
    Decide whether to prefer MLX for a given HF model id.

    Currently enabled for Qwen2.5-3B and Qwen2.5-3B-Instruct, which benefit
    significantly from 4-bit quantization on Apple Silicon.
    """
    name = hf_model_name.lower()
    return (
        "qwen2.5-3b" in name
        or "qwen2.5/3b" in name  # defensive, in case of alternate naming
    )


def _import_mlx_lm():
    try:
        from mlx_lm import load as mlx_load  # type: ignore
        from mlx_lm import generate as mlx_generate  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mlx-lm is not installed. Install with `pip install -U mlx mlx-lm`."
        ) from exc
    return mlx_load, mlx_generate


def get_or_load_mlx_q4(hf_model_name: str) -> Tuple[object, object]:
    """Load (or fetch from cache) an MLX q4 model and tokenizer."""
    if hf_model_name in _MLX_MODEL_CACHE:
        return _MLX_MODEL_CACHE[hf_model_name]

    mlx_load, _ = _import_mlx_lm()
    model, tokenizer = mlx_load(hf_model_name, quantize="q4")
    _MLX_MODEL_CACHE[hf_model_name] = (model, tokenizer)
    return model, tokenizer


def maybe_generate_with_mlx(
    hf_model_name: str,
    prompt: str,
    *,
    max_new_tokens: int = 20,
) -> Optional[str]:
    """
    If on Apple Silicon and the model is supported, generate using MLX q4 and
    return the string output. Otherwise return None so callers can fall back to
    their standard Hugging Face path.
    """
    if not is_apple_silicon():
        return None
    if not can_use_mlx_for_model(hf_model_name):
        return None

    _, mlx_generate = _import_mlx_lm()
    model, tokenizer = get_or_load_mlx_q4(hf_model_name)

    # mlx_lm.generate uses `max_tokens` (new tokens count)
    text: str = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
    )
    return text


