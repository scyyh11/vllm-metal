# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``AWQQuantLoader`` under ``vllm_metal/quant/``.

Covers detection (``for_model``), cache-key dtype isolation, and the
``text_config.quantization_config`` fallback that mirrors mlx-lm. Pure
helpers; no model load, no downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
import torch

from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.quant.awq_config import UnsupportedQuantizationConfigError
from vllm_metal.quant.awq_loader import (
    AWQQuantLoader,
    _read_raw_quantization_config,
)


def _mlx_dtype(torch_dtype):
    """Mirror what production does to derive the cache-key dtype: convert a
    torch dtype through ``torch_to_mlx``. Tests use this instead of literal
    strings so we pin the *actual* MLX dtype the production path passes
    (e.g. ``mlx.core.bfloat16``), not just "different strings produce
    different keys".
    """
    return torch_to_mlx(torch.empty(0, dtype=torch_dtype)).dtype


_AWQ_INNER = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "version": "gemm",
}


def _write_config(tmp_path: Path, config: dict) -> Path:
    """Drop a config.json into ``tmp_path`` and return the directory."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


# ---- cache-key dtype isolation ---------------------------------------------


def test_cache_key_isolates_by_dtype():
    """Two engines requesting the same model with different dtypes must NOT
    share a cache entry, since AWQ post-load alignment mutates the model
    in place."""
    loader = AWQQuantLoader(_AWQ_INNER)
    bf16_key = loader.cache_key(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ", target_dtype=_mlx_dtype(torch.bfloat16)
    )
    fp16_key = loader.cache_key(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ", target_dtype=_mlx_dtype(torch.float16)
    )
    assert bf16_key != fp16_key


def test_cache_key_pins_loader_segment_with_dtype():
    """Pin the wire format: the second key segment is ``mlx_lm-awq:<dtype>``
    (encoding the dtype inside the loader segment, not as a third tuple
    element). A future refactor that strips the dtype or swaps to torch
    repr would fail loudly here.
    """
    loader = AWQQuantLoader(_AWQ_INNER)
    key = loader.cache_key("foo", target_dtype=_mlx_dtype(torch.bfloat16))
    assert key == ("foo", f"mlx_lm-awq:{mx.bfloat16}")


def test_cache_key_distinct_from_generic_lifecycle_key():
    """AWQ cache key must not collide with the generic mlx_lm key for the
    same model: an AWQ-mutated cached model must not be served to a
    non-AWQ caller, nor vice versa."""
    from vllm_metal.v1.model_lifecycle import _generation_cache_key

    awq_key = AWQQuantLoader(_AWQ_INNER).cache_key(
        "x", target_dtype=_mlx_dtype(torch.bfloat16)
    )
    generic_key = _generation_cache_key("x", is_vlm=False)
    assert awq_key != generic_key


def test_cache_key_stable_for_same_inputs():
    loader = AWQQuantLoader(_AWQ_INNER)
    bf16 = _mlx_dtype(torch.bfloat16)
    assert loader.cache_key("foo", target_dtype=bf16) == loader.cache_key(
        "foo", target_dtype=bf16
    )


# ---- text_config.quantization_config fallback ------------------------------


def test_read_quantization_config_top_level(tmp_path):
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
    )
    assert _read_raw_quantization_config(str(model_dir)) == _AWQ_INNER


def test_read_quantization_config_nested_text_config(tmp_path):
    """Multimodal wrapper configs nest the quant config under
    ``text_config``. ``mlx_lm.utils.load_model`` falls back to it; the
    AWQ owner must do the same so the alias normalization / reject logic
    still runs.
    """
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "wrapper_vlm",
            "text_config": {
                "model_type": "qwen2",
                "quantization_config": _AWQ_INNER,
            },
        },
    )
    assert _read_raw_quantization_config(str(model_dir)) == _AWQ_INNER


def test_read_quantization_config_top_level_wins_over_text_config(tmp_path):
    """If both are present, top-level takes precedence (matches mlx-lm)."""
    nested = {**_AWQ_INNER, "bits": 8}  # would normally reject
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "wrapper",
            "quantization_config": _AWQ_INNER,
            "text_config": {"quantization_config": nested},
        },
    )
    assert _read_raw_quantization_config(str(model_dir)) == _AWQ_INNER


def test_read_quantization_config_absent(tmp_path):
    model_dir = _write_config(tmp_path, {"model_type": "qwen2"})
    assert _read_raw_quantization_config(str(model_dir)) is None


def test_read_quantization_config_missing_dir():
    """Non-existent path / non-HF repo: silently inactive, returns None."""
    assert _read_raw_quantization_config("/nonexistent/path/zzz") is None


# ---- AWQQuantLoader.for_model ---------------------------------------------


def test_for_model_returns_none_for_non_awq(tmp_path):
    """Non-AWQ/GPTQ checkpoints get None back, no exception."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "qwen2",
            "quantization_config": {"quant_method": "fp8", "bits": 8},
        },
    )
    assert AWQQuantLoader.for_model(str(model_dir)) is None


def test_for_model_returns_none_for_no_quant_config(tmp_path):
    model_dir = _write_config(tmp_path, {"model_type": "qwen2"})
    assert AWQQuantLoader.for_model(str(model_dir)) is None


def test_for_model_returns_loader_for_awq(tmp_path):
    """AWQ checkpoint: ``for_model`` returns a configured loader whose
    cache key reflects the requested dtype."""
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
    )
    loader = AWQQuantLoader.for_model(str(model_dir))
    assert loader is not None
    key = loader.cache_key(str(model_dir), target_dtype=_mlx_dtype(torch.bfloat16))
    assert key[1].startswith("mlx_lm-awq:")


def test_for_model_propagates_reject_via_text_config(tmp_path):
    """The text_config fallback must surface reject errors too — nesting
    must not be a hole that lets unsupported configs through."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "wrapper",
            "text_config": {
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 8,  # reject
                    "group_size": 128,
                },
            },
        },
    )
    with pytest.raises(UnsupportedQuantizationConfigError):
        AWQQuantLoader.for_model(str(model_dir))


def test_for_model_accepts_aliased_config(tmp_path):
    """``for_model`` runs ``normalize_quant_config`` on detection, so an
    AutoAWQ-style aliased config (``w_bit``, ``q_group_size``, uppercase
    ``GEMM``) yields a valid loader rather than raising.
    """
    aliased = {
        "quant_method": "awq",
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
        "version": "GEMM",
    }
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": aliased},
    )
    assert AWQQuantLoader.for_model(str(model_dir)) is not None
