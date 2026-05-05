# SPDX-License-Identifier: Apache-2.0
"""AWQ load owner for vllm-metal.

Encapsulates the entry-point preflight, the ``mlx_lm.load`` invocation
with a normalized ``model_config={"quantization_config": ...}`` kwarg,
the dtype-scoped cache key, and the post-load alignment of non-quantized
floating params. ``ModelLifecycle`` delegates the entire AWQ branch to
instances of this class so quantization policy stays cohesive in one
place rather than leaking into the generic loader flow.

GPTQ checkpoints currently flow through the same code path because
mlx-lm's ``_transform_awq_weights`` accepts both ``quant_method`` values
under the v1 supported subset. GPTQ is not part of the public support
claim until a real GPTQ checkpoint is validated end-to-end.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import HFValidationError
from mlx_lm import load as mlx_lm_load
from vllm.logger import init_logger

from vllm_metal.quant.awq_config import normalize_quant_config

logger = init_logger(__name__)


_QUANT_METHODS = ("awq", "gptq")


def _read_raw_quantization_config(model_name: str) -> Mapping[str, Any] | None:
    """Read ``quantization_config`` from the model's ``config.json`` without
    invoking ``mlx_lm.load``. Returns ``None`` if the field is absent or the
    config cannot be located.

    Mirrors ``mlx_lm.utils.load_model``'s fallback to
    ``text_config.quantization_config`` for wrapper / multimodal configs.
    Without this, multimodal AWQ checkpoints that nest the quant config
    under ``text_config`` would skip the preflight entirely while mlx_lm
    itself would still apply the transform.
    """
    model_path = Path(model_name)
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if not config_path.is_file():
            return None
    else:
        try:
            config_path = Path(hf_hub_download(model_name, "config.json"))
        except (HfHubHTTPError, HFValidationError, OSError):
            # ``model_name`` cannot be reached as a repo id: Hub-side
            # failure (404, auth, transport), an unparseable repo id, or
            # a filesystem error. Leave the preflight inactive and let
            # ``mlx_lm.load`` surface its own error later.
            return None
    with open(config_path) as fid:
        config = json.load(fid)
    qc = config.get("quantization_config")
    if isinstance(qc, dict):
        return qc
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        nested = text_config.get("quantization_config")
        if isinstance(nested, dict):
            return nested
    return None


def _align_non_quantized_dtypes(model: Any, target_dtype: Any) -> int:
    """Cast floating-dtype params on non-``QuantizedLinear`` leaf modules to
    ``target_dtype``. Returns the number of cast tensors.

    Quantized layers' ``scales`` / ``biases`` are intentionally left at the
    dtype produced by mlx_lm's AWQ transform (typically fp16); only the
    surrounding floating params (embeddings, layernorms, q/k/v biases) are
    aligned with the engine's runtime dtype.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    # `tree_flatten` is overloaded `list[tuple[str, Any]] | dict[str, Any]`
    # depending on the `destination` kwarg; with `destination=None` (default)
    # it returns the list. Narrow at runtime so mypy can unpack the tuples.
    leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    assert isinstance(leaves, list)

    n_cast = 0
    for _path, module in leaves:
        if isinstance(module, nn.QuantizedLinear):
            continue
        updates = {}
        for name, value in module.parameters().items():
            dtype = getattr(value, "dtype", None)
            if dtype is None:
                continue
            if not mx.issubdtype(dtype, mx.floating):
                continue
            if dtype == target_dtype:
                continue
            updates[name] = value.astype(target_dtype)
        if updates:
            module.update(updates)
            n_cast += len(updates)
    return n_cast


class AWQQuantLoader:
    """Owner for the AWQ (and internally also GPTQ) load flow.

    Construct via :meth:`for_model`, which inspects the checkpoint's
    ``config.json`` and returns ``None`` when the checkpoint is not
    AWQ/GPTQ. Lifecycle dispatches the quantized branch to this owner so
    the dtype-scoped cache key and the post-load alignment policy do not
    bleed into generic loader code paths.
    """

    def __init__(self, normalized_quant_config: Mapping[str, Any]) -> None:
        # `normalize_quant_config` already canonicalized aliases and
        # rejected unsupported variants; stash a kwarg dict ready to hand
        # to ``mlx_lm.load(model_config=...)``.
        self._mlx_lm_model_config: dict[str, Any] = {
            "quantization_config": dict(normalized_quant_config)
        }

    @classmethod
    def for_model(cls, model_name: str) -> AWQQuantLoader | None:
        """Detect AWQ/GPTQ in ``model_name``'s ``config.json`` (local dir or
        HF Hub) and return a configured loader. Returns ``None`` when the
        checkpoint is not AWQ/GPTQ.

        Raises:
            UnsupportedQuantizationConfigError: AWQ/GPTQ but outside v1 scope.
        """
        raw_qc = _read_raw_quantization_config(model_name)
        if raw_qc is None:
            return None
        if raw_qc.get("quant_method") not in _QUANT_METHODS:
            return None
        return cls(normalize_quant_config(raw_qc))

    @staticmethod
    def cache_key(model_name: str, *, target_dtype: Any) -> tuple[str, str]:
        """Cache key for an AWQ load.

        ``target_dtype`` is encoded into the loader segment because the
        post-load alignment mutates the model in place: a model first
        loaded with bf16 and later requested as fp16 must NOT be served
        from cache, since the cached object would carry the wrong dtype
        on its non-quantized floating params. Encoding the dtype inside
        the loader segment keeps the cache key shape identical to the
        generic ``_generation_cache_key`` (a 2-tuple), so dtype scoping
        is a property of this owner rather than of the generic cache.

        Static so lifecycle can compute the speculative AWQ cache key
        before deciding whether to invoke detection (which involves an
        HF Hub config fetch on cache miss).
        """
        return (model_name, f"mlx_lm-awq:{target_dtype}")

    def load(
        self,
        model_path: str,
        *,
        target_dtype: Any,
        tokenizer_config: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Run ``mlx_lm.load`` with the normalized quant config, then align
        non-quantized floating params to ``target_dtype``.

        ``model_path`` is the (possibly compatibility-adapted) path that
        the lifecycle resolves via ``_mlx_lm_compatible_model_path``; the
        owner does not duplicate that path discovery.
        """
        model, tokenizer = mlx_lm_load(
            model_path,
            tokenizer_config=dict(tokenizer_config) if tokenizer_config else None,
            model_config=self._mlx_lm_model_config,
        )
        n_cast = _align_non_quantized_dtypes(model, target_dtype)
        logger.info(
            "AWQ load: aligned %d non-quantized floating params to %s",
            n_cast,
            target_dtype,
        )
        return model, tokenizer
