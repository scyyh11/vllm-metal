# SPDX-License-Identifier: Apache-2.0
"""Quantization helpers for vllm-metal load paths.

Today this module only contains the AWQ/GPTQ entry-point shim
(`awq_config.normalize_quant_config`) used by `model_lifecycle` to validate
HF AWQ checkpoint metadata before delegating to `mlx_lm.load`. The actual
quantized GEMM kernel is `mx.quantized_matmul` in MLX core; mlx_lm 0.31.3+
provides the AWQ -> MLX-affine repack via `_transform_awq_weights`.
"""
