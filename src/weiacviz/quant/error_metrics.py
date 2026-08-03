"""Quantization error metrics: MSE, cosine similarity, output diff."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.float() - b.float()) ** 2))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)))


def output_diff(module, weight_quantized, sample_input,
                quantize_activation: bool = False, act_bits: int = 8,
                act_granularity: str = "per-token") -> dict:
    """Compare original module output vs output with quantized weights
    (and optionally quantized activation).

    Temporarily swaps the module's weight, runs forward on ``sample_input``
    (fake-quantized per-token if ``quantize_activation``), then restores the
    original weight. ``ref_out`` always uses the original (fp) activation and
    original weight, so the returned mse/cosine measures the combined
    weight (+activation) quantization error vs the fp reference.
    """
    from .fake_quant import fake_quantize_activation
    original_weight = module.weight.data.clone()
    device = module.weight.device
    x = sample_input
    if quantize_activation:
        x = fake_quantize_activation(sample_input, bits=act_bits,
                                     granularity=act_granularity)
    try:
        module.weight.data = weight_quantized.to(module.weight.dtype).to(device)
        with torch.no_grad():
            q_out = module(x.to(device))
        module.weight.data = original_weight
        with torch.no_grad():
            ref_out = module(sample_input.to(device))
    finally:
        module.weight.data = original_weight
    return {"mse": mse(ref_out, q_out), "cosine": cosine_similarity(ref_out, q_out)}
