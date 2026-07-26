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


def output_diff(module, weight_quantized, sample_input) -> dict:
    """Compare original module output vs output with quantized weights.

    Temporarily swaps the module's weight, runs forward on sample_input,
    then restores the original weight.
    """
    original_weight = module.weight.data.clone()
    device = module.weight.device
    try:
        module.weight.data = weight_quantized.to(module.weight.dtype).to(device)
        with torch.no_grad():
            q_out = module(sample_input.to(device))
        module.weight.data = original_weight
        with torch.no_grad():
            ref_out = module(sample_input.to(device))
    finally:
        module.weight.data = original_weight
    return {"mse": mse(ref_out, q_out), "cosine": cosine_similarity(ref_out, q_out)}
