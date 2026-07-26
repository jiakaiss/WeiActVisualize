"""Fake quantization (round-clamp-dequant) for weights and activations."""
from __future__ import annotations

import torch

from ..shared.types import Granularity, QuantConfig, Symmetry


def _per_axis_scale_zero(t: torch.Tensor, bits: int, symmetry: Symmetry, dim: int):
    """Compute (scale, zero) along `dim`, broadcastable to t."""
    if symmetry == Symmetry.SYMMETRIC:
        qmax = (1 << (bits - 1)) - 1
        amax = t.abs().amax(dim=dim, keepdim=True)
        scale = torch.clamp(amax / qmax, min=1e-12)
        zero = torch.zeros_like(scale)
    else:
        qmax = (1 << bits) - 1
        tmin = t.amin(dim=dim, keepdim=True)
        tmax = t.amax(dim=dim, keepdim=True)
        scale = torch.clamp((tmax - tmin) / qmax, min=1e-12)
        zero = -torch.round(tmin / scale)
    return scale, zero


def _quant_dequant(t: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                   bits: int, symmetry: Symmetry) -> torch.Tensor:
    if symmetry == Symmetry.SYMMETRIC:
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
        q = torch.clamp(torch.round(t / scale), qmin, qmax)
        return q * scale
    qmax = (1 << bits) - 1
    q = torch.clamp(torch.round(t / scale + zero), 0, qmax)
    return (q - zero) * scale


def fake_quantize_tensor(t: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Apply fake quantization per config.

    Granularity:
      - per-tensor: single scale over all values
      - per-channel: scale per output channel (axis 0)
      - per-group: groups along the last axis (input dim for 2D weights)
    """
    t = t.detach().float()
    original_shape = t.shape
    g = config.granularity

    if g == Granularity.PER_TENSOR:
        flat = t.reshape(-1)
        scale, zero = _per_axis_scale_zero(flat, config.bits, config.symmetry, 0)
        out = _quant_dequant(flat, scale, zero, config.bits, config.symmetry)
        return out.reshape(original_shape).to(t.dtype)

    if g == Granularity.PER_CHANNEL:
        work = t.unsqueeze(0) if t.dim() == 1 else t
        scale, zero = _per_axis_scale_zero(work, config.bits, config.symmetry, 0)
        out = _quant_dequant(work, scale, zero, config.bits, config.symmetry)
        return out.reshape(original_shape).to(t.dtype)

    # PER_GROUP: groups along the last axis
    gs = config.group_size or 128
    if t.dim() < 2:
        flat = t.reshape(-1)
        n = flat.numel()
        if n < gs:
            return t.clone()
        trim = (n // gs) * gs
        work = flat[:trim].reshape(-1, gs)
        scale, zero = _per_axis_scale_zero(work, config.bits, config.symmetry, 1)
        q = _quant_dequant(work, scale, zero, config.bits, config.symmetry).reshape(-1)
        out = flat.clone()
        out[:trim] = q
        return out.reshape(original_shape).to(t.dtype)

    # 2D [out, in]: groups along in (pad if not divisible)
    out_f, in_f = t.shape
    pad = (gs - in_f % gs) % gs
    padded = torch.nn.functional.pad(t, (0, pad)) if pad else t
    work = padded.reshape(out_f, -1, gs)
    scale, zero = _per_axis_scale_zero(work, config.bits, config.symmetry, 2)
    q = _quant_dequant(work, scale, zero, config.bits, config.symmetry)
    q = q.reshape(out_f, -1)
    if pad:
        q = q[:, :in_f]
    return q.reshape(original_shape).to(t.dtype)
