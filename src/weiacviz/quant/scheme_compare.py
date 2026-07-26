"""Compare multiple quantization schemes on the same tensor."""
from __future__ import annotations

from typing import List, Sequence

import torch

from .error_metrics import cosine_similarity, mse
from .fake_quant import fake_quantize_tensor
from ..shared.types import Granularity, QuantConfig, Symmetry


def compare_schemes(
    weight: torch.Tensor,
    bits_options: Sequence[int] = (4, 8),
    granularities: Sequence[Granularity] = (Granularity.PER_TENSOR, Granularity.PER_CHANNEL),
    symmetries: Sequence[Symmetry] = (Symmetry.SYMMETRIC, Symmetry.ASYMMETRIC),
    group_size: int = 128,
) -> List[dict]:
    """Run fake quant under a cartesian product of schemes, returning an error table."""
    rows: List[dict] = []
    for bits in bits_options:
        for g in granularities:
            for sym in symmetries:
                cfg = QuantConfig(bits=bits, granularity=g, symmetry=sym, group_size=group_size)
                try:
                    q = fake_quantize_tensor(weight, cfg)
                except Exception as e:  # noqa: BLE001
                    rows.append({"bits": bits, "granularity": g.value,
                                 "symmetry": sym.value, "group_size": group_size,
                                 "mse": None, "cosine": None, "error": str(e)})
                    continue
                rows.append({
                    "bits": bits, "granularity": g.value, "symmetry": sym.value,
                    "group_size": group_size if g == Granularity.PER_GROUP else None,
                    "mse": mse(weight, q), "cosine": cosine_similarity(weight, q),
                })
    return rows
