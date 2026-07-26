"""Sensitivity analysis: rank layers/channels by quantization loss."""
from __future__ import annotations

from typing import Dict, List

import torch

from .error_metrics import mse
from .fake_quant import fake_quantize_tensor
from ..shared.types import QuantConfig


def layer_sensitivity(weights: Dict[str, torch.Tensor], config: QuantConfig,
                      topk: int = 10) -> List[dict]:
    """Rank modules by per-layer quantization MSE (descending).

    The top-k highest-loss modules are flagged for attention.
    """
    rows: List[dict] = []
    for path, w in weights.items():
        q = fake_quantize_tensor(w, config)
        rows.append({"module_path": path, "mse": mse(w, q)})
    rows.sort(key=lambda r: r["mse"], reverse=True)
    if topk and len(rows) > topk:
        for r in rows[:topk]:
            r["top"] = True
    else:
        for r in rows[:topk]:
            r["top"] = True
    return rows
