"""Activation distribution statistics: per-token / per-channel."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..shared.types import StatLevel, StatResult, TensorRole
from ._util import to_numpy


def _flatten_tokens(a: np.ndarray) -> np.ndarray:
    """[..., seq, hidden] -> [N, hidden]."""
    if a.ndim < 2:
        a = a.reshape(1, -1)
    return a.reshape(-1, a.shape[-1])


def activation_stats_per_token(
    activation, module_path: str,
    percentiles: Sequence[float] = (50, 99, 99.9),
) -> List[StatResult]:
    """Per-token activation stats (one result per token row)."""
    tokens = _flatten_tokens(to_numpy(activation))
    level = StatLevel.PER_CHANNEL
    results: List[StatResult] = []
    for row in tokens:
        pct = {str(p): float(np.percentile(row, p)) for p in percentiles}
        results.append(StatResult(
            module_path=module_path, role=TensorRole.ACTIVATION, level=level,
            min=float(row.min()), max=float(row.max()),
            mean=float(row.mean()), std=float(row.std()),
            percentiles=pct,
        ))
    return results


def activation_stats_per_channel(
    activation, module_path: str,
    percentiles: Sequence[float] = (50, 99, 99.9),
) -> List[StatResult]:
    """Per-channel (hidden-dim) activation stats across all tokens."""
    tokens = _flatten_tokens(to_numpy(activation))  # [N, hidden]
    level = StatLevel.PER_CHANNEL
    results: List[StatResult] = []
    for c in range(tokens.shape[1]):
        col = tokens[:, c]
        pct = {str(p): float(np.percentile(col, p)) for p in percentiles}
        results.append(StatResult(
            module_path=module_path, role=TensorRole.ACTIVATION, level=level,
            min=float(col.min()), max=float(col.max()),
            mean=float(col.mean()), std=float(col.std()),
            percentiles=pct,
        ))
    return results
