"""Weight distribution statistics: per-tensor / per-channel / per-group."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..loading.weights import slice_weight
from ..shared.types import Granularity, StatLevel, StatResult, TensorRole
from ._util import to_numpy


def weight_stats(
    weight,
    module_path: str,
    granularity: Granularity = Granularity.PER_CHANNEL,
    group_size: Optional[int] = None,
    percentiles: Sequence[float] = (0.1, 1, 50, 99, 99.9),
) -> List[StatResult]:
    """Compute weight statistics at the requested granularity.

    Returns one StatResult per slice (per-channel / per-group) or a single
    per-tensor result.
    """
    slices = slice_weight(weight, granularity, group_size)
    level = StatLevel(granularity.value)
    results: List[StatResult] = []
    for s in slices:
        a = to_numpy(s).flatten()
        if a.size == 0:
            continue
        pct = {str(p): float(np.percentile(a, p)) for p in percentiles}
        results.append(StatResult(
            module_path=module_path, role=TensorRole.WEIGHT, level=level,
            min=float(a.min()), max=float(a.max()),
            mean=float(a.mean()), std=float(a.std()),
            percentiles=pct,
        ))
    return results
