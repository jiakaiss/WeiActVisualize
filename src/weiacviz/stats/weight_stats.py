"""Weight distribution statistics: per-tensor / per-channel / per-group."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..loading.weights import slice_weight
from ..shared.types import Granularity, StatLevel, StatResult, TensorRole
from ._util import to_numpy
from .histogram import histogram
from .outliers import detect_outliers_percentile
from .shape import excess_kurtosis, robust_tail_ratio, shape_label
from .shape import skewness as skewness_fn


def weight_stats(
    weight,
    module_path: str,
    granularity: Granularity = Granularity.PER_CHANNEL,
    group_size: Optional[int] = None,
    percentiles: Sequence[float] = (0.1, 1, 50, 99, 99.9),
    num_bins: int = 256,
) -> List[StatResult]:
    """Compute weight statistics at the requested granularity.

    Returns one StatResult per slice (per-channel / per-group) or a single
    per-tensor result. Each result carries min/max/mean/std/percentiles,
    distribution shape metrics (kurtosis / skewness / tail-ratio /
    shape_label), outlier ratio, and a per-slice histogram.
    """
    slices = slice_weight(weight, granularity, group_size)
    level = StatLevel(granularity.value)
    results: List[StatResult] = []
    for s in slices:
        a = to_numpy(s).flatten()
        if a.size == 0:
            continue
        pct = {str(p): float(np.percentile(a, p)) for p in percentiles}
        k = excess_kurtosis(a)
        sk = skewness_fn(a)
        tr = robust_tail_ratio(a)
        out = detect_outliers_percentile(a, percentile=99.9).outlier_ratio
        results.append(StatResult(
            module_path=module_path, role=TensorRole.WEIGHT, level=level,
            min=float(a.min()), max=float(a.max()),
            mean=float(a.mean()), std=float(a.std()),
            percentiles=pct,
            kurtosis=k, skewness=sk, tail_ratio=tr,
            outlier_ratio=out, shape_label=shape_label(k, sk),
            histogram=histogram(a, num_bins=num_bins),
        ))
    return results
