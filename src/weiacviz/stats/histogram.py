"""Histogram bucketing for tensor distributions."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..shared.types import HistogramResult
from ._util import to_numpy


def histogram(t, num_bins: int = 256, value_range: Optional[tuple] = None) -> HistogramResult:
    """Compute a histogram over tensor values.

    Args:
        t: tensor-like (torch.Tensor or np.ndarray).
        num_bins: number of bins.
        value_range: (min, max) or None for auto from data.
    """
    arr = to_numpy(t).flatten()
    if arr.size == 0:
        return HistogramResult(counts=[], bin_edges=[], num_bins=0)
    if value_range is None:
        lo, hi = float(arr.min()), float(arr.max())
        if lo == hi:
            hi = lo + 1.0
        value_range = (lo, hi)
    counts, edges = np.histogram(arr, bins=num_bins, range=value_range)
    return HistogramResult(counts=counts.tolist(), bin_edges=edges.tolist(), num_bins=num_bins)
