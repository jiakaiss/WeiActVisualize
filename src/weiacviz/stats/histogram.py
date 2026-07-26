"""Histogram bucketing for tensor distributions."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ..loading.weights import slice_weight
from ..shared.types import Granularity, HistogramResult
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


def histogram_sliced(
    t,
    granularity: Granularity,
    group_size: Optional[int] = None,
    num_bins: int = 256,
    value_range: Optional[tuple] = None,
) -> List[HistogramResult]:
    """Compute a histogram per slice (per-channel / per-group / per-tensor).

    Reuses ``slice_weight`` to split the tensor, then buckets each slice
    independently. When ``value_range`` is None each slice uses its own auto
    range; passing a (min, max) aligns bins across slices for comparison.

    Args:
        t: tensor-like (torch.Tensor or np.ndarray).
        granularity: PER_TENSOR / PER_CHANNEL / PER_GROUP.
        group_size: required for PER_GROUP.
        num_bins: bins per slice.
        value_range: shared (min, max) for cross-slice alignment, or None.
    """
    if not hasattr(t, "dim"):
        t = torch.as_tensor(np.asarray(t))
    slices = slice_weight(t, granularity, group_size)
    results: List[HistogramResult] = []
    for s in slices:
        arr = to_numpy(s).flatten()
        if arr.size == 0:
            results.append(HistogramResult(counts=[], bin_edges=[], num_bins=0))
            continue
        vr = value_range
        if vr is None:
            lo, hi = float(arr.min()), float(arr.max())
            if lo == hi:
                hi = lo + 1.0
            vr = (lo, hi)
        counts, edges = np.histogram(arr, bins=num_bins, range=vr)
        results.append(HistogramResult(
            counts=counts.tolist(), bin_edges=edges.tolist(), num_bins=num_bins,
        ))
    return results
