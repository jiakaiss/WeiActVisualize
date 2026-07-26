"""Outlier detection: percentile / Z-score."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ._util import to_numpy


@dataclass
class OutlierResult:
    method: str
    threshold: float
    outlier_ratio: float
    indices: List[int]
    max_abs: float


def detect_outliers_percentile(t, percentile: float = 99.9) -> OutlierResult:
    """Flag values whose absolute magnitude exceeds the given percentile."""
    a = np.abs(to_numpy(t)).flatten()
    if a.size == 0:
        return OutlierResult("percentile", float("nan"), 0.0, [], 0.0)
    thr = float(np.percentile(a, percentile))
    idx = np.where(a >= thr)[0].tolist()
    return OutlierResult(
        method="percentile", threshold=thr,
        outlier_ratio=len(idx) / a.size,
        indices=idx, max_abs=float(a.max()),
    )


def detect_outliers_zscore(t, z_threshold: float = 3.0) -> OutlierResult:
    """Flag values whose |z-score| exceeds the threshold."""
    a = to_numpy(t).flatten()
    if a.size == 0:
        return OutlierResult("zscore", z_threshold, 0.0, [], 0.0)
    mu, sigma = float(a.mean()), float(a.std())
    if sigma == 0:
        return OutlierResult("zscore", z_threshold, 0.0, [], float(np.abs(a).max()))
    z = np.abs((a - mu) / sigma)
    idx = np.where(z >= z_threshold)[0].tolist()
    return OutlierResult(
        method="zscore", threshold=z_threshold,
        outlier_ratio=len(idx) / a.size,
        indices=idx, max_abs=float(np.abs(a).max()),
    )
