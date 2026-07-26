"""Distribution shape metrics: excess kurtosis, skewness, robust tail-ratio."""
from __future__ import annotations

import numpy as np

from ._util import to_numpy

# Empirical thresholds for shape labels (see design.md D3).
HEAVY_TAIL_KURTOSIS = 3.0   # Laplace ~= 3
LIGHT_TAIL_KURTOSIS = -0.5  # Uniform ~= -1.2
SKEWNESS_THRESHOLD = 0.5


def excess_kurtosis(a) -> float:
    """Population (biased) excess kurtosis = m4 / m2^2 - 3.

    Normal => 0, heavy-tail > 0, light-tail < 0. Returns NaN for
    zero-variance / empty inputs without raising.
    """
    arr = to_numpy(a).flatten()
    if arr.size == 0:
        return float("nan")
    d = arr - arr.mean()
    var = float((d ** 2).mean())
    if var == 0:
        return float("nan")
    m4 = float((d ** 4).mean())
    return m4 / (var * var) - 3.0


def skewness(a) -> float:
    """Population (biased) skewness = m3 / m2^1.5. Normal => 0.

    Returns NaN for zero-variance / empty inputs without raising.
    """
    arr = to_numpy(a).flatten()
    if arr.size == 0:
        return float("nan")
    d = arr - arr.mean()
    var = float((d ** 2).mean())
    if var == 0:
        return float("nan")
    m3 = float((d ** 3).mean())
    return m3 / (var ** 1.5)


def robust_tail_ratio(a, lo: float = 0.1, hi: float = 99.9) -> float:
    """Robust tail ratio = (p_hi - p_lo) / (2 * std).

    Large values indicate a heavy tail wasting the quantization scale.
    Returns NaN for zero-variance / empty inputs without raising.
    """
    arr = to_numpy(a).flatten()
    if arr.size == 0:
        return float("nan")
    std = float(arr.std())
    if std == 0:
        return float("nan")
    p_lo = float(np.percentile(arr, lo))
    p_hi = float(np.percentile(arr, hi))
    return (p_hi - p_lo) / (2.0 * std)


def shape_label(kurtosis: float, skewness: float) -> str:
    """Readable shape label from excess kurtosis, with skewness annotation.

    Returns '正态' / '重尾' / '轻尾' ('未知' for NaN kurtosis). When
    |skewness| exceeds SKEWNESS_THRESHOLD, appends an asymmetric-quant hint.
    """
    if kurtosis != kurtosis:  # NaN check
        base = "未知"
    elif kurtosis > HEAVY_TAIL_KURTOSIS:
        base = "重尾"
    elif kurtosis < LIGHT_TAIL_KURTOSIS:
        base = "轻尾"
    else:
        base = "正态"
    if skewness == skewness and abs(skewness) > SKEWNESS_THRESHOLD:
        return f"{base}（偏态，非对称量化可能更优）"
    return base
