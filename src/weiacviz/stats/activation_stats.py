"""Activation distribution statistics: per-token / per-channel."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..shared.types import StatLevel, StatResult, TensorRole
from ._util import to_numpy
from .shape import excess_kurtosis, robust_tail_ratio, shape_label, skewness


def _flatten_tokens(a: np.ndarray) -> np.ndarray:
    """[..., seq, hidden] -> [N, hidden]."""
    if a.ndim < 2:
        a = a.reshape(1, -1)
    return a.reshape(-1, a.shape[-1])


def activation_stats_per_token(
    activation, module_path: str,
    percentiles: Sequence[float] = (50, 99, 99.9),
) -> List[StatResult]:
    """Per-token activation stats (one result per token row).

    Each token's hidden-dim distribution gets shape metrics (kurtosis /
    skewness / tail_ratio / shape_label) so per-token slices can be rendered
    with the same violin + heatmap pipeline as per-channel weights.
    """
    tokens = _flatten_tokens(to_numpy(activation))
    level = StatLevel.PER_CHANNEL
    results: List[StatResult] = []
    for row in tokens:
        pct = {str(p): float(np.percentile(row, p)) for p in percentiles}
        k = excess_kurtosis(row)
        sk = skewness(row)
        results.append(StatResult(
            module_path=module_path, role=TensorRole.ACTIVATION, level=level,
            min=float(row.min()), max=float(row.max()),
            mean=float(row.mean()), std=float(row.std()),
            percentiles=pct,
            kurtosis=k, skewness=sk,
            tail_ratio=robust_tail_ratio(row),
            shape_label=shape_label(k, sk),
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


# --- per-token abs_max distribution (online aggregator output) ---

def _hist_percentile(counts, bin_edges, p: float) -> float:
    """Approximate the p-th percentile from a histogram (linear interp in bin)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return float("nan")
    target = p / 100.0 * total
    cdf = np.cumsum(counts)
    idx = int(np.searchsorted(cdf, target))
    if idx >= len(counts):
        idx = len(counts) - 1
    lo = float(bin_edges[idx])
    hi = float(bin_edges[idx + 1])
    prev = float(cdf[idx - 1]) if idx > 0 else 0.0
    frac = (target - prev) / counts[idx] if counts[idx] > 0 else 0.0
    return lo + frac * (hi - lo)


def activation_token_summary(token_stats, module_path: str,
                             role: str = "output") -> StatResult:
    """Summarize the per-token abs_max distribution as a StatResult.

    ``token_stats`` is duck-typed (a RunningTokenStats): exposes mean/std/cv/
    kurtosis/skewness/min/max/abs_max_hist. The per-token abs_max distribution
    answers whether per-tensor activation quantization suffices (concentrated)
    or per-token quantization is needed (spread / heavy-tailed) -- the
    standard W8A8 per-token activation decision.

    Shape metrics (kurtosis/skewness) come from online raw moments (available
    even in single-pass); tail_ratio/percentiles need the two-pass histogram
    and are NaN when it was not collected.
    """
    k = token_stats.kurtosis
    sk = token_stats.skewness
    std = token_stats.std
    hist = getattr(token_stats, "abs_max_hist", None)
    pct: dict = {}
    tr = float("nan")
    hr = None
    if hist is not None:
        hr = hist.to_result()
        pct = {
            str(p): _hist_percentile(hr.counts, hr.bin_edges, p)
            for p in (0.1, 50, 99, 99.9)
        }
        p_lo = pct.get("0.1", float("nan"))
        p_hi = pct.get("99.9", float("nan"))
        if std and std > 0 and p_lo == p_lo and p_hi == p_hi:
            tr = (p_hi - p_lo) / (2.0 * std)
    return StatResult(
        module_path=module_path, role=TensorRole.ACTIVATION,
        level=StatLevel.PER_TENSOR,
        min=token_stats.min, max=token_stats.max,
        mean=token_stats.mean, std=std,
        percentiles=pct, histogram=hr,
        kurtosis=k, skewness=sk, tail_ratio=tr,
        outlier_ratio=float("nan"),
        shape_label=shape_label(k, sk),
    )


def activation_token_outliers(token_stats, method: str = "zscore",
                              k: float = 3.0, percentile: float = 99.0) -> dict:
    """Quantify outlier-token magnitude on the per-token abs_max distribution.

    ``token_stats`` is duck-typed (a RunningTokenStats): exposes mean/std/min/max
    and optional ``abs_max_hist``. Returns ``outlier_ratio`` (fraction of tokens
    whose abs_max is beyond the threshold), ``severity`` (max / median), the
    threshold, and ``max_abs``.

    - ``method="zscore"``: threshold = mean + k*std (online moments; the
      threshold itself needs no histogram, but ratio/severity do).
    - ``method="percentile"``: threshold = the p-th percentile of the per-token
      abs_max histogram.

    Metrics that need the histogram (ratio, severity, percentile threshold)
    return NaN when it was not collected (single-pass calibration); ``max_abs``
    and the zscore threshold remain available from online moments.
    """
    count = getattr(token_stats, "count", 0)
    if count == 0:
        return {"method": method, "threshold": float("nan"),
                "outlier_ratio": float("nan"), "severity": float("nan"),
                "max_abs": float("nan")}
    mean = float(getattr(token_stats, "mean", float("nan")))
    std = float(getattr(token_stats, "std", float("nan")))
    max_abs = float(getattr(token_stats, "max", float("nan")))

    hist = getattr(token_stats, "abs_max_hist", None)
    counts_list = edges_list = None
    total = 0.0
    if hist is not None:
        hr = hist.to_result()
        counts_list = hr.counts
        edges_list = hr.bin_edges
        total = float(sum(counts_list))

    if method == "percentile":
        threshold = _hist_percentile(counts_list, edges_list, percentile) \
            if total > 0 else float("nan")
    else:  # zscore
        threshold = (mean + k * std) if std == std else float("nan")

    outlier_ratio = float("nan")
    if total > 0 and threshold == threshold:
        counts = np.asarray(counts_list, dtype=np.float64)
        edges = np.asarray(edges_list, dtype=np.float64)
        # fraction of tokens with abs_max >= threshold: count bins whose upper
        # edge > threshold (a bin spans values up to its upper edge, so it may
        # hold values >= threshold; using centers would miss the top bin when
        # the threshold falls in its upper half).
        outlier_ratio = float(counts[edges[1:] > threshold].sum() / total)

    severity = float("nan")
    if total > 0:
        median = _hist_percentile(counts_list, edges_list, 50.0)
        if median == median and median > 0 and max_abs == max_abs:
            severity = float(max_abs / median)

    return {
        "method": method,
        "threshold": threshold,
        "outlier_ratio": outlier_ratio,
        "severity": severity,
        "max_abs": max_abs,
    }
