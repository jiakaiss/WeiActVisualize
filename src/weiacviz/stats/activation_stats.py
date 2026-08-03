"""Activation distribution statistics: per-token / per-channel."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..shared.types import StatLevel, StatResult, TensorRole
from ._util import to_numpy
from .shape import shape_label


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


def activation_outlier_channels(channel_stats, k: float = 5.0) -> dict:
    """Detect outlier hidden channels by per-channel abs_mean (SmoothQuant diag).

    ``channel_stats`` is duck-typed (a RunningChannelStats) or its to_result()
    dict. Returns outlier_ratio (fraction of channels with abs_mean > k*median),
    severity (max(abs_mean)/median(abs_mean)), and the top-10 channel indices.
    High severity / outlier_ratio => a few channels dominate the activation
    range => SmoothQuant (outlier migration) may help.
    """
    res = channel_stats.to_result() if hasattr(channel_stats, "to_result") else channel_stats
    if not res or not res.get("abs_mean"):
        return {"outlier_ratio": float("nan"), "severity": float("nan"),
                "top_channels": [], "median_abs_mean": float("nan")}
    am = np.asarray(res["abs_mean"], dtype=np.float64)
    if am.size == 0:
        return {"outlier_ratio": float("nan"), "severity": float("nan"),
                "top_channels": [], "median_abs_mean": float("nan")}
    med = float(np.median(am))
    if med <= 0:
        return {"outlier_ratio": 0.0, "severity": float("nan"),
                "top_channels": [], "median_abs_mean": med}
    mask = am > k * med
    return {
        "outlier_ratio": float(mask.sum() / am.size),
        "severity": float(am.max() / med),
        "top_channels": np.argsort(am)[::-1][:10].tolist(),
        "median_abs_mean": med,
    }
