"""Distribution distance metrics: KL divergence and Wasserstein."""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from ._util import to_numpy


def _as_distribution(t, num_bins: int = 256):
    a = to_numpy(t).flatten()
    if a.size == 0:
        return np.ones(num_bins) / num_bins, (0.0, 1.0)
    lo, hi = float(a.min()), float(a.max())
    if lo == hi:
        hi = lo + 1.0
    hist, _ = np.histogram(a, bins=num_bins, range=(lo, hi), density=True)
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist, (lo, hi)


def kl_divergence(p, q, num_bins: int = 256, eps: float = 1e-12) -> float:
    """KL(p || q) discretised over each sample's own range."""
    pa, _ = _as_distribution(p, num_bins)
    qb, _ = _as_distribution(q, num_bins)
    n = min(len(pa), len(qb))
    pa, qb = pa[:n] + eps, qb[:n] + eps
    pa /= pa.sum()
    qb /= qb.sum()
    return float(np.sum(pa * np.log(pa / qb)))


def wasserstein(p, q) -> float:
    """Wasserstein-1 (earth mover's) distance between two samples."""
    return float(wasserstein_distance(to_numpy(p).flatten(), to_numpy(q).flatten()))
