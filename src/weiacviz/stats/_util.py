"""Shared helpers for the stats module."""
from __future__ import annotations

import numpy as np


def to_numpy(t):
    """Convert torch.Tensor / np.ndarray / list to a float64 numpy array."""
    if hasattr(t, "detach"):
        arr = t.detach().cpu().numpy()
    elif hasattr(t, "numpy"):
        arr = t.numpy()
    else:
        arr = np.asarray(t)
    return np.asarray(arr, dtype=np.float64)
