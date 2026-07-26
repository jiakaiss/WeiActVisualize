"""Shared helpers for the stats module."""
from __future__ import annotations

import numpy as np


def to_numpy(t):
    """Convert torch.Tensor / np.ndarray / list to a float64 numpy array."""
    if hasattr(t, "detach"):
        # .float() upcasts bf16/fp16 -> fp32: numpy has no bfloat16 dtype, so
        # calling .numpy() on a bf16 tensor raises "unsupported ScalarType".
        arr = t.detach().cpu().float().numpy()
    elif hasattr(t, "numpy"):
        arr = t.numpy()
    else:
        arr = np.asarray(t)
    return np.asarray(arr, dtype=np.float64)
