"""Weight tensor access with per-tensor/channel/group slicing."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..shared.types import Granularity


def get_weight(model: nn.Module, module_path: str) -> torch.Tensor:
    """Read a module's weight tensor (offline, no inference)."""
    mod = model.get_submodule(module_path)
    return mod.weight.detach()


def slice_weight(weight: torch.Tensor, granularity: Granularity,
                 group_size: Optional[int] = None) -> List[torch.Tensor]:
    """Slice a 2D weight [out_features, in_features] by granularity.

    Returns a list of slices usable for per-axis statistics.
    """
    if weight.dim() <= 1:
        return [weight.flatten()]
    if granularity == Granularity.PER_TENSOR:
        return [weight.flatten()]
    if granularity == Granularity.PER_CHANNEL:
        return [weight[i] for i in range(weight.shape[0])]
    if granularity == Granularity.PER_GROUP:
        gs = group_size or 128
        flat = weight.reshape(-1)
        return [flat[i:i + gs] for i in range(0, flat.numel(), gs)]
    return [weight.flatten()]
