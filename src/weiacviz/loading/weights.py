"""Weight tensor access with per-tensor/channel/group slicing."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..shared.types import Granularity


def get_weight(model: nn.Module, module_path: str) -> torch.Tensor:
    """Read a module's weight tensor (offline, no inference).

    For modules offloaded by ``accelerate`` (``device_map=auto`` on a model
    larger than the GPU), ``module.weight`` is a meta placeholder; the real
    weight is recovered from the offload hook's ``weights_map`` (it lives on
    CPU). Raises if recovery is impossible.
    """
    mod = model.get_submodule(module_path)
    w = mod.weight
    if w.is_meta:
        hook = getattr(mod, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is not None:
            try:
                return weights_map["weight"].detach()
            except Exception:  # noqa: BLE001 -- fall through to the error
                pass
        raise RuntimeError(
            f"weight of '{module_path}' is on the meta device (accelerate "
            f"offload) and could not be recovered from the offload hook")
    return w.detach()


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
