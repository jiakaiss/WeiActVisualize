"""Batched inference runner with online activation aggregation.

Key design: activations are folded into running statistics per batch and the
raw tensors released, so memory usage is independent of the number of batches
and decoupled from model/calibration scale.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .hook import ActivationCapture
from ..shared.types import CaptureConfig


class RunningStats:
    """Online aggregation of tensor statistics without retaining raw tensors."""

    def __init__(self):
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.sum = 0.0
        self.sumsq = 0.0
        self.abs_max = 0.0

    def update(self, t: torch.Tensor) -> None:
        t = t.detach().float()
        n = t.numel()
        if n == 0:
            return
        self.count += n
        self.min = min(self.min, t.min().item())
        self.max = max(self.max, t.max().item())
        self.sum += t.sum().item()
        self.sumsq += (t * t).sum().item()
        self.abs_max = max(self.abs_max, t.abs().max().item())

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count == 0:
            return 0.0
        var = self.sumsq / self.count - self.mean ** 2
        return var ** 0.5 if var > 0 else 0.0

    def to_dict(self) -> dict:
        return {"count": self.count, "min": self.min, "max": self.max,
                "mean": self.mean, "std": self.std, "abs_max": self.abs_max}


class OnlineAggregator:
    """Per-module per-role running stats, updated batch by batch."""

    def __init__(self, module_paths: List[str]):
        self.paths = module_paths
        self.stats: Dict[str, Dict[str, RunningStats]] = {
            p: {"input": RunningStats(), "output": RunningStats()} for p in module_paths
        }

    def update_from_capture(self, capture: ActivationCapture) -> None:
        for path, entry in capture.buffer.items():
            if path not in self.stats:
                continue
            for role, tensor in entry.items():
                self.stats[path][role].update(tensor)

    def to_dict(self) -> Dict[str, Dict[str, dict]]:
        return {p: {r: s.to_dict() for r, s in roles.items()}
                for p, roles in self.stats.items()}


def run_calibration(
    model, tokenizer, texts: List[str], module_paths: List[str],
    config: Optional[CaptureConfig] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    seq_length: int = 2048,
) -> OnlineAggregator:
    """Run batched calibration inference, aggregating activation stats online.

    Memory is independent of the number of batches: raw activations are
    released after each batch's stats are folded into the aggregator.
    """
    cfg = config or CaptureConfig(max_samples=len(texts))
    texts = texts[: cfg.max_samples]
    aggregator = OnlineAggregator(module_paths)
    capture = ActivationCapture(module_paths, cfg.capture_inputs, cfg.capture_outputs)
    capture.attach(model)

    batch_size = max(1, cfg.batch_size)
    n_batches = (len(texts) + batch_size - 1) // batch_size
    device = getattr(model, "device", torch.device("cpu"))

    try:
        with torch.no_grad():
            done = 0
            for bi in range(0, len(texts), batch_size):
                batch_texts = texts[bi:bi + batch_size]
                enc = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=seq_length,
                )
                input_ids = enc["input_ids"].to(device)
                model(input_ids)
                aggregator.update_from_capture(capture)
                capture.clear()
                done += 1
                if progress_cb is not None:
                    progress_cb(done, n_batches)
    finally:
        capture.detach()
    return aggregator
