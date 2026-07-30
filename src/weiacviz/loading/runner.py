"""Batched inference runner with online activation aggregation.

Key design: activations are folded into running statistics per batch and the
raw tensors released, so memory usage is independent of the number of batches
and decoupled from model/calibration scale.

With ``collect_histogram=True`` a two-pass scheme is used: Pass 1 collects
scalar running stats to fix each module/role's global [min, max] range, Pass 2
re-runs inference bucketing activations into an online histogram with that
fixed range. Memory stays O(stats + bins), independent of batch count.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .hook import ActivationCapture
from ..shared.types import CaptureConfig, HistogramResult
from ..stats._util import to_numpy


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


class RunningHistogram:
    """Online histogram with a FIXED value range, accumulated batch by batch.

    The range is fixed at construction (determined externally from a first
    Pass 1 of ``RunningStats``) so counts from every batch land in aligned
    bins. Memory is O(num_bins), independent of batch count / sequence length
    / model size. The model MUST be in eval mode for Pass 2 so values stay
    within Pass 1's [min, max]; any out-of-range values are silently dropped
    by ``np.histogram``.
    """

    def __init__(self, value_range: tuple, num_bins: int = 256):
        lo, hi = float(value_range[0]), float(value_range[1])
        if not (hi > lo):
            hi = lo + 1.0
        self.num_bins = int(num_bins)
        self.value_range = (lo, hi)
        self.counts = np.zeros(self.num_bins, dtype=np.int64)
        self.bin_edges = np.linspace(lo, hi, self.num_bins + 1).tolist()

    def update(self, t) -> None:
        arr = to_numpy(t).flatten()
        if arr.size == 0:
            return
        c, _ = np.histogram(arr, bins=self.num_bins, range=self.value_range)
        self.counts += c

    def to_result(self) -> HistogramResult:
        return HistogramResult(
            counts=self.counts.tolist(),
            bin_edges=self.bin_edges,
            num_bins=self.num_bins,
        )


class OnlineAggregator:
    """Per-module per-role running stats, updated batch by batch."""

    def __init__(self, module_paths: List[str]):
        self.paths = module_paths
        self.stats: Dict[str, Dict[str, RunningStats]] = {
            p: {"input": RunningStats(), "output": RunningStats()} for p in module_paths
        }
        self.histograms: Dict[str, Dict[str, Optional[RunningHistogram]]] = {
            p: {"input": None, "output": None} for p in module_paths
        }

    def update_from_capture(self, capture: ActivationCapture,
                            update_stats: bool = True,
                            update_histogram: bool = False) -> None:
        for path, entry in capture.buffer.items():
            if path not in self.stats:
                continue
            for role, tensor in entry.items():
                if update_stats:
                    self.stats[path][role].update(tensor)
                if update_histogram:
                    h = self.histograms[path][role]
                    if h is not None:
                        h.update(tensor)

    def build_histograms(self, num_bins: int) -> None:
        """Construct a RunningHistogram per module/role from Pass 1 stats range."""
        for p, roles in self.stats.items():
            for r, s in roles.items():
                if s.count == 0:
                    continue
                self.histograms[p][r] = RunningHistogram(
                    (s.min, s.max), num_bins=num_bins,
                )

    def to_dict(self) -> Dict[str, Dict[str, dict]]:
        out: Dict[str, Dict[str, dict]] = {}
        for p, roles in self.stats.items():
            out[p] = {}
            for r, s in roles.items():
                entry = dict(s.to_dict())
                h = self.histograms[p][r]
                if h is None:
                    entry["histogram"] = None
                else:
                    hr = h.to_result()
                    entry["histogram"] = {
                        "counts": hr.counts,
                        "bin_edges": hr.bin_edges,
                        "num_bins": hr.num_bins,
                    }
                out[p][r] = entry
        return out


def _run_pass(model, tokenizer, texts: List[str], batch_size: int, device,
              seq_length: int, aggregator: OnlineAggregator,
              capture: ActivationCapture, update_stats: bool,
              update_histogram: bool,
              progress_cb: Optional[Callable[[int, int], None]],
              done_offset: int, total: int) -> int:
    """Run one pass of batched inference, folding captures into the aggregator."""
    with torch.no_grad():
        done = done_offset
        for bi in range(0, len(texts), batch_size):
            batch_texts = texts[bi:bi + batch_size]
            enc = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=seq_length,
            )
            input_ids = enc["input_ids"].to(device)
            model(input_ids)
            aggregator.update_from_capture(
                capture, update_stats=update_stats,
                update_histogram=update_histogram,
            )
            capture.clear()
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)
    return done


def run_calibration(
    model, tokenizer, texts: List[str], module_paths: List[str],
    config: Optional[CaptureConfig] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    seq_length: int = 2048,
    collect_histogram: bool = False,
    num_bins: int = 256,
) -> OnlineAggregator:
    """Run batched calibration inference, aggregating activation stats online.

    Memory is independent of the number of batches: raw activations are
    released after each batch's stats are folded into the aggregator.

    When ``collect_histogram`` is True a two-pass scheme is used: Pass 1
    collects scalar running stats to fix each module/role's global [min, max],
    then Pass 2 re-runs inference bucketing activations into an online
    histogram with that fixed range. The model is set to eval() for the
    duration so Pass 2 stays within Pass 1's range; any out-of-range values
    are dropped by np.histogram.
    """
    cfg = config or CaptureConfig(max_samples=len(texts))
    texts = texts[: cfg.max_samples]
    aggregator = OnlineAggregator(module_paths)
    capture = ActivationCapture(module_paths, cfg.capture_inputs, cfg.capture_outputs)
    capture.attach(model)

    batch_size = max(1, cfg.batch_size)
    n_batches = (len(texts) + batch_size - 1) // batch_size
    device = getattr(model, "device", torch.device("cpu"))
    total = n_batches * (2 if collect_histogram else 1)

    was_training = getattr(model, "training", False)
    model.eval()

    try:
        _run_pass(model, tokenizer, texts, batch_size, device, seq_length,
                  aggregator, capture, update_stats=True, update_histogram=False,
                  progress_cb=progress_cb, done_offset=0, total=total)

        if collect_histogram:
            aggregator.build_histograms(num_bins)
            _run_pass(model, tokenizer, texts, batch_size, device, seq_length,
                      aggregator, capture, update_stats=False, update_histogram=True,
                      progress_cb=progress_cb, done_offset=n_batches, total=total)
    finally:
        capture.detach()
        if was_training:
            model.train()
    return aggregator
