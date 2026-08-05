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


class RunningTokenStats:
    """Online aggregation of per-token activation magnitudes.

    For an activation ``[..., seq, hidden]``, reshapes to ``[N, hidden]`` and
    reduces along the hidden axis to get one ``abs_max`` (and optionally
    ``abs_mean``) per token. Those scalars are folded into running raw moments
    (S1..S4) so the distribution shape (mean / std / CV / kurtosis / skewness)
    is available in a **single pass** with O(1) memory.

    When ``build_histogram`` is called (after a Pass-1 of moments, riding the
    existing two-pass scheme), per-token abs_max is also bucketed into a
    ``RunningHistogram`` for visualization and percentile-based tail_ratio.

    The distribution of per-token abs_max answers whether per-tensor
    activation quantization is enough (concentrated: low CV, no heavy tail)
    or whether per-token quantization is needed (spread / heavy-tailed) --
    the standard W8A8 per-token activation decision.
    """

    def __init__(self, num_bins: int = 256, collect_abs_mean: bool = False):
        self.count = 0
        self.s1 = 0.0  # sum, sum(x^2..4) of per-token abs_max
        self.s2 = 0.0
        self.s3 = 0.0
        self.s4 = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self._num_bins = int(num_bins)
        self.abs_max_hist: Optional[RunningHistogram] = None
        self.collect_abs_mean = collect_abs_mean
        self.abs_mean_stats = RunningStats() if collect_abs_mean else None

    def update_moments(self, t: torch.Tensor) -> None:
        """Pass 1: accumulate raw moments (and abs_mean) of per-token abs_max."""
        per_token = self._per_token_abs_max(t)
        n = per_token.numel()
        if n == 0:
            return
        a = per_token.cpu().numpy().astype(np.float64)
        self.count += n
        self.s1 += float(a.sum())
        self.s2 += float((a * a).sum())
        self.s3 += float((a ** 3).sum())
        self.s4 += float((a ** 4).sum())
        self.min = min(self.min, float(a.min()))
        self.max = max(self.max, float(a.max()))
        if self.abs_mean_stats is not None:
            tt = t.detach().float()
            if tt.dim() < 2:
                tt = tt.reshape(1, -1)
            self.abs_mean_stats.update(tt.abs().mean(dim=-1).reshape(-1))

    def update_histogram(self, t: torch.Tensor) -> None:
        """Pass 2: bucket per-token abs_max into the fixed-range histogram."""
        if self.abs_max_hist is None:
            return
        self.abs_max_hist.update(self._per_token_abs_max(t))

    @staticmethod
    def _per_token_abs_max(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().float()
        if t.dim() < 2:
            t = t.reshape(1, -1)
        return t.abs().amax(dim=-1).reshape(-1)

    def build_histogram(self) -> None:
        """Construct the per-token abs_max histogram from the Pass-1 moment range."""
        if self.count == 0:
            return
        self.abs_max_hist = RunningHistogram((self.min, self.max), num_bins=self._num_bins)

    @property
    def mean(self) -> float:
        return self.s1 / self.count if self.count else 0.0

    @property
    def var(self) -> float:
        if self.count == 0:
            return 0.0
        v = self.s2 / self.count - self.mean ** 2
        return v if v > 0 else 0.0

    @property
    def std(self) -> float:
        return self.var ** 0.5

    @property
    def cv(self) -> float:
        """Coefficient of variation of per-token abs_max (std/mean)."""
        return self.std / self.mean if self.mean > 0 else float("nan")

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis of per-token abs_max (population, biased)."""
        if self.count == 0 or self.var <= 0:
            return float("nan")
        n = self.count
        m = self.mean
        # central mu4 from raw moments: only the S_k terms are divided by n,
        # the pure-mean terms (-3 m^4) are not.
        mu4 = self.s4 / n - 4 * m * self.s3 / n + 6 * m * m * self.s2 / n - 3 * m ** 4
        return mu4 / (self.var * self.var) - 3.0

    @property
    def skewness(self) -> float:
        if self.count == 0 or self.var <= 0:
            return float("nan")
        n = self.count
        m = self.mean
        mu3 = self.s3 / n - 3 * m * self.s2 / n + 2 * m ** 3
        return mu3 / (self.var ** 1.5)

    def to_result(self) -> dict:
        h = None
        if self.abs_max_hist is not None:
            hr = self.abs_max_hist.to_result()
            h = {"counts": hr.counts, "bin_edges": hr.bin_edges, "num_bins": hr.num_bins}
        return {
            "count": self.count, "min": self.min, "max": self.max,
            "mean": self.mean, "std": self.std, "cv": self.cv,
            "kurtosis": self.kurtosis, "skewness": self.skewness,
            "histogram": h,
            "abs_mean": self.abs_mean_stats.to_dict() if self.abs_mean_stats else None,
        }


class RunningChannelStats:
    """Online per-channel (hidden-dim) activation magnitude aggregation.

    For an activation ``[..., seq, hidden]``, reshapes to ``[N, hidden]`` and
    accumulates per-channel ``abs_sum`` and ``abs_max`` along the token axis,
    so each hidden channel gets an ``abs_mean`` (= abs_sum/count) and
    ``abs_max``. Memory is O(hidden), independent of batch count.

    This surfaces outlier channels (a few hidden channels with abnormally
    large abs_mean) -- the SmoothQuant motivation. Default OFF: per-channel
    activation quantization has limited inference-engine support, so this is
    a SmoothQuant-suitability diagnostic, not the main activation analysis.
    """

    def __init__(self):
        self.abs_sum = None  # np.ndarray [hidden], lazy alloc on first update
        self.abs_max = None  # np.ndarray [hidden]
        self.count = 0  # token count (same for all channels)
        self.hidden = None

    def update(self, t) -> None:
        a = to_numpy(t)
        if a.ndim < 2:
            a = a.reshape(1, -1)
        flat = a.reshape(-1, a.shape[-1])  # [N, hidden]
        n, h = flat.shape
        if n == 0:
            return
        absv = np.abs(flat).astype(np.float64)
        if self.abs_sum is None:
            self.abs_sum = np.zeros(h, dtype=np.float64)
            self.abs_max = np.zeros(h, dtype=np.float64)
            self.hidden = h
        if h != self.hidden:
            return  # shape mismatch across batches for this module; skip
        self.abs_sum += absv.sum(axis=0)
        self.abs_max = np.maximum(self.abs_max, absv.max(axis=0))
        self.count += n

    @property
    def abs_mean(self):
        return (self.abs_sum / self.count) if (self.abs_sum is not None and self.count) else None

    def to_result(self):
        if self.abs_sum is None:
            return None
        am = self.abs_mean
        return {
            "hidden": self.hidden,
            "count": self.count,
            "abs_mean": am.tolist() if am is not None else [],
            "abs_max": self.abs_max.tolist(),
        }


class OnlineAggregator:
    """Per-module per-role running stats, updated batch by batch.

    Optionally also tracks per-token activation magnitudes (``token_stats``)
    for the per-token activation quantization decision (block A), and/or
    per-channel (hidden-dim) abs_mean (``channel_stats``) as a SmoothQuant
    outlier-channel diagnostic (default off).
    """

    def __init__(self, module_paths: List[str], collect_token_stats: bool = False,
                 token_bins: int = 256, collect_channel_stats: bool = False):
        self.paths = module_paths
        self.stats: Dict[str, Dict[str, RunningStats]] = {
            p: {"input": RunningStats(), "output": RunningStats()} for p in module_paths
        }
        self.histograms: Dict[str, Dict[str, Optional[RunningHistogram]]] = {
            p: {"input": None, "output": None} for p in module_paths
        }
        self.collect_token_stats = collect_token_stats
        self.token_bins = token_bins
        self.token_stats: Dict[str, Dict[str, Optional[RunningTokenStats]]] = {
            p: {"input": None, "output": None} for p in module_paths
        }
        if collect_token_stats:
            for p in module_paths:
                self.token_stats[p] = {
                    "input": RunningTokenStats(num_bins=token_bins),
                    "output": RunningTokenStats(num_bins=token_bins),
                }
        self.collect_channel_stats = collect_channel_stats
        self.channel_stats: Dict[str, Dict[str, Optional[RunningChannelStats]]] = {
            p: {"input": None, "output": None} for p in module_paths
        }
        if collect_channel_stats:
            for p in module_paths:
                self.channel_stats[p] = {
                    "input": RunningChannelStats(),
                    "output": RunningChannelStats(),
                }

    def update_from_capture(self, capture: ActivationCapture,
                            update_stats: bool = True,
                            update_histogram: bool = False) -> None:
        for path, entry in capture.buffer.items():
            if path not in self.stats:
                continue
            for role, tensor in entry.items():
                ts = self.token_stats[path][role]
                cs = self.channel_stats[path][role]
                if update_stats:
                    self.stats[path][role].update(tensor)
                    if ts is not None:
                        ts.update_moments(tensor)
                    if cs is not None:
                        cs.update(tensor)
                if update_histogram:
                    h = self.histograms[path][role]
                    if h is not None:
                        h.update(tensor)
                    if ts is not None and ts.abs_max_hist is not None:
                        ts.update_histogram(tensor)

    def build_histograms(self, num_bins: int) -> None:
        """Construct a RunningHistogram per module/role from Pass 1 stats range."""
        for p, roles in self.stats.items():
            for r, s in roles.items():
                if s.count == 0:
                    continue
                self.histograms[p][r] = RunningHistogram(
                    (s.min, s.max), num_bins=num_bins,
                )
                ts = self.token_stats[p][r]
                if ts is not None:
                    ts.build_histogram()

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
                ts = self.token_stats[p][r]
                entry["token_stats"] = ts.to_result() if ts is not None else None
                cs = self.channel_stats[p][r]
                entry["channel_stats"] = cs.to_result() if cs is not None else None
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
    collect_token_stats: bool = True,
    collect_channel_stats: bool = False,
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

    When ``collect_token_stats`` is True (default), per-token abs_max moments
    are aggregated in Pass 1 (O(1) memory) for the per-token activation
    quantization decision; if ``collect_histogram`` is also True the per-token
    abs_max histogram is filled in Pass 2 for visualization.

    When ``collect_channel_stats`` is True, per-channel (hidden-dim) abs_mean
    is aggregated (O(hidden) memory) as a SmoothQuant outlier-channel
    diagnostic.
    """
    cfg = config or CaptureConfig(max_samples=len(texts))
    texts = texts[: cfg.max_samples]
    aggregator = OnlineAggregator(
        module_paths, collect_token_stats=collect_token_stats,
        token_bins=num_bins, collect_channel_stats=collect_channel_stats,
    )
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


def capture_sample_inputs(model, tokenizer, module_paths: List[str],
                          text: str = "", seq_length: int = 128,
                          ) -> Dict[str, torch.Tensor]:
    """Run one forward pass and return one input activation per target module.

    Used by sensitivity analysis (output_diff): provides a real input tensor
    for each module so quantization error can be measured at the module output
    rather than just the weight. Memory is one batch per module; the caller
    releases the tensors when done.
    """
    cap = ActivationCapture(module_paths, capture_inputs=True, capture_outputs=False)
    cap.attach(model)
    device = getattr(model, "device", torch.device("cpu"))
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        enc = tokenizer([text] if text else ["test"], return_tensors="pt",
                        padding=True, truncation=True, max_length=seq_length)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids)
        inputs: Dict[str, torch.Tensor] = {}
        for p in module_paths:
            entry = cap.buffer.get(p, {})
            if "input" in entry:
                inputs[p] = entry["input"]
    finally:
        cap.detach()
        if was_training:
            model.train()
    return inputs


def capture_module_output_sample(model, tokenizer, module_path: str,
                                 text: str = "", seq_length: int = 512,
                                 ) -> Optional[torch.Tensor]:
    """Run one forward pass and return one module's output activation.

    Used by per-token slice visualization: provides a real activation sample
    for the viewed module so per-token hidden-dim distributions can be rendered
    with the same violin + heatmap pipeline as weights. Memory is one tensor
    of ``[batch, seq, hidden]``; the caller releases it when done.

    Uses an independent ``ActivationCapture`` registration (output only), so it
    does not touch any calibration ``OnlineAggregator``. Returns ``None`` if the
    module produced no capturable output.
    """
    cap = ActivationCapture([module_path], capture_inputs=False, capture_outputs=True)
    cap.attach(model)
    device = getattr(model, "device", torch.device("cpu"))
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        enc = tokenizer([text] if text else ["test"], return_tensors="pt",
                        padding=True, truncation=True, max_length=seq_length)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids)
        entry = cap.buffer.get(module_path, {})
        if "output" not in entry:
            return None
        return entry["output"]
    finally:
        cap.detach()
        if was_training:
            model.train()
