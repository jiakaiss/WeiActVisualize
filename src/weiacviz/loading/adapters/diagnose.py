"""Adapter self-check: verify a custom ``ModelAdapter`` before wiring it in.

``verify_adapter`` walks the same paths the pipeline uses (module resolution,
calibration stream, input capture, output_diff sensitivity) and reports what
works and what degrades. Run it right after writing an adapter -- the most
common hidden bug is a non-deterministic ``calib_batches`` (two-pass
calibration replays the same data, so the stream must be reproducible).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import torch

from ...quant.error_metrics import output_diff
from ...quant.fake_quant import fake_quantize_tensor
from ...shared.types import Granularity, QuantConfig, Symmetry
from ..adapter import ModelAdapter
from ..weights import get_weight


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class VerifyReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def format(self) -> str:
        lines = ["adapter verification:"]
        lines += [f"  {'PASS' if c.ok else 'FAIL'}  {c.name}: {c.detail}"
                  for c in self.checks]
        lines.append("  -> all checks passed" if self.all_ok
                     else "  -> fix FAIL items above (see adapters/README.md)")
        return "\n".join(lines)

    def __str__(self) -> str:  # so `print(report)` just works
        return self.format()


def _same_batch(a: Any, b: Any) -> bool:
    """Structural equality of two calibration batches (tensor tree)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_batch(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_batch(x, y) for x, y in zip(a, b))
    return a == b


def verify_adapter(adapter: ModelAdapter, n_samples: int = 4,
                   batch_size: int = 2) -> VerifyReport:
    """Run the pipeline's entry points against ``adapter`` and collect a
    per-step report: module enumeration, calibration determinism, input
    capture, and output_diff sensitivity (the step most likely to degrade
    for layers that cannot forward independently)."""
    rep = VerifyReport()

    # 1. module enumeration -- family / degradation / kind split
    result = adapter.enumerate_modules()
    paths = [m.path for m in result.modules]
    kinds: dict = {}
    for m in result.modules:
        kinds[m.kind.name] = kinds.get(m.kind.name, 0) + 1
    kind_str = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"
    rep.checks.append(CheckResult(
        "enumerate_modules",
        len(paths) > 0,
        f"family={result.family}, {len(paths)} linear modules ({kind_str})"
        + (" [degraded: unknown family, all Linears listed]" if result.degraded else ""),
    ))

    # 2. calibration stream determinism (two-pass calibration replays it)
    if not adapter.has_calibration_data:
        rep.checks.append(CheckResult(
            "calib_batches", False,
            "no calibration data yet (HF adapters: call set_texts() first)",
        ))
        rep.checks.append(CheckResult("input_capture", False, "skipped"))
        rep.checks.append(CheckResult("output_diff", False, "skipped"))
        return rep
    try:
        pass1 = list(adapter.calib_batches(n_samples, batch_size))
        pass2 = list(adapter.calib_batches(n_samples, batch_size))
        det = (len(pass1) > 0 and len(pass1) == len(pass2)
               and all(_same_batch(a, b) for a, b in zip(pass1, pass2)))
        rep.checks.append(CheckResult(
            "calib_batches", det,
            f"{len(pass1)} batch(es) for n_samples={n_samples}"
            + ("" if det else " -- NOT deterministic; seed a torch.Generator "
                             "inside calib_batches (two-pass calibration replays "
                             "the same stream)"),
        ))
    except NotImplementedError:
        rep.checks.append(CheckResult(
            "calib_batches", False, "not implemented (override required)"))
        return rep

    # 3. input capture -- one forward, hook-collected per-module inputs
    try:
        inputs = adapter.sample_inputs(paths)
        got = len(inputs)
        rep.checks.append(CheckResult(
            "input_capture", got > 0,
            f"captured inputs for {got}/{len(paths)} modules"
            + ("" if got == len(paths) else " (missed modules get NaN sensitivity)"),
        ))
    except Exception as e:  # noqa: BLE001 -- report, don't crash
        rep.checks.append(CheckResult("input_capture", False, f"{type(e).__name__}: {e}"))
        rep.checks.append(CheckResult("output_diff", False, "skipped"))
        return rep

    # 4. output_diff sensitivity on the first captured module
    cfg = QuantConfig(bits=4, granularity=Granularity.PER_CHANNEL,
                      symmetry=Symmetry.SYMMETRIC)
    path = next((p for p in paths if p in inputs), None)
    if path is None:
        rep.checks.append(CheckResult(
            "output_diff", False, "no module input captured; cannot measure"))
        return rep
    try:
        module = adapter.model.get_submodule(path)
        q = fake_quantize_tensor(get_weight(adapter.model, path), cfg)
        mse = output_diff(module, q, inputs[path], quantize_activation=False)["mse"]
        ok = mse == mse  # NaN means the layer cannot forward independently
        rep.checks.append(CheckResult(
            "output_diff", ok,
            f"{path}: mse={mse:.3e}" if ok else
            f"{path}: NaN (layer cannot forward independently; weight/activation "
            f"analysis still work, sensitivity degrades)"))
    except Exception as e:  # noqa: BLE001
        rep.checks.append(CheckResult(
            "output_diff", False,
            f"{path}: {type(e).__name__}: {e} (sensitivity degrades for this layer)"))
    return rep
