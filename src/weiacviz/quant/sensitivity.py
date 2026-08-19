"""Sensitivity analysis: rank layers/channels by quantization loss."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .error_metrics import mse, output_diff
from .fake_quant import fake_quantize_tensor
from ..shared.types import QuantConfig


def _forwardable(module):
    """Return a module ``output_diff`` can run, substituting accelerate-
    offloaded modules. ``device_map=auto`` offload leaves ``module.weight``
    on the meta device and dispatches the real weight via forward hooks, so
    the weight-swap inside ``output_diff`` breaks. For those, build a plain
    CPU ``nn.Linear`` from the real weight recovered via the offload hook;
    the quantization error itself is unchanged. Returns the module unchanged
    when recovery fails (the caller's try/except degrades the row to NaN).
    """
    if not module.weight.is_meta:
        return module
    hook = getattr(module, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None:
        return module
    try:
        w = weights_map["weight"].detach().float()
        try:
            b = weights_map["bias"].detach().float()
        except Exception:  # noqa: BLE001 -- no bias on this module
            b = None
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        lin.weight = torch.nn.Parameter(w)
        if b is not None:
            lin.bias = torch.nn.Parameter(b)
        lin.eval()
        return lin
    except Exception:  # noqa: BLE001
        return module


def layer_sensitivity(weights: Dict[str, torch.Tensor], config: QuantConfig,
                      topk: int = 10) -> List[dict]:
    """Rank modules by per-layer quantization MSE (descending).

    The top-k highest-loss modules are flagged for attention.
    """
    rows: List[dict] = []
    for path, w in weights.items():
        q = fake_quantize_tensor(w, config)
        rows.append({"module_path": path, "mse": mse(w, q)})
    rows.sort(key=lambda r: r["mse"], reverse=True)
    if topk and len(rows) > topk:
        for r in rows[:topk]:
            r["top"] = True
    else:
        for r in rows[:topk]:
            r["top"] = True
    return rows


def layer_sensitivity_output(
    model, module_paths: List[str], weights: Dict[str, torch.Tensor],
    sample_inputs: Dict[str, torch.Tensor], config: QuantConfig,
    kinds: Optional[Dict[str, str]] = None, topk: int = 0, act_bits: int = 8,
) -> List[dict]:
    """Rank modules by quantization-induced output error (descending).

    Reports two output errors per module:
      - ``output_mse``: weight-only quantization (W4A16/W8A16 view) -- quantize
        the weight, keep the activation fp.
      - ``joint_output_mse``: W8A8 -- quantize the weight AND per-token quantize
        the activation. The gap (joint - weight-only) is the activation-side
        quantization loss, which is large exactly when outlier channels blow
        up the per-token scale (SmoothQuant territory).

    ``sample_inputs`` maps module path -> one input activation (from
    ``ModelAdapter.sample_inputs``). Modules without a sample input get NaN and
    sort last. Sorted by ``joint_output_mse`` descending.
    """
    kinds = kinds or {}
    rows: List[dict] = []
    for path in module_paths:
        w = weights[path]
        q = fake_quantize_tensor(w, config)
        inp = sample_inputs.get(path)
        out_mse = float("nan")
        joint_mse = float("nan")
        if inp is not None:
            try:
                module = _forwardable(model.get_submodule(path))
                out_mse = output_diff(module, q, inp,
                                      quantize_activation=False)["mse"]
                joint_mse = output_diff(module, q, inp,
                                        quantize_activation=True,
                                        act_bits=act_bits)["mse"]
            except Exception:  # noqa: BLE001
                pass
        rows.append({
            "module_path": path, "kind": kinds.get(path, ""),
            "output_mse": out_mse, "joint_output_mse": joint_mse,
        })
    rows.sort(key=lambda r: r["joint_output_mse"]
              if r["joint_output_mse"] == r["joint_output_mse"]
              else float("-inf"), reverse=True)
    if topk:
        for r in rows[:topk]:
            r["top"] = True
    return rows


def layer_sensitivity_output_multi(
    model, module_paths: List[str], weights: Dict[str, torch.Tensor],
    input_batches, config: QuantConfig,
    kinds: Optional[Dict[str, str]] = None, topk: int = 0, act_bits: int = 8,
) -> List[dict]:
    """Multi-sample version of :func:`layer_sensitivity_output`.

    ``input_batches`` is an iterable of ``{path: input activation}`` dicts
    (see ``ModelAdapter.sample_inputs_batched``). Per-module ``output_mse`` /
    ``joint_output_mse`` are **averaged over the batches** that provided an
    input for that module, so the ranking reflects the calibration
    distribution rather than one fixed sample.

    The fake-quantized weight is recomputed per (module, batch) instead of
    being precomputed for ALL modules up front: holding one q-weight per
    module simultaneously is another full copy of the weights and OOMs on
    models that near GPU capacity (e.g. a 7B fp16 on 16 GB). Only one
    q-weight is alive at a time; the extra quantize passes are cheap next to
    the output_diff forwards.

    Row format matches :func:`layer_sensitivity_output` plus ``n_samples``
    (how many batches actually contributed; modules never captured get NaN
    mse and ``n_samples=0``).
    """
    kinds = kinds or {}
    sums = {p: [0.0, 0.0] for p in module_paths}  # [sum output_mse, sum joint]
    counts = {p: 0 for p in module_paths}
    for inputs in input_batches:
        for path in module_paths:
            inp = inputs.get(path)
            if inp is None:
                continue
            try:
                module = _forwardable(model.get_submodule(path))
                q = fake_quantize_tensor(weights[path], config)
                sums[path][0] += output_diff(
                    module, q, inp,
                    quantize_activation=False)["mse"]
                sums[path][1] += output_diff(
                    module, q, inp,
                    quantize_activation=True, act_bits=act_bits)["mse"]
                counts[path] += 1
            except Exception:  # noqa: BLE001
                pass
    rows: List[dict] = []
    for path in module_paths:
        n = counts[path]
        rows.append({
            "module_path": path, "kind": kinds.get(path, ""),
            "output_mse": sums[path][0] / n if n else float("nan"),
            "joint_output_mse": sums[path][1] / n if n else float("nan"),
            "n_samples": n,
        })
    rows.sort(key=lambda r: r["joint_output_mse"]
              if r["joint_output_mse"] == r["joint_output_mse"]
              else float("-inf"), reverse=True)
    if topk:
        for r in rows[:topk]:
            r["top"] = True
    return rows
