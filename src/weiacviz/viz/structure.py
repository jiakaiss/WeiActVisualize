"""Model structure overview and per-module table building."""
from __future__ import annotations

import re
from typing import List

from ..loading.module_resolver import ResolveResult
from ..shared.types import ModuleKind

_LAYER_RE = re.compile(r"(?:layers|blocks)\.(\d+)\.")

_KIND_ORDER = {
    ModuleKind.ATTENTION: 0,
    ModuleKind.MLP: 1,
    ModuleKind.OTHER: 2,
}


def _layer_index(path: str) -> int:
    """Extract transformer layer index from a module path (e.g. layers.5. -> 5)."""
    m = _LAYER_RE.search(path)
    return int(m.group(1)) if m else -1


def _module_params(model, path: str) -> int:
    """Total parameter count (weight + bias) of the module at `path`."""
    mod = model.get_submodule(path)
    return sum(p.numel() for p in mod.parameters())


def build_module_table(resolve_result: ResolveResult, model) -> List[dict]:
    """Build per-module table rows sorted by layer index, then attn -> mlp."""
    rows: List[dict] = []
    for m in resolve_result.modules:
        rows.append({
            "layer": _layer_index(m.path),
            "path": m.path,
            "kind": m.kind.value,
            "shape": str(tuple(m.shape)),
            "dtype": m.dtype,
            "params": _module_params(model, m.path),
        })
    rows.sort(key=lambda r: (
        r["layer"],
        _KIND_ORDER.get(ModuleKind(r["kind"]), 3),
        r["path"],
    ))
    return rows


def build_overview(resolve_result: ResolveResult, model) -> str:
    """Build a markdown overview of the loaded model's structure."""
    total_params = sum(p.numel() for p in model.parameters())
    n_modules = len(resolve_result.modules)
    layer_ids = {
        _layer_index(m.path) for m in resolve_result.modules
        if _layer_index(m.path) >= 0
    }
    n_layers = len(layer_ids)
    degraded = "是" if resolve_result.degraded else "否"
    return (
        f"- 架构族: `{resolve_result.family}`\n"
        f"- 降级: {degraded}\n"
        f"- 全模型总参数量: {total_params:,}\n"
        f"- transformer 层数: {n_layers}\n"
        f"- 目标 module 数: {n_modules}"
    )
