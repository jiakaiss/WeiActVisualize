"""Filtering / drill-down helpers for the UI."""
from __future__ import annotations

from typing import List

from ..shared.types import ModuleKind


def filter_modules(modules, kind: str = "all", name_contains: str = "") -> list:
    """Filter module infos by kind (attn/mlp/other/all) and name substring."""
    out = []
    for m in modules:
        if kind != "all" and m.kind != ModuleKind(kind):
            continue
        if name_contains and name_contains not in m.path:
            continue
        out.append(m)
    return out


def filter_scheme_rows(rows: List[dict], bits: int = 0, granularity: str = "",
                       symmetry: str = "") -> List[dict]:
    """Filter quantization scheme comparison rows."""
    out = []
    for r in rows:
        if bits and r.get("bits") != bits:
            continue
        if granularity and r.get("granularity") != granularity:
            continue
        if symmetry and r.get("symmetry") != symmetry:
            continue
        out.append(r)
    return out
