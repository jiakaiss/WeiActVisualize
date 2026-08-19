"""Resolve and classify target modules (attention / MLP linears) per architecture family."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch.nn as nn

from ..shared.types import ModuleInfo, ModuleKind

logger = logging.getLogger(__name__)

# Projection suffixes by architecture family. Llama-like naming
# (q/k/v/o_proj + gate/up/down_proj) covers Llama, Qwen, Mistral, DeepSeek;
# the rest cover common non-LLama naming (GPT-2 fused, BERT, DiT).
_ARCH_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "llama": {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
              "mlp": ["gate_proj", "up_proj", "down_proj"]},
    "qwen": {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
             "mlp": ["gate_proj", "up_proj", "down_proj"]},
    "mistral": {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mlp": ["gate_proj", "up_proj", "down_proj"]},
    "deepseek": {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "mlp": ["gate_proj", "up_proj", "down_proj"]},
    "gpt2": {"attn": ["c_attn", "c_proj"],
             "mlp": ["c_fc", "c_proj"]},
    "bert": {"attn": ["query", "key", "value"],
             "mlp": ["dense"]},
    "dit": {"attn": ["qkv", "q", "k", "v", "proj"],
            "mlp": ["fc1", "fc2", "adaLN_modulation"]},
}


@dataclass
class ResolveResult:
    modules: List[ModuleInfo]
    family: str
    degraded: bool


def detect_arch_family(model) -> str:
    """Infer architecture family from model config (architectures) first,
    falling back to the model class name only when config is absent."""
    config = getattr(model, "config", None)
    archs = getattr(config, "architectures", None) or []
    candidate = archs[0].lower() if archs else type(model).__name__.lower()
    for fam in _ARCH_PATTERNS:
        if fam in candidate:
            return fam
    return "unknown"


def _classify(name: str, patterns: Dict[str, List[str]]) -> ModuleKind:
    last = name.split(".")[-1]
    if last in patterns["attn"] and last in patterns["mlp"]:
        # Shared suffix (e.g. GPT-2 uses c_proj in both attn and mlp):
        # disambiguate by the parent module's name.
        parent = name.split(".")[-2].lower() if "." in name else ""
        return (ModuleKind.MLP if "mlp" in parent or "feed" in parent
                else ModuleKind.ATTENTION)
    if last in patterns["attn"]:
        return ModuleKind.ATTENTION
    if last in patterns["mlp"]:
        return ModuleKind.MLP
    return ModuleKind.OTHER


def resolve_modules(model, arch_family: Optional[str] = None) -> ResolveResult:
    """Enumerate target linear modules with kind classification.

    For known architecture families, only attention/MLP proj layers are returned.
    For unknown architectures, falls back to enumerating ALL nn.Linear modules
    (classified as OTHER) and records a degradation notice via logging.
    """
    family = arch_family or detect_arch_family(model)
    degraded = family not in _ARCH_PATTERNS
    patterns = _ARCH_PATTERNS.get(family, {"attn": [], "mlp": []})
    if degraded:
        logger.warning(
            "Unknown architecture family '%s'; falling back to enumerating "
            "all nn.Linear modules as OTHER.", family,
        )

    modules: List[ModuleInfo] = []
    for path, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        kind = ModuleKind.OTHER if degraded else _classify(path, patterns)
        if not degraded and kind == ModuleKind.OTHER:
            continue  # known family: keep only attn/mlp targets
        weight = mod.weight
        modules.append(ModuleInfo(
            path=path, kind=kind,
            shape=tuple(weight.shape), dtype=str(weight.dtype),
        ))
    return ResolveResult(modules=modules, family=family, degraded=degraded)
