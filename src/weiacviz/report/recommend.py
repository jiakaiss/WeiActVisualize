"""Generate quantization suitability recommendations from sensitivity + shape.

Block D rule engine: combines W8A8 joint output sensitivity (block C,
weight + per-token activation quantization), weight shape (heavy channels),
and activation outlier severity (block A, diagnostic) to give a per-module
recommendation with a readable "why + how" reason.

This is a distribution-diagnosis tool, not an algorithm recommender: the
reason describes *why* a layer is hard to quantize (weight heavy channels,
activation outliers, activation-loss share) so the user can judge which PTQ
algorithm (GPTQ / AWQ / SmoothQuant / ...) is likely to help. Activation
quantization is assumed per-token (industry W8A8 default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..shared.types import QuantConfig

# Heuristic thresholds (calibrate on real models).
HEAVY_TAIL_KURTOSIS = 3.0             # per-channel excess kurtosis > 3 => "heavy" channel
HEAVY_CHANNEL_RATIO_THRESHOLD = 0.05  # >5% heavy channels => per-group worth it
SKEWNESS_THRESHOLD = 0.5              # |skew| > 0.5 => asymmetric quant may help
ACTIVATION_OUTLIER_NOTABLE = 5.0      # severity above this => notable act outliers (diagnostic)


@dataclass
class ModuleRecommendation:
    module_path: str
    kind: str = ""
    recommended_bits: int = 4
    recommended_granularity: str = "per-channel"
    recommended_symmetry: str = "symmetric"
    output_mse: float = float("nan")        # weight-only (W4A16)
    joint_output_mse: float = float("nan")  # W8A8 (weight + per-token act)
    weight_kurtosis_max: float = float("nan")   # max per-channel excess kurtosis
    heavy_channel_ratio: float = float("nan")   # fraction of channels with kurtosis > 3
    act_channel_severity: float = float("nan")  # activation outlier severity (diagnostic)
    reason: str = ""


@dataclass
class RecommendationReport:
    model: str
    config: dict
    recommendations: List[ModuleRecommendation] = field(default_factory=list)
    summary: str = ""


def _is_num(x) -> bool:
    return x is not None and x == x  # NaN check


def recommend(
    sensitivity_rows: List[dict],
    model_name: str = "",
    config: Optional[QuantConfig] = None,
) -> RecommendationReport:
    """Produce per-module quantization recommendations.

    Each ``sensitivity_row`` carries (from blocks A+C): ``module_path``,
    ``kind``, ``output_mse`` (weight-only), ``joint_output_mse`` (W8A8),
    ``weight_kurtosis_max``, ``heavy_channel_ratio``, ``weight_skewness``,
    and ``act_channel_severity``. Missing/NaN degrade gracefully.

    Sensitivity driver is ``joint_output_mse`` (W8A8) -- it already includes
    activation quantization loss.

    Weight-scheme rules:
      - high joint sensitivity + heavy weight channels (>5%) -> W8 / per-group(128)
      - high joint sensitivity (other)                        -> W8 / per-channel
      - low sensitivity                                        -> W4 / per-channel
      - |weight skew| > 0.5                                    -> asymmetric

    The reason also reports activation-loss share and activation outlier
    severity as *diagnostic* signals (which PTQ algorithm may help), but does
    NOT recommend a specific algorithm.
    """
    joint_mses = [r["joint_output_mse"] for r in sensitivity_rows
                  if _is_num(r.get("joint_output_mse"))]
    median_joint = float(np.median(joint_mses)) if joint_mses else 0.0

    recs: List[ModuleRecommendation] = []
    for row in sensitivity_rows:
        path = row["module_path"]
        kind = row.get("kind", "")
        out_mse = row.get("output_mse", float("nan"))       # weight-only
        joint_mse = row.get("joint_output_mse", float("nan"))  # W8A8
        w_kurt_max = row.get("weight_kurtosis_max", float("nan"))
        heavy_ratio = row.get("heavy_channel_ratio", float("nan"))
        w_skew = row.get("weight_skewness", float("nan"))
        ch_sev = row.get("act_channel_severity", float("nan"))

        high_sens = _is_num(joint_mse) and joint_mse > median_joint
        has_heavy_channels = (_is_num(heavy_ratio)
                              and heavy_ratio > HEAVY_CHANNEL_RATIO_THRESHOLD)
        skew = _is_num(w_skew) and abs(w_skew) > SKEWNESS_THRESHOLD
        notable_act_outliers = (_is_num(ch_sev)
                                and ch_sev > ACTIVATION_OUTLIER_NOTABLE)
        is_down_or_gate = kind == "mlp" and (
            path.endswith("down_proj") or path.endswith("gate_proj"))

        # --- weight scheme ---
        if high_sens and has_heavy_channels:
            bits, gran = 8, "per-group(128)"
        elif high_sens:
            bits, gran = 8, "per-channel"
        else:
            bits, gran = 4, "per-channel"
        sym = "asymmetric" if skew else "symmetric"

        # --- reason: why hard + how to quantize ---
        why: List[str] = []
        if high_sens:
            why.append(f"joint_output_mse={joint_mse:.2e} 高于中位数({median_joint:.2e})")
        else:
            why.append(f"joint_output_mse={joint_mse:.2e} 低（低敏感）"
                       if _is_num(joint_mse) else "无 output 数据")
        # activation-side loss share (joint - weight-only) when both available
        if _is_num(joint_mse) and _is_num(out_mse) and joint_mse > out_mse:
            why.append(f"激活量化损失占比{(joint_mse - out_mse) / joint_mse:.0%}")
        if has_heavy_channels:
            why.append(f"权重重尾通道占比{heavy_ratio:.1%}(max kurtosis={w_kurt_max:.1f})")
        if skew:
            why.append(f"权重偏态(skew={w_skew:.2f})")
        if notable_act_outliers:
            # diagnostic: notable activation outliers hint that AWQ/SmoothQuant
            # (which target outlier channels) may help -- but we don't pick one.
            why.append(f"激活离群通道(severity={ch_sev:.1f})")
        if is_down_or_gate:
            why.append("MLP down/gate 投影已知更敏感")
        how = f"W{bits} {gran} {sym} | 激活 per-token"
        reason = "why: " + "; ".join(why) + " | how: " + how

        recs.append(ModuleRecommendation(
            module_path=path, kind=kind,
            recommended_bits=bits, recommended_granularity=gran,
            recommended_symmetry=sym,
            output_mse=out_mse, joint_output_mse=joint_mse,
            weight_kurtosis_max=w_kurt_max, heavy_channel_ratio=heavy_ratio,
            act_channel_severity=ch_sev, reason=reason,
        ))

    n4 = sum(1 for r in recs if r.recommended_bits == 4)
    n8 = sum(1 for r in recs if r.recommended_bits == 8)
    summary = (f"{len(recs)} modules analyzed. "
               f"{n4} suitable for 4-bit, {n8} recommend 8-bit "
               f"(W8A8, activation per-token).")
    return RecommendationReport(
        model=model_name,
        config={"bits": config.bits if config else None},
        recommendations=recs, summary=summary,
    )
