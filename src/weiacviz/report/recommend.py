"""Generate quantization suitability recommendations from stats & sensitivity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..shared.types import QuantConfig


@dataclass
class ModuleRecommendation:
    module_path: str
    recommended_bits: int
    recommended_granularity: str
    reason: str
    mse: float = 0.0
    outlier_ratio: float = 0.0


@dataclass
class RecommendationReport:
    model: str
    config: dict
    recommendations: List[ModuleRecommendation] = field(default_factory=list)
    summary: str = ""


def recommend(
    sensitivity_rows: List[dict],
    outlier_ratios: dict,
    model_name: str = "",
    config: Optional[QuantConfig] = None,
) -> RecommendationReport:
    """Produce per-module recommendations.

    Heuristics:
      - MSE above median OR high outlier ratio (>1%) -> keep 8-bit / per-channel
      - Otherwise -> suitable for 4-bit / per-tensor
    """
    mses = [r["mse"] for r in sensitivity_rows if r.get("mse") is not None]
    median_mse = sorted(mses)[len(mses) // 2] if mses else 0.0

    recs: List[ModuleRecommendation] = []
    for row in sensitivity_rows:
        path = row["module_path"]
        mse = row.get("mse", 0.0)
        out = outlier_ratios.get(path, 0.0)
        if mse > median_mse or out > 0.01:
            bits, gran = 8, "per-channel"
            reason = f"high sensitivity (mse={mse:.4e}) or outliers ({out:.2%})"
        else:
            bits, gran = 4, "per-tensor"
            reason = f"low sensitivity (mse={mse:.4e}), outliers {out:.2%}"
        recs.append(ModuleRecommendation(
            module_path=path, recommended_bits=bits,
            recommended_granularity=gran, reason=reason,
            mse=mse, outlier_ratio=out,
        ))

    n4 = sum(1 for r in recs if r.recommended_bits == 4)
    n8 = sum(1 for r in recs if r.recommended_bits == 8)
    summary = (f"{len(recs)} modules analyzed. "
               f"{n4} suitable for 4-bit, {n8} recommend 8-bit.")
    return RecommendationReport(
        model=model_name,
        config={"bits": config.bits if config else None},
        recommendations=recs, summary=summary,
    )
