"""Export reports and data to Markdown / CSV / JSON / PNG / Parquet."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


def export_markdown(report, path) -> str:
    """Write a Markdown report; return the written text."""
    lines = [f"# 量化建议报告 — {report.model}", "", report.summary, ""]
    lines.append("| module | bits | granularity | mse | outlier_ratio | reason |")
    lines.append("|---|---|---|---|---|---|")
    for r in report.recommendations:
        lines.append(
            f"| {r.module_path} | {r.recommended_bits} | {r.recommended_granularity} "
            f"| {r.mse:.4e} | {r.outlier_ratio:.2%} | {r.reason} |"
        )
    text = "\n".join(lines)
    Path(path).write_text(text, encoding="utf-8")
    return text


def export_json(report, path) -> None:
    data = {
        "model": report.model, "config": report.config,
        "summary": report.summary,
        "recommendations": [r.__dict__ for r in report.recommendations],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_csv(rows: List[dict], path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def export_figure_png(fig, path) -> None:
    fig.write_image(path)


def export_dataframe(df: pd.DataFrame, path) -> None:
    p = str(path)
    if p.endswith(".csv"):
        df.to_csv(p, index=False, encoding="utf-8-sig")
    elif p.endswith(".json"):
        df.to_json(p, orient="records", force_ascii=False)
    else:
        df.to_parquet(p, index=False)
