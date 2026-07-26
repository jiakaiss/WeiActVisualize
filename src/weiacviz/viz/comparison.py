"""Comparison views for before/after quantization and multi-scheme errors."""
from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go

from ..quant.scheme_compare import compare_schemes


def scheme_comparison_chart(weight, **kwargs):
    """Bar chart of MSE across quantization schemes; returns (fig, rows)."""
    rows = compare_schemes(weight, **kwargs)
    valid = [r for r in rows if r.get("mse") is not None]
    labels = [f"{r['bits']}b|{r['granularity']}|{r['symmetry']}" for r in valid]
    mses = [r["mse"] for r in valid]
    cosines = [round(r["cosine"], 4) for r in valid]
    fig = go.Figure(data=[go.Bar(x=labels, y=mses, text=cosines)])
    fig.update_layout(title="Quantization scheme error comparison",
                      xaxis_title="scheme", yaxis_title="MSE",
                      template="plotly_white")
    return fig, rows
