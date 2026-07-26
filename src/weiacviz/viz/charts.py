"""Plotly chart rendering for distributions and statistics."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import plotly.graph_objects as go

from ..shared.types import HistogramResult
from ..stats._util import to_numpy
from ..stats.histogram import histogram


def distribution_histogram(t, name: str = "values", num_bins: int = 256) -> go.Figure:
    """Render a distribution histogram for a tensor."""
    h = histogram(t, num_bins=num_bins)
    centers = [(a + b) / 2 for a, b in zip(h.bin_edges[:-1], h.bin_edges[1:])]
    fig = go.Figure(data=[go.Bar(x=centers, y=h.counts, name=name)])
    fig.update_layout(
        title=f"Distribution: {name}",
        xaxis_title="value", yaxis_title="count",
        bargap=0.01, template="plotly_white",
    )
    return fig


def channel_heatmap(values: Sequence[float], title: str = "per-channel stats") -> go.Figure:
    """Render a 1D heatmap of per-channel values."""
    arr = np.asarray(list(values), dtype=np.float64).reshape(1, -1)
    fig = go.Figure(data=go.Heatmap(z=arr, colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_white")
    return fig


def layer_stats_heatmap(matrix: List[List[float]], title: str = "per-layer stats") -> go.Figure:
    """Render a 2D heatmap (layers x metric)."""
    fig = go.Figure(data=go.Heatmap(z=matrix, colorscale="Viridis"))
    fig.update_layout(title=title, template="plotly_white")
    return fig


def comparison_histograms(t_before, t_after, name: str = "values",
                          num_bins: int = 256) -> go.Figure:
    """Overlay before/after-quantization distributions."""
    a = to_numpy(t_before).flatten()
    b = to_numpy(t_after).flatten()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=a, name="before", opacity=0.7, nbinsx=num_bins))
    fig.add_trace(go.Histogram(x=b, name="after", opacity=0.7, nbinsx=num_bins))
    fig.update_layout(barmode="overlay", title=f"Before vs after: {name}",
                      template="plotly_white")
    return fig


def channel_stats_heatmap(
    stats,
    metrics=("kurtosis", "skewness", "tail_ratio", "outlier_ratio"),
    title: str = "per-channel shape metrics",
) -> go.Figure:
    """Render a [metric x channel] heatmap from a StatResult list.

    Heavy-tail channels are flagged via a count in the title and surfaced
    through hovertext (shape_label), giving a distinguishable marking.
    """
    z, hover = [], []
    for m in metrics:
        zrow, hrow = [], []
        for s in stats:
            v = float(getattr(s, m, float("nan")))
            if v != v:
                v = float("nan")
            zrow.append(v)
            hrow.append(f"{m}: {v:.4g}\nlabel: {getattr(s, 'shape_label', '')}")
        z.append(zrow)
        hover.append(hrow)
    heavy = sum(1 for s in stats if "重尾" in (getattr(s, "shape_label", "") or ""))
    suffix = "s" if heavy != 1 else ""
    fig = go.Figure(data=go.Heatmap(
        z=z, hovertext=hover, colorscale="Viridis", y=list(metrics),
    ))
    fig.update_layout(
        title=f"{title} ({heavy} heavy-tail channel{suffix})",
        template="plotly_white", xaxis_title="channel", yaxis_title="metric",
    )
    return fig


def channel_violin(weight, max_channels: int = 64,
                   name: str = "per-channel distribution") -> go.Figure:
    """Render per-channel violins; top-k by |max| when channels exceed limit.

    When the output-channel count exceeds ``max_channels``, the channels with
    the largest absolute magnitude are kept -- typically the outlier channels
    most relevant to quantization difficulty (outlier_ratio fallback).
    """
    arr = to_numpy(weight)
    if arr.ndim < 2:
        arr = arr.reshape(1, -1)
    n = arr.shape[0]
    if n > max_channels:
        maxabs = np.abs(arr).max(axis=1)
        idx = np.argsort(maxabs)[::-1][:max_channels]
        arr = arr[idx]
        n = max_channels
    fig = go.Figure()
    for i in range(n):
        fig.add_trace(go.Violin(
            y=arr[i], name=f"ch{i}", points=False, box_visible=False,
            showlegend=False,
        ))
    fig.update_layout(
        title=name, template="plotly_white",
        xaxis_title="channel", yaxis_title="value",
    )
    return fig
