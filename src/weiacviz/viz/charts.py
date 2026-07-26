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
