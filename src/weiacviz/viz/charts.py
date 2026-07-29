"""Plotly chart rendering for distributions and statistics."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import plotly.graph_objects as go

from ..loading.weights import slice_weight
from ..shared.types import Granularity, HistogramResult
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
    metrics=("kurtosis", "skewness", "tail_ratio"),
    title: str = "per-channel shape metrics",
) -> go.Figure:
    """Render a [metric x slice] heatmap from a StatResult list.

    Each metric row is **independently min-max normalized** to [0, 1] for
    color, so metrics with very different scales (kurtosis ~tens vs tail_ratio
    ~single digits) each show their own gradient. Hover still shows the raw
    value. Default metrics exclude outlier_ratio (near-constant ~0.1% across
    slices, no signal in a heatmap). Each cell's hover carries the slice index
    so it can be cross-referenced with the violin's `#<index>` labels.
    """
    z_norm, hover = [], []
    for m in metrics:
        raw = [float(getattr(s, m, float("nan"))) for s in stats]
        valid = [v for v in raw if v == v]
        lo = min(valid) if valid else 0.0
        hi = max(valid) if valid else 1.0
        rng = (hi - lo) if hi > lo else 1.0
        zrow, hrow = [], []
        for j, v in enumerate(raw):
            nv = (v - lo) / rng if v == v else float("nan")
            zrow.append(nv)
            hrow.append(f"slice #{j}\n{m}: {v:.4g}\nlabel: {getattr(stats[j], 'shape_label', '')}")
        z_norm.append(zrow)
        hover.append(hrow)
    heavy = sum(1 for s in stats if "重尾" in (getattr(s, "shape_label", "") or ""))
    suffix = "s" if heavy != 1 else ""
    fig = go.Figure(data=go.Heatmap(
        z=z_norm, hovertext=hover, colorscale="Viridis", y=list(metrics),
    ))
    fig.update_layout(
        title=f"{title} ({heavy} heavy-tail slice{suffix})",
        template="plotly_white", xaxis_title="slice index", yaxis_title="metric",
    )
    return fig


def channel_violin(weight, granularity: Granularity = Granularity.PER_CHANNEL,
                   group_size=None, stats=None, max_channels: int = 64,
                   name: str = "per-channel distribution") -> go.Figure:
    """Render per-slice violins; top-k by excess kurtosis (heavy-tail severity).

    Slices come from ``slice_weight`` (per-channel axis 0 / per-group flattened
    groups share one path). When the slice count exceeds ``max_channels``, the
    slices with the highest excess kurtosis are kept -- kurtosis directly
    measures heavy-tailedness, so this surfaces the most outlier-prone slices.
    Each violin is labeled with its **original slice index** and kurtosis
    (e.g. `#42 k=12.3`) so it can be cross-referenced with the shape heatmap.
    """
    slices = slice_weight(weight, granularity, group_size)
    arrs = [to_numpy(s).flatten() for s in slices]
    n = len(arrs)
    if stats is not None and len(stats) == n:
        kurts = [float(getattr(s, "kurtosis", float("nan"))) for s in stats]
    else:
        kurts = [float("nan")] * n
    if n > max_channels:
        order = sorted(
            range(n),
            key=lambda i: kurts[i] if kurts[i] == kurts[i] else float("-inf"),
            reverse=True,
        )
        sel = order[:max_channels]
    else:
        sel = list(range(n))
    fig = go.Figure()
    for i in sel:
        a = arrs[i]
        k = kurts[i]
        label = f"#{i} k={k:.2g}" if k == k else f"#{i}"
        hover = f"slice #{i}\nkurtosis={k:.4g}" if k == k else f"slice #{i}"
        fig.add_trace(go.Violin(
            y=a, name=label, points=False, box_visible=False,
            showlegend=False, hovertext=hover,
        ))
    fig.update_layout(
        title=name, template="plotly_white",
        xaxis_title="slice (top-k by kurtosis)", yaxis_title="value",
    )
    return fig
