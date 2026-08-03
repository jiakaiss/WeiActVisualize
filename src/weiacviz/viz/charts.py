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


def render_histogram_result(h: HistogramResult, name: str = "values") -> go.Figure:
    """Render an already-aggregated ``HistogramResult`` (e.g. online activation histogram).

    Unlike ``distribution_histogram`` (which buckets a raw tensor), this takes
    a pre-aggregated histogram so it can render activations folded online
    across batches without retaining raw tensors.
    """
    centers = [(a + b) / 2 for a, b in zip(h.bin_edges[:-1], h.bin_edges[1:])]
    fig = go.Figure(data=[go.Bar(x=centers, y=h.counts, name=name)])
    fig.update_layout(
        title=f"Distribution: {name}",
        xaxis_title="value", yaxis_title="count",
        bargap=0.01, template="plotly_white",
    )
    return fig


def channel_absmean_bar(channel_stats, name: str = "per-channel abs_mean",
                        k: float = 5.0) -> go.Figure:
    """Bar chart of per-channel (hidden-dim) abs_mean; outlier channels red.

    Surfaces the SmoothQuant motivation: a few channels with abnormally large
    abs_mean that dominate the activation range. ``channel_stats`` is a
    RunningChannelStats (duck-typed: ``to_result()`` -> dict with abs_mean).
    """
    res = channel_stats.to_result() if hasattr(channel_stats, "to_result") else channel_stats
    am = np.asarray(res.get("abs_mean", []), dtype=np.float64)
    if am.size == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{name} (no data)", template="plotly_white")
        return fig
    med = float(np.median(am))
    colors = ["crimson" if (med > 0 and v > k * med) else "steelblue" for v in am]
    fig = go.Figure(data=go.Bar(y=am, marker_color=colors, name=name))
    n_out = sum(1 for c in colors if c == "crimson")
    fig.update_layout(
        title=f"{name} (median={med:.3g}, {n_out} outlier channel{'s' if n_out != 1 else ''} > {k}x)",
        xaxis_title="channel index", yaxis_title="abs_mean",
        template="plotly_white",
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


def sensitivity_heatmap(
    rows: List[dict],
    metrics: Sequence[str] = ("output_mse", "joint_output_mse",
                              "weight_kurtosis_max", "heavy_channel_ratio",
                              "act_channel_severity"),
    title: str = "per-module sensitivity (each metric normalized)",
) -> go.Figure:
    """Normalized [metric x module] heatmap from sensitivity rows.

    Each metric row is independently min-max normalized to [0, 1] for color
    (metrics have very different scales: mse ~1e-4 vs kurtosis ~tens vs
    cosine ~1); hover shows the raw value. Modules are ordered as given
    (the caller pre-sorts the rows).
    """
    x_labels = [f"#{i} {r.get('module_path', '').rsplit('.', 1)[-1]}"
                for i, r in enumerate(rows)]
    z_norm: List[List[float]] = []
    hover: List[List[str]] = []
    for m in metrics:
        raw = [float(r.get(m, float("nan"))) for r in rows]
        valid = [v for v in raw if v == v]
        lo = min(valid) if valid else 0.0
        hi = max(valid) if valid else 1.0
        rng = (hi - lo) if hi > lo else 1.0
        zrow, hrow = [], []
        for j, v in enumerate(raw):
            nv = (v - lo) / rng if v == v else float("nan")
            zrow.append(nv)
            hrow.append(f"{x_labels[j]}\n{m}: {v:.4g}" if v == v
                        else f"{x_labels[j]}\n{m}: NaN")
        z_norm.append(zrow)
        hover.append(hrow)
    fig = go.Figure(data=go.Heatmap(
        z=z_norm, hovertext=hover, colorscale="Viridis", y=list(metrics),
    ))
    fig.update_layout(title=title, template="plotly_white",
                      xaxis_title="module", yaxis_title="metric")
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
