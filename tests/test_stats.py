"""Tests for stats module with known inputs."""
import math

import numpy as np
import torch
import plotly.graph_objects as go

from weiacviz.stats.histogram import histogram, histogram_sliced
from weiacviz.stats.shape import (
    excess_kurtosis,
    skewness,
    robust_tail_ratio,
    shape_label,
)
from weiacviz.stats.weight_stats import weight_stats
from weiacviz.stats.activation_stats import (
    activation_stats_per_token, activation_stats_per_channel,
)
from weiacviz.stats.outliers import detect_outliers_percentile, detect_outliers_zscore
from weiacviz.stats.distance import kl_divergence, wasserstein
from weiacviz.viz.charts import channel_stats_heatmap, channel_violin
from weiacviz.shared.types import Granularity


def test_histogram_counts_sum():
    h = histogram(np.random.randn(1000), num_bins=10)
    assert h.num_bins == 10
    assert sum(h.counts) == 1000


def test_histogram_supports_torch():
    h = histogram(torch.randn(500), num_bins=16)
    assert sum(h.counts) == 500


def test_weight_stats_per_channel():
    w = torch.randn(4, 8)
    stats = weight_stats(w, "m", granularity=Granularity.PER_CHANNEL)
    assert len(stats) == 4
    for s in stats:
        assert s.min <= s.max


def test_weight_stats_per_tensor():
    w = torch.randn(4, 8)
    stats = weight_stats(w, "m", granularity=Granularity.PER_TENSOR)
    assert len(stats) == 1
    assert abs(stats[0].mean - w.float().mean().item()) < 1e-5


def test_activation_stats_per_token_and_channel():
    a = torch.randn(3, 5, 7)  # [batch, seq, hidden]
    tok = activation_stats_per_token(a, "m")
    assert len(tok) == 3 * 5
    ch = activation_stats_per_channel(a, "m")
    assert len(ch) == 7


def test_outliers_percentile():
    a = np.array([0.0] * 100 + [100.0])
    res = detect_outliers_percentile(a, percentile=99.0)
    assert res.outlier_ratio > 0
    assert res.max_abs == 100.0


def test_outliers_zscore():
    a = np.array([0.0] * 1000 + [50.0])
    res = detect_outliers_zscore(a, z_threshold=3.0)
    assert len(res.indices) > 0


def test_distance_kl_identical_is_zero():
    a = np.random.randn(2000)
    assert kl_divergence(a, a) < 1e-6


def test_distance_wasserstein_identical_is_zero():
    a = np.random.randn(2000)
    assert wasserstein(a, a) < 1e-9


def test_distance_distinguishes_shifted():
    a = np.random.randn(2000)
    b = np.random.randn(2000) + 5
    assert wasserstein(a, b) > 1.0
    assert kl_divergence(a, b) > 0


# --- distribution shape metrics ---

def test_shape_normal_kurtosis_near_zero():
    rng = np.random.RandomState(42)
    a = rng.randn(200000)
    assert abs(excess_kurtosis(a)) < 0.1


def test_shape_laplace_is_heavy_tail():
    rng = np.random.RandomState(0)
    a = rng.laplace(0, 1, 200000)
    assert excess_kurtosis(a) > 2.5  # Laplace excess kurtosis ~= 3


def test_shape_uniform_is_light_tail():
    rng = np.random.RandomState(0)
    a = rng.uniform(-1, 1, 200000)
    assert excess_kurtosis(a) < -1.0  # uniform ~= -1.2


def test_shape_zero_variance_returns_nan():
    a = np.ones(100)
    assert math.isnan(excess_kurtosis(a))
    assert math.isnan(skewness(a))
    assert math.isnan(robust_tail_ratio(a))


def test_shape_tail_ratio_value():
    rng = np.random.RandomState(42)
    a = rng.randn(200000)
    tr = robust_tail_ratio(a)
    # normal: (p99.9 - p0.1) ~= 6.58, 2*std = 2 => ~3.3
    assert 2.5 < tr < 4.0


def test_shape_label_classification():
    assert shape_label(5.0, 0.0) == "重尾"
    assert shape_label(-1.0, 0.0) == "轻尾"
    assert shape_label(0.0, 0.0) == "正态"
    assert "偏态" in shape_label(0.0, 1.0)


# --- sliced histogram ---

def test_histogram_sliced_per_channel():
    w = torch.randn(4, 8)
    hs = histogram_sliced(w, Granularity.PER_CHANNEL, num_bins=16)
    assert len(hs) == 4
    for h in hs:
        assert h.num_bins == 16
        assert sum(h.counts) == 8


def test_histogram_sliced_per_group():
    w = torch.randn(2, 16)
    hs = histogram_sliced(w, Granularity.PER_GROUP, group_size=8, num_bins=8)
    # slice_weight per_group flattens (32 elems) then groups of 8 => 4 slices
    assert len(hs) == 4
    for h in hs:
        assert h.num_bins == 8


def test_histogram_sliced_aligned_range():
    w = torch.randn(3, 10)
    hs = histogram_sliced(w, Granularity.PER_CHANNEL, num_bins=5, value_range=(-5, 5))
    for h in hs:
        assert h.bin_edges[0] == -5.0
        assert h.bin_edges[-1] == 5.0


# --- weight_stats integration with shape + histogram ---

def test_weight_stats_has_shape_and_histogram():
    w = torch.randn(4, 8)
    stats = weight_stats(w, "m", granularity=Granularity.PER_CHANNEL)
    for s in stats:
        assert s.histogram is not None
        assert s.histogram.num_bins == 256
        assert s.kurtosis == s.kurtosis  # not NaN for normal data
        assert s.tail_ratio == s.tail_ratio
        assert s.outlier_ratio >= 0.0
        assert s.shape_label  # non-empty label


# --- visualization ---

def test_channel_stats_heatmap_returns_figure():
    w = torch.randn(4, 8)
    stats = weight_stats(w, "m", granularity=Granularity.PER_CHANNEL)
    fig = channel_stats_heatmap(stats)
    assert isinstance(fig, go.Figure)


def test_channel_violin_topk_sampling():
    w = torch.randn(100, 8)
    fig = channel_violin(w, max_channels=16)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 16  # capped to top-k


def test_channel_violin_no_sampling_below_limit():
    w = torch.randn(8, 8)
    fig = channel_violin(w, max_channels=64)
    assert len(fig.data) == 8


def test_view_weight_returns_three_figures():
    import torch.nn as nn
    from weiacviz.viz.app import App
    app = App()
    app._model = nn.Sequential(nn.Linear(8, 16))
    fig, hm, vio = app.view_weight("0", 64, "per-channel", None)
    assert isinstance(fig, go.Figure)
    assert isinstance(hm, go.Figure)
    assert isinstance(vio, go.Figure)


def test_view_weight_per_group():
    import torch.nn as nn
    from weiacviz.viz.app import App
    app = App()
    app._model = nn.Sequential(nn.Linear(8, 16))
    fig, hm, vio = app.view_weight("0", 64, "per-group", 4)
    assert isinstance(fig, go.Figure)
    assert isinstance(hm, go.Figure)
    assert isinstance(vio, go.Figure)


def test_view_weight_per_group_invalid_size():
    import pytest
    import torch.nn as nn
    from weiacviz.viz.app import App
    app = App()
    app._model = nn.Sequential(nn.Linear(8, 16))
    with pytest.raises(Exception):
        app.view_weight("0", 64, "per-group", 0)


def test_channel_violin_per_group():
    w = torch.randn(4, 16)
    fig = channel_violin(w, granularity=Granularity.PER_GROUP, group_size=4)
    assert isinstance(fig, go.Figure)
