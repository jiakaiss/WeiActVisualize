"""Tests for stats module with known inputs."""
import math

import numpy as np
import pytest
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


def test_activation_token_summary():
    """activation_token_summary builds a StatResult from per-token moments."""
    from weiacviz.loading.runner import RunningTokenStats
    from weiacviz.stats.activation_stats import activation_token_summary
    ts = RunningTokenStats()
    for _ in range(3):
        ts.update_moments(torch.randn(2, 5, 8))
    s = activation_token_summary(ts, "m", "output")
    assert s.role.value == "activation"
    assert s.kurtosis == s.kurtosis  # not NaN
    assert s.shape_label  # non-empty


def test_activation_stats_per_token_has_shape_metrics():
    """Per-token stats carry kurtosis/skewness/tail_ratio/shape_label; a
    heavy-tailed token (laplace) is flagged vs a normal one."""
    rng = np.random.RandomState(0)
    a = np.stack([rng.randn(4000), rng.laplace(0, 1, 4000)])  # [2, hidden]
    tok = activation_stats_per_token(torch.tensor(a), "m")
    assert len(tok) == 2
    assert tok[1].kurtosis > tok[0].kurtosis  # laplace heavier than normal
    assert tok[1].shape_label == "重尾"
    assert tok[0].shape_label  # non-empty
    assert tok[0].tail_ratio == tok[0].tail_ratio  # not NaN


def test_activation_stats_per_token_zero_variance_robust():
    """Constant tokens (zero variance) yield NaN shape metrics, no exception."""
    a = torch.ones(2, 5)
    tok = activation_stats_per_token(a, "m")
    assert len(tok) == 2
    for s in tok:
        assert math.isnan(s.kurtosis)
        assert math.isnan(s.tail_ratio)
        assert s.shape_label  # '未知' label, non-empty


def test_activation_token_outliers_severity_and_ratio():
    """activation_token_outliers reports max_abs, severity (max/median) and
    outlier_ratio from the per-token abs_max histogram."""
    from weiacviz.loading.runner import RunningTokenStats
    from weiacviz.stats.activation_stats import activation_token_outliers
    ts = RunningTokenStats(num_bins=32)
    base = torch.ones(20, 8) * 0.5    # 20 tokens, abs_max = 0.5
    spike = torch.ones(2, 8) * 5.0    # 2 tokens, abs_max = 5.0
    ts.update_moments(base)
    ts.update_moments(spike)
    ts.build_histogram()
    ts.update_histogram(base)
    ts.update_histogram(spike)
    info = activation_token_outliers(ts, method="percentile", percentile=99)
    assert info["max_abs"] == 5.0
    assert info["severity"] > 1.0      # 5.0 / 0.5 = 10
    assert 0.0 < info["outlier_ratio"] <= 0.2


def test_activation_token_outliers_without_histogram_degrades():
    """Without the two-pass histogram, max_abs + zscore threshold still come
    from online moments; ratio/severity are NaN."""
    from weiacviz.loading.runner import RunningTokenStats
    from weiacviz.stats.activation_stats import activation_token_outliers
    ts = RunningTokenStats(num_bins=32)
    ts.update_moments(torch.randn(10, 8))
    assert ts.abs_max_hist is None
    info = activation_token_outliers(ts, method="zscore", k=3.0)
    assert info["max_abs"] == info["max_abs"]          # available
    assert info["threshold"] == info["threshold"]      # mean + k*std
    assert math.isnan(info["outlier_ratio"])           # needs histogram
    assert math.isnan(info["severity"])                # needs median


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


def test_channel_stats_heatmap_per_row_normalize():
    w = torch.randn(8, 32)
    stats = weight_stats(w, "m", granularity=Granularity.PER_CHANNEL)
    fig = channel_stats_heatmap(stats)
    z = fig.data[0].z
    # default metrics exclude outlier_ratio
    assert list(fig.data[0].y) == ["kurtosis", "skewness", "tail_ratio"]
    # each row independently normalized to [0, 1]
    for row in z:
        valid = [v for v in row if v == v]
        assert valid
        assert min(valid) >= 0.0
        assert max(valid) <= 1.0


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


def test_channel_violin_kurtosis_topk_and_label():
    w = torch.randn(100, 16)
    stats = weight_stats(w, "m", granularity=Granularity.PER_CHANNEL)
    fig = channel_violin(w, granularity=Granularity.PER_CHANNEL,
                         stats=stats, max_channels=16)
    assert len(fig.data) == 16
    # kept violins are labeled with original slice index + kurtosis
    assert all(t.name.startswith("#") for t in fig.data)
    assert all("k=" in t.name for t in fig.data)


def test_channel_violin_fallback_label_without_stats():
    w = torch.randn(100, 16)
    fig = channel_violin(w, granularity=Granularity.PER_CHANNEL, max_channels=16)
    assert len(fig.data) == 16
    # no stats -> "#<index>" labels without kurtosis
    assert all(t.name.startswith("#") for t in fig.data)
    assert all("k=" not in t.name for t in fig.data)


# --- per-token activation slice + abs_max violin ---

def test_channel_violin_and_heatmap_accept_per_token_stats():
    """The weight-side violin + heatmap render per-token StatResults unchanged
    (each token is a slice, axis 0)."""
    rng = np.random.RandomState(0)
    a = torch.tensor(np.stack([rng.randn(200) for _ in range(10)]))  # [10, 200]
    tok = activation_stats_per_token(a, "m")
    vio = channel_violin(a, granularity=Granularity.PER_CHANNEL,
                         stats=tok, max_channels=64)
    hm = channel_stats_heatmap(tok, title="per-token")
    assert isinstance(vio, go.Figure)
    assert len(vio.data) == 10  # one violin per token
    assert isinstance(hm, go.Figure)


def test_per_token_slice_flattens_3d_activation():
    """A 3D [batch, seq, hidden] activation must flatten to [N, hidden] so each
    token is a violin slice -- not 1 slice sliced by the batch axis."""
    rng = np.random.RandomState(0)
    a = torch.tensor(rng.randn(2, 5, 16))  # [batch=2, seq=5, hidden=16]
    tok = activation_stats_per_token(a, "m")
    assert len(tok) == 2 * 5  # 10 tokens
    flat = a.reshape(-1, a.shape[-1])  # [10, 16] -- what view_activation passes
    fig = channel_violin(flat, granularity=Granularity.PER_CHANNEL,
                         stats=tok, max_channels=64)
    assert len(fig.data) == 10  # one violin per token, not 1


def test_render_token_absmax_violin_without_histogram():
    """No abs_max_hist (single-pass) -> placeholder figure, no exception."""
    from weiacviz.loading.runner import RunningTokenStats
    from weiacviz.viz.charts import render_token_absmax_violin
    ts = RunningTokenStats(num_bins=16)
    ts.update_moments(torch.randn(4, 8))
    assert ts.abs_max_hist is None
    fig = render_token_absmax_violin(ts)
    assert isinstance(fig, go.Figure)
    assert "未采集" in fig.layout.title.text


def test_render_token_absmax_violin_with_histogram_and_outliers():
    """With the two-pass histogram, the violin renders and outlier lines are
    added when outlier_info is provided."""
    from weiacviz.loading.runner import RunningTokenStats
    from weiacviz.viz.charts import render_token_absmax_violin
    from weiacviz.stats.activation_stats import activation_token_outliers
    ts = RunningTokenStats(num_bins=16)
    a = torch.randn(40, 8)
    ts.update_moments(a)
    ts.build_histogram()
    ts.update_histogram(a)
    info = activation_token_outliers(ts)
    fig = render_token_absmax_violin(ts, outlier_info=info)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # violin trace present


def test_channel_shape_stats_matches_per_slice_scalars():
    """Vectorized per-channel kurtosis/skewness == the scalar functions run
    per row (incl. zero-variance rows -> NaN)."""
    from weiacviz.stats.shape import channel_shape_stats

    rng = np.random.default_rng(0)
    w = rng.normal(size=(7, 50))
    w[2] = 0.0                      # zero-variance row -> NaN both ways
    w[4, :25] *= 40.0               # heavy-tail-ish row
    kurt, skew = channel_shape_stats(w)
    assert kurt.shape == (7,) and skew.shape == (7,)
    for i in range(7):
        assert kurt[i] == pytest.approx(excess_kurtosis(w[i]), nan_ok=True)
        assert skew[i] == pytest.approx(skewness(w[i]), nan_ok=True)

    # 1D input behaves like a single channel
    k1, s1 = channel_shape_stats(w[0])
    assert k1[0] == pytest.approx(excess_kurtosis(w[0]))
    assert s1[0] == pytest.approx(skewness(w[0]))

    # torch tensors accepted (fp16 like real weights); reference computed on
    # the fp16-rounded values themselves (fp16 storage loses precision)
    wt = torch.tensor(w, dtype=torch.float16)
    wnp = wt.float().numpy().astype(np.float64)
    kt, st = channel_shape_stats(wt)
    for i in range(7):
        assert kt[i] == pytest.approx(excess_kurtosis(wnp[i]), nan_ok=True)
        assert st[i] == pytest.approx(skewness(wnp[i]), nan_ok=True)


def test_sensitivity_weight_shape_fields_match_weight_stats():
    """The vectorized fast path in _sensitivity_rows produces the same
    weight-shape fields as the old per-slice weight_stats route."""
    import torch.nn as nn

    from weiacviz.loading.adapter import HFCausalLMAdapter
    from weiacviz.loading.module_resolver import resolve_modules
    from weiacviz.loading.weights import get_weight
    from weiacviz.stats.weight_stats import weight_stats
    from weiacviz.viz.app import App

    class MiniLlama(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"architectures": ["LlamaForCausalLM"]})()
            blk = nn.Module()
            blk.self_attn = nn.Module()
            blk.self_attn.q_proj = nn.Linear(8, 8)
            blk.mlp = nn.Module()
            blk.mlp.down_proj = nn.Linear(8, 8)
            self.layers = nn.ModuleList([blk])

        def forward(self, x):
            return x

    model = MiniLlama()
    app = App()
    app._adapter = HFCausalLMAdapter(model, lambda t, **k: {"input_ids": torch.randn(len(t), 8)})
    app._model = model
    app._resolve_result = resolve_modules(model)
    app._modules = app._resolve_result.modules
    app._adapter.set_texts(["a b c"] * 4)

    rows = app._sensitivity_rows(bits=4, seq_length=8, n_samples=2)
    for r in rows:
        ws = weight_stats(get_weight(model, r["module_path"]),
                          r["module_path"], granularity=Granularity.PER_CHANNEL)
        kurts = [s.kurtosis for s in ws if s.kurtosis == s.kurtosis]
        skews = [s.skewness for s in ws if s.skewness == s.skewness]
        assert r["weight_kurtosis_max"] == pytest.approx(max(kurts))
        assert r["heavy_channel_ratio"] == pytest.approx(
            sum(k > 3.0 for k in kurts) / len(kurts))
        assert r["weight_skewness"] == pytest.approx(float(np.median(skews)))
