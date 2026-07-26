"""Tests for stats module with known inputs."""
import numpy as np
import torch

from weiacviz.stats.histogram import histogram
from weiacviz.stats.weight_stats import weight_stats
from weiacviz.stats.activation_stats import (
    activation_stats_per_token, activation_stats_per_channel,
)
from weiacviz.stats.outliers import detect_outliers_percentile, detect_outliers_zscore
from weiacviz.stats.distance import kl_divergence, wasserstein
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
