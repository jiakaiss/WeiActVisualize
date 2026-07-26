"""Tests for quantization simulation."""
import torch

from weiacviz.quant.fake_quant import fake_quantize_tensor
from weiacviz.quant.error_metrics import mse, cosine_similarity
from weiacviz.quant.scheme_compare import compare_schemes
from weiacviz.quant.sensitivity import layer_sensitivity
from weiacviz.shared.types import Granularity, QuantConfig, Symmetry


def test_symmetric_per_tensor_shape_preserved():
    t = torch.randn(4, 8)
    cfg = QuantConfig(bits=4, granularity=Granularity.PER_TENSOR, symmetry=Symmetry.SYMMETRIC)
    q = fake_quantize_tensor(t, cfg)
    assert q.shape == t.shape
    assert mse(t, q) >= 0


def test_asymmetric_per_tensor_preserves_range():
    t = torch.rand(4, 8)  # non-negative
    cfg = QuantConfig(bits=8, granularity=Granularity.PER_TENSOR, symmetry=Symmetry.ASYMMETRIC)
    q = fake_quantize_tensor(t, cfg)
    step = (t.max() - t.min()) / 255  # quantization step size
    assert q.min() >= t.min() - step
    assert q.max() <= t.max() + step


def test_per_channel_and_group_shape():
    t = torch.randn(4, 16)
    cfg_c = QuantConfig(bits=4, granularity=Granularity.PER_CHANNEL, symmetry=Symmetry.SYMMETRIC)
    assert fake_quantize_tensor(t, cfg_c).shape == t.shape
    cfg_g = QuantConfig(bits=4, granularity=Granularity.PER_GROUP, symmetry=Symmetry.SYMMETRIC, group_size=8)
    assert fake_quantize_tensor(t, cfg_g).shape == t.shape


def test_higher_bits_lower_error():
    t = torch.randn(8, 32)
    q4 = fake_quantize_tensor(t, QuantConfig(bits=4, granularity=Granularity.PER_TENSOR, symmetry=Symmetry.SYMMETRIC))
    q8 = fake_quantize_tensor(t, QuantConfig(bits=8, granularity=Granularity.PER_TENSOR, symmetry=Symmetry.SYMMETRIC))
    assert mse(t, q8) < mse(t, q4)


def test_cosine_identical_is_one():
    t = torch.randn(100)
    assert abs(cosine_similarity(t, t) - 1.0) < 1e-5


def test_compare_schemes_returns_rows():
    t = torch.randn(4, 16)
    rows = compare_schemes(t, bits_options=(4, 8),
                           granularities=(Granularity.PER_TENSOR, Granularity.PER_CHANNEL),
                           symmetries=(Symmetry.SYMMETRIC,))
    assert len(rows) == 4
    assert all("mse" in r and "cosine" in r for r in rows)


def test_layer_sensitivity_sorted_and_topk_flagged():
    weights = {
        "a": torch.randn(4, 16),
        "b": torch.randn(4, 16) * 10,  # larger range -> larger MSE
    }
    cfg = QuantConfig(bits=4, granularity=Granularity.PER_TENSOR, symmetry=Symmetry.SYMMETRIC)
    rows = layer_sensitivity(weights, cfg, topk=1)
    assert rows[0]["mse"] >= rows[1]["mse"]
    assert rows[0].get("top") is True
