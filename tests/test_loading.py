"""Tests for loading module using a tiny synthetic model (no HF download)."""
import pytest
import torch
import torch.nn as nn

from weiacviz.loading.adapter import HFCausalLMAdapter
from weiacviz.loading.module_resolver import resolve_modules, detect_arch_family
from weiacviz.loading.weights import get_weight, slice_weight
from weiacviz.loading.hook import ActivationCapture
from weiacviz.loading.runner import RunningStats, OnlineAggregator, run_calibration
from weiacviz.shared.types import Granularity, ModuleKind


class TinyAttn(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.o_proj(self.v_proj(x) + self.q_proj(x) + self.k_proj(x))


class TinyMLP(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2)
        self.up_proj = nn.Linear(d, d * 2)
        self.down_proj = nn.Linear(d * 2, d)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * torch.sigmoid(self.gate_proj(x)))


class TinyBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.self_attn = TinyAttn(d)
        self.mlp = TinyMLP(d)

    def forward(self, x):
        return x + self.mlp(self.self_attn(x))


class TinyLlamaLike(nn.Module):
    """Mimics Llama-like module naming for resolver tests."""

    def __init__(self, d=8, n_layers=2):
        super().__init__()
        self.config = type("Cfg", (), {"architectures": ["LlamaForCausalLM"]})()
        self.layers = nn.ModuleList([TinyBlock(d) for _ in range(n_layers)])
        self.embed = nn.Linear(d, d)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_resolve_llama_like_modules():
    model = TinyLlamaLike()
    result = resolve_modules(model)
    assert result.family == "llama"
    assert not result.degraded
    paths = [m.path for m in result.modules]
    assert any("q_proj" in p for p in paths)
    assert any("gate_proj" in p for p in paths)
    assert all(m.kind in (ModuleKind.ATTENTION, ModuleKind.MLP) for m in result.modules)


def test_unknown_arch_degrades_to_all_linears():
    model = TinyLlamaLike()
    model.config.architectures = ["SomeUnknownArch"]
    result = resolve_modules(model)
    assert result.degraded
    assert result.family == "unknown"
    assert all(m.kind == ModuleKind.OTHER for m in result.modules)
    assert len(result.modules) > 0


def test_weight_access_and_slice():
    model = TinyLlamaLike()
    w = get_weight(model, "layers.0.self_attn.q_proj")
    assert w.shape[0] == 8
    chans = slice_weight(w, Granularity.PER_CHANNEL)
    assert len(chans) == 8
    groups = slice_weight(w, Granularity.PER_GROUP, group_size=4)
    assert len(groups) == 16  # 8*8 / 4
    tensor = slice_weight(w, Granularity.PER_TENSOR)
    assert len(tensor) == 1 and tensor[0].numel() == 64


def test_activation_capture():
    model = TinyLlamaLike()
    paths = [m.path for m in resolve_modules(model).modules]
    cap = ActivationCapture(paths)
    cap.attach(model)
    model(torch.randn(2, 4, 8))
    assert len(cap.buffer) == len(paths)
    sample = next(iter(cap.buffer.values()))
    assert "output" in sample
    cap.detach()
    # after detach, running again should not populate buffer
    model(torch.randn(1, 4, 8))
    assert len(cap.buffer) == len(paths)  # unchanged from prior


def test_running_stats_online_aggregation():
    rs = RunningStats()
    rs.update(torch.tensor([1.0, 2.0, 3.0]))
    rs.update(torch.tensor([4.0, 5.0]))
    assert rs.count == 5
    assert rs.min == 1.0 and rs.max == 5.0
    assert abs(rs.mean - 3.0) < 1e-5


def test_online_aggregator_memory_independent_of_batches():
    model = TinyLlamaLike()
    paths = [m.path for m in resolve_modules(model).modules]

    class DummyTok:
        def __call__(self, texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048):
            n = len(texts)
            return {"input_ids": torch.randn(n, max_length)}

    texts = ["a b c"] * 20
    agg = run_calibration(HFCausalLMAdapter(model, DummyTok(), texts, seq_length=8), paths,
                          config=None)
    # every target module should have non-zero activation counts
    for p in paths:
        assert agg.stats[p]["output"].count > 0


def test_running_histogram_fixed_range_accumulates():
    from weiacviz.loading.runner import RunningHistogram
    h = RunningHistogram((0.0, 1.0), num_bins=4)
    h.update(torch.tensor([0.1, 0.2, 0.3]))
    h.update(torch.tensor([0.6, 0.9]))
    res = h.to_result()
    assert res.num_bins == 4
    assert len(res.counts) == 4
    assert len(res.bin_edges) == 5
    assert sum(res.counts) == 5
    # edges [0, 0.25, 0.5, 0.75, 1.0]: 0.1,0.2->bin0; 0.3->bin1; 0.6->bin2; 0.9->bin3
    assert res.counts == [2, 1, 1, 1]


def test_two_pass_calibration_collects_histogram():
    model = TinyLlamaLike()
    paths = [m.path for m in resolve_modules(model).modules]

    class DummyTok:
        def __call__(self, texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048):
            n = len(texts)
            return {"input_ids": torch.ones(n, max_length)}

    texts = ["a b c"] * 20
    agg = run_calibration(HFCausalLMAdapter(model, DummyTok(), texts, seq_length=8), paths,
                          config=None,
                          collect_histogram=True, num_bins=32)
    for p in paths:
        s = agg.stats[p]["output"]
        assert s.count > 0
        h = agg.histograms[p]["output"]
        assert h is not None
        res = h.to_result()
        assert res.num_bins == 32
        # deterministic input -> Pass 2 stays within Pass 1 range -> no drops
        assert sum(res.counts) == s.count


def test_calibration_without_histogram_leaves_histograms_none():
    model = TinyLlamaLike()
    paths = [m.path for m in resolve_modules(model).modules]

    class DummyTok:
        def __call__(self, texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048):
            n = len(texts)
            return {"input_ids": torch.ones(n, max_length)}

    texts = ["a b c"] * 8
    agg = run_calibration(HFCausalLMAdapter(model, DummyTok(), texts, seq_length=8), paths,
                          config=None, collect_histogram=False)
    for p in paths:
        assert agg.histograms[p]["output"] is None
        assert agg.stats[p]["output"].count > 0


def test_running_histogram_degenerate_range():
    from weiacviz.loading.runner import RunningHistogram
    # min == max: must not raise, expands to non-zero width
    h = RunningHistogram((2.5, 2.5), num_bins=4)
    h.update(torch.tensor([2.5, 2.5, 2.5]))
    res = h.to_result()
    assert res.num_bins == 4
    assert sum(res.counts) == 3


def test_running_token_stats_online_matches_one_shot():
    """Per-token abs_max moments aggregated online match a one-shot compute."""
    from weiacviz.loading.runner import RunningTokenStats
    from weiacviz.stats.shape import excess_kurtosis, skewness
    torch.manual_seed(1)
    acts = [torch.randn(3, 4, 8) for _ in range(3)]
    ts = RunningTokenStats(num_bins=16)
    for a in acts:
        ts.update_moments(a)
    all_absmax = torch.cat(
        [a.abs().amax(dim=-1).reshape(-1) for a in acts]).numpy()
    assert ts.count == len(all_absmax)
    assert abs(ts.mean - all_absmax.mean()) < 1e-6
    assert abs(ts.std - all_absmax.std()) < 1e-6
    assert abs(ts.cv - all_absmax.std() / all_absmax.mean()) < 1e-6
    assert abs(ts.kurtosis - excess_kurtosis(all_absmax)) < 1e-3
    assert abs(ts.skewness - skewness(all_absmax)) < 1e-3


def test_two_pass_calibration_collects_token_histogram():
    """collect_histogram + collect_token_stats builds a per-token abs_max
    histogram whose counts equal the token count (no out-of-range drops)."""
    model = TinyLlamaLike()
    paths = [m.path for m in resolve_modules(model).modules]

    class DummyTok:
        def __call__(self, texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048):
            n = len(texts)
            return {"input_ids": torch.arange(n * max_length).reshape(n, max_length).float()}

    texts = ["a b c"] * 12
    agg = run_calibration(HFCausalLMAdapter(model, DummyTok(), texts, seq_length=8), paths, config=None,
                          collect_histogram=True, num_bins=32,
                          collect_token_stats=True)
    for p in paths:
        ts = agg.token_stats[p]["output"]
        assert ts is not None and ts.count > 0
        assert ts.abs_max_hist is not None  # built between passes
        assert ts.kurtosis == ts.kurtosis  # not NaN (varying input)
        assert sum(ts.abs_max_hist.counts.tolist()) == ts.count


def test_single_pass_calibration_token_stats_without_histogram():
    """Without collect_histogram, per-token moments are still computed
    (single pass) but no per-token histogram is built."""
    model = TinyLlamaLike()
    paths = [m.path for m in resolve_modules(model).modules]

    class DummyTok:
        def __call__(self, texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048):
            n = len(texts)
            return {"input_ids": torch.arange(n * max_length).reshape(n, max_length).float()}

    texts = ["a b c"] * 8
    agg = run_calibration(HFCausalLMAdapter(model, DummyTok(), texts, seq_length=8), paths, config=None,
                          collect_histogram=False,
                          collect_token_stats=True)
    for p in paths:
        ts = agg.token_stats[p]["output"]
        assert ts is not None and ts.count > 0
        assert ts.abs_max_hist is None  # no two-pass -> no histogram
        assert ts.cv == ts.cv  # moments available


def test_detect_available_backend_returns_valid():
    from weiacviz.loading.model_loader import detect_available_backend
    assert detect_available_backend() in {"cpu", "cuda", "npu"}


def test_move_to_backend_npu_missing_raises_runtime_error():
    # Without torch_npu installed, requesting npu must raise a clear RuntimeError
    # (not an ImportError crash).
    from weiacviz.loading.model_loader import _move_to_backend
    try:
        import torch_npu  # noqa: F401
        has_npu = True
    except Exception:
        has_npu = False
    if has_npu:
        pytest.skip("torch_npu installed; cannot test missing-backend path")
    with pytest.raises(RuntimeError):
        _move_to_backend(torch.zeros(1), "npu")


def test_load_model_cpu_branch(monkeypatch):
    from weiacviz.loading import model_loader as ml

    calls = {}

    class FakeModel:
        def eval(self):
            calls["eval"] = True

        def to(self, d):
            calls["to"] = d
            return self

    def fake_from_pretrained(path, **kw):
        calls["from_pretrained"] = kw
        return FakeModel()

    monkeypatch.setattr(ml, "_resolve_model_path", lambda p: p)
    monkeypatch.setattr(ml.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(ml.AutoTokenizer, "from_pretrained", lambda p, **kw: "tok")

    model, tok = ml.load_model("dummy", device="cpu")
    assert tok == "tok"
    assert calls["from_pretrained"]["device_map"] == "cpu"
    assert calls.get("eval") is True
