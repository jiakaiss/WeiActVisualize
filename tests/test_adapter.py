"""ModelAdapter tests.

HFCausalLMAdapter (causal LM, backward-compatible with the pre-adapter pipeline)
and DiTAdapter (non-causal, no tokenizer, forward(x,t,y)) both drive the Linear
quantization main line end-to-end on tiny models -- no HF download.
"""
import torch
import torch.nn as nn

from weiacviz.loading.adapter import DiTAdapter, HFCausalLMAdapter
from weiacviz.loading.adapters.dit_demo import build_demo_dit_adapter
from weiacviz.loading.runner import run_calibration
from weiacviz.loading.weights import get_weight
from weiacviz.quant.error_metrics import output_diff
from weiacviz.quant.fake_quant import fake_quantize_tensor
from weiacviz.shared.types import Granularity, QuantConfig, Symmetry
from weiacviz.stats.weight_stats import weight_stats


class LlamaLike(nn.Module):
    """Tiny Llama-named model: a couple of Linear layers, forward(x)."""

    def __init__(self, d=8, n=2):
        super().__init__()
        self.config = type("C", (), {"architectures": ["LlamaForCausalLM"]})()
        self.layers = nn.ModuleList()
        for _ in range(n):
            blk = nn.Module()
            blk.self_attn = nn.Module()
            blk.self_attn.q_proj = nn.Linear(d, d)
            blk.self_attn.o_proj = nn.Linear(d, d)
            blk.mlp = nn.Module()
            blk.mlp.down_proj = nn.Linear(d, d)
            self.layers.append(blk)
        self.embed = nn.Linear(d, d)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.layers:
            x = blk.self_attn.o_proj(blk.self_attn.q_proj(x)) + blk.mlp.down_proj(x)
        return x


class DummyTok:
    def __call__(self, texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048):
        return {"input_ids": torch.randn(len(texts), max_length)}


# --- HFCausalLMAdapter (default, reproduces pre-adapter behavior) ---

def test_hf_adapter_calib_batches_and_sample_inputs():
    model = LlamaLike()
    adapter = HFCausalLMAdapter(model, DummyTok(), texts=["a b c"] * 4, seq_length=8)
    paths = [m.path for m in adapter.enumerate_modules().modules]
    assert paths  # Llama-named -> non-degraded; q_proj/o_proj/down_proj found

    batches = list(adapter.calib_batches(n_samples=4, batch_size=2))
    assert len(batches) == 2
    assert batches[0].shape == (2, 8)

    # sample_inputs works right after construction (uses sample_text, not texts)
    si = adapter.sample_inputs(paths)
    assert set(si.keys()) <= set(paths)
    assert all(t.dim() >= 2 for t in si.values())


def test_hf_adapter_run_calibration_end_to_end():
    model = LlamaLike()
    adapter = HFCausalLMAdapter(model, DummyTok(), texts=["a b c"] * 8, seq_length=8)
    paths = [m.path for m in adapter.enumerate_modules().modules]
    agg = run_calibration(adapter, paths, collect_histogram=True, num_bins=16)
    for p in paths:
        assert agg.stats[p]["output"].count > 0


def test_hf_adapter_calibration_texts_settable_after_construction():
    """texts may be set later (as the App does at calibration time); the slice
    view (sample_inputs) works before that."""
    model = LlamaLike()
    adapter = HFCausalLMAdapter(model, DummyTok(), seq_length=8)  # no texts yet
    paths = [m.path for m in adapter.enumerate_modules().modules]
    si = adapter.sample_inputs(paths)  # works without texts
    assert si
    adapter.set_texts(["a b c"] * 4)
    agg = run_calibration(adapter, paths, num_bins=16)
    for p in paths:
        assert agg.stats[p]["output"].count > 0


# --- DiTAdapter (non-causal, no tokenizer) ---

def test_dit_adapter_calib_batches_deterministic():
    """Two calls produce identical batches (seeded) -- required for two-pass
    calibration and non-polluting across sample_inputs / calibration."""
    adapter = build_demo_dit_adapter()
    b1 = list(adapter.calib_batches(n_samples=4, batch_size=2))
    b2 = list(adapter.calib_batches(n_samples=4, batch_size=2))
    assert len(b1) == len(b2) == 2
    for a, b in zip(b1, b2):
        assert torch.equal(a["x"], b["x"])
        assert torch.equal(a["t"], b["t"])
        assert torch.equal(a["y"], b["y"])


def test_dit_adapter_enumerates_linear_modules():
    adapter = build_demo_dit_adapter()
    result = adapter.enumerate_modules()
    # MiniDiT is not Llama-named -> degraded (all Linear as OTHER), but every
    # DiT projection layer is present.
    assert result.degraded
    names = {m.path.split(".")[-1] for m in result.modules}
    assert {"attn_qkv", "attn_proj", "mlp_fc1", "mlp_fc2", "adaLN_modulation"} <= names


def test_dit_adapter_sample_inputs_captures_linear_input():
    adapter = build_demo_dit_adapter()
    paths = [m.path for m in adapter.enumerate_modules().modules]
    si = adapter.sample_inputs(paths)
    assert si
    for t in si.values():
        assert t.dim() >= 2


def test_dit_adapter_linear_mainline_end_to_end():
    """DiT drives the full Linear quantization main line: calibration (online
    aggregation) + weight_stats + fake_quant + output_diff (module(x)
    independent forward, valid for pure Linear layers)."""
    adapter = build_demo_dit_adapter()
    paths = [m.path for m in adapter.enumerate_modules().modules]

    agg = run_calibration(adapter, paths, collect_histogram=True, num_bins=16)
    for p in paths:
        assert agg.stats[p]["output"].count > 0

    p = paths[0]
    w = get_weight(adapter.model, p)
    ws = weight_stats(w, p, granularity=Granularity.PER_CHANNEL)
    assert ws

    cfg = QuantConfig(bits=4, granularity=Granularity.PER_CHANNEL,
                      symmetry=Symmetry.SYMMETRIC)
    q = fake_quantize_tensor(w, cfg)
    assert q.shape == w.shape

    si = adapter.sample_inputs([p])
    if p in si:
        module = adapter.model.get_submodule(p)
        diff = output_diff(module, q, si[p], quantize_activation=False)
        assert diff["mse"] >= 0.0
        assert 0.0 <= diff["cosine"] <= 1.0 + 1e-6


def test_dit_adapter_is_model_adapter_subclass():
    adapter = build_demo_dit_adapter()
    assert isinstance(adapter, DiTAdapter)
    assert hasattr(adapter, "calib_batches") and hasattr(adapter, "run_forward")
