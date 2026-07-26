"""End-to-end integration test on a synthetic model (no HF download)."""
import torch
import torch.nn as nn

from weiacviz.loading.calibration import load_calibration_texts  # noqa: F401
from weiacviz.loading.module_resolver import resolve_modules
from weiacviz.loading.runner import run_calibration
from weiacviz.loading.weights import get_weight
from weiacviz.quant.fake_quant import fake_quantize_tensor
from weiacviz.quant.sensitivity import layer_sensitivity
from weiacviz.report.export import export_json, export_markdown
from weiacviz.report.recommend import recommend
from weiacviz.shared.types import Granularity, QuantConfig, Symmetry
from weiacviz.stats.outliers import detect_outliers_percentile
from weiacviz.stats.weight_stats import weight_stats


class TinyBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d)
        self.self_attn.k_proj = nn.Linear(d, d)
        self.self_attn.v_proj = nn.Linear(d, d)
        self.self_attn.o_proj = nn.Linear(d, d)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, d * 2)
        self.mlp.up_proj = nn.Linear(d, d * 2)
        self.mlp.down_proj = nn.Linear(d * 2, d)

    def forward(self, x):
        a = self.self_attn.o_proj(
            self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x)
        )
        m = self.mlp.down_proj(self.mlp.up_proj(x) * torch.sigmoid(self.mlp.gate_proj(x)))
        return x + a + m


class TinyModel(nn.Module):
    def __init__(self, d=8, n=2):
        super().__init__()
        self.config = type("C", (), {"architectures": ["LlamaForCausalLM"]})()
        self.embed = nn.Linear(d, d)
        self.layers = nn.ModuleList([TinyBlock(d) for _ in range(n)])

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DummyTok:
    def __call__(self, texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048):
        n = len(texts)
        return {"input_ids": torch.randn(n, max_length)}


def test_end_to_end_pipeline(tmp_path):
    model = TinyModel()

    # 1. resolve target modules
    result = resolve_modules(model)
    assert not result.degraded
    paths = [m.path for m in result.modules]
    assert len(paths) > 0

    # 2. calibration + online activation aggregation
    texts = ["a b c d"] * 16
    agg = run_calibration(model, DummyTok(), texts, paths, seq_length=8)
    for p in paths:
        assert agg.stats[p]["output"].count > 0

    # 3. weight stats + outliers
    weights = {p: get_weight(model, p) for p in paths}
    for p, w in weights.items():
        st = weight_stats(w, p, granularity=Granularity.PER_CHANNEL)
        assert len(st) > 0
        detect_outliers_percentile(w, percentile=99.0)

    # 4. quantization simulation + sensitivity
    cfg = QuantConfig(bits=4, granularity=Granularity.PER_TENSOR, symmetry=Symmetry.SYMMETRIC)
    sens = layer_sensitivity(weights, cfg, topk=3)
    assert len(sens) == len(paths)
    assert sens[0]["mse"] >= sens[-1]["mse"]

    # 5. fake quant shape preserved on every module
    for p, w in weights.items():
        assert fake_quantize_tensor(w, cfg).shape == w.shape

    # 6. recommendation + export
    outlier_ratios = {
        p: detect_outliers_percentile(w, 99.0).outlier_ratio for p, w in weights.items()
    }
    rep = recommend(sens, outlier_ratios, model_name="tiny-test", config=cfg)
    assert len(rep.recommendations) == len(paths)

    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    export_markdown(rep, md_path)
    export_json(rep, json_path)
    assert md_path.exists() and json_path.exists()
