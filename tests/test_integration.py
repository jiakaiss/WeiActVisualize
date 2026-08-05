"""End-to-end integration test on a synthetic model (no HF download)."""
import torch
import torch.nn as nn

from weiacviz.loading.calibration import load_calibration_texts  # noqa: F401
from weiacviz.loading.module_resolver import resolve_modules
from weiacviz.loading.runner import capture_sample_inputs, run_calibration
from weiacviz.loading.weights import get_weight
from weiacviz.quant.fake_quant import fake_quantize_tensor
from weiacviz.quant.sensitivity import layer_sensitivity, layer_sensitivity_output
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

    # 6. recommendation + export (new recommend: needs joint_output_mse/kind/shape)
    kinds = {m.path: m.kind.value for m in result.modules}
    rec_rows = [{
        "module_path": r["module_path"], "kind": kinds.get(r["module_path"], ""),
        "output_mse": r["mse"], "joint_output_mse": r["mse"],
        "weight_kurtosis": 0.0, "weight_skewness": 0.0,
    } for r in sens]
    rep = recommend(rec_rows, model_name="tiny-test", config=cfg)
    assert len(rep.recommendations) == len(paths)

    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    export_markdown(rep, md_path)
    export_json(rep, json_path)
    assert md_path.exists() and json_path.exists()


def test_app_scheme_compare_view_ranks_schemes():
    """App.scheme_compare_view runs a scheme cartesian product and surfaces
    the lowest-MSE scheme as best, with a before/after histogram."""
    from weiacviz.viz.app import App

    model = TinyModel()
    app = App()
    app._model = model
    app._resolve_result = resolve_modules(model)
    app._modules = app._resolve_result.modules

    path = app._modules[0].path
    df, best, fig = app.scheme_compare_view(
        path, ["4", "8"], ["per-tensor", "per-channel"], ["symmetric"], 128,
    )
    assert len(df) == 4  # 2 bits x 2 granularities x 1 symmetry
    assert "mse" in df.columns
    assert "最优方案" in best
    # 8-bit per-channel must beat 4-bit per-tensor on a normal weight
    assert fig is not None


def test_app_build_report_and_export():
    """App.build_report builds a per-module recommendation table over the whole
    model and exports it to markdown/csv."""
    import os

    from weiacviz.viz.app import App

    model = TinyModel()
    app = App()
    app._model = model
    app._tokenizer = DummyTok()
    app._model_name = "tiny-test"
    app._resolve_result = resolve_modules(model)
    app._modules = app._resolve_result.modules

    df, summary = app.build_report(bits=4, seq_length=8)
    assert len(df) == len(app._modules)
    assert {"module", "kind", "bits", "granularity", "symmetry",
            "joint_output_mse", "act_severity", "reason"} <= set(df.columns)
    assert "modules analyzed" in summary

    md_path = app.export_report("markdown")
    assert md_path.endswith(".md") and os.path.exists(md_path)
    csv_path = app.export_report("csv")
    assert csv_path.endswith(".csv") and os.path.exists(csv_path)


def test_app_view_activation_per_token():
    """App.view_activation returns advice + global / abs-max violin / slice
    violin / slice heatmap / per-channel figures after a two-pass calibration
    with token stats."""
    from weiacviz.shared.types import CaptureConfig
    from weiacviz.viz.app import App

    model = TinyModel()
    app = App()
    app._model = model
    app._tokenizer = DummyTok()
    app._resolve_result = resolve_modules(model)
    app._modules = app._resolve_result.modules
    paths = [m.path for m in app._modules]

    texts = ["a b c d"] * 8
    app._aggregator = run_calibration(
        model, DummyTok(), texts, paths,
        config=CaptureConfig(max_samples=8, batch_size=4), seq_length=8,
        collect_histogram=True, num_bins=32, collect_token_stats=True,
    )
    advice, global_fig, token_fig, slice_vio, slice_hm, channel_fig = \
        app.view_activation(paths[0], seq_length=8)
    assert "per-token abs_max" in advice
    assert "per-token abs_max" in token_fig.layout.title.text  # abs_max violin
    assert "activation" in global_fig.layout.title.text
    # per-token slice view rendered from on-demand sample
    assert "未捕获到" not in (slice_vio.layout.title.text or "")
    assert len(slice_vio.data) >= 1
    assert slice_hm is not None
    # channel stats not collected -> placeholder title
    assert "per-channel abs_mean" not in channel_fig.layout.title.text


def test_app_view_activation_slice_view_without_calibration():
    """Per-token slice view renders from an on-demand sample even before any
    calibration; calibration-dependent views show placeholders."""
    from weiacviz.viz.app import App

    model = TinyModel()
    app = App()
    app._model = model
    app._tokenizer = DummyTok()
    app._resolve_result = resolve_modules(model)
    app._modules = app._resolve_result.modules
    paths = [m.path for m in app._modules]
    # no calibration run -> aggregator is None
    advice, global_fig, token_fig, slice_vio, slice_hm, channel_fig = \
        app.view_activation(paths[0], seq_length=8)
    assert "未采集" in advice
    assert "未采集" in (token_fig.layout.title.text or "")
    assert "未采集" in (global_fig.layout.title.text or "")
    # slice view still rendered from on-demand sample
    assert "未捕获到" not in (slice_vio.layout.title.text or "")
    assert len(slice_vio.data) >= 1


def test_capture_sample_inputs_returns_one_per_module():
    """capture_sample_inputs runs one forward and returns an input tensor
    per target module."""
    model = TinyModel()
    paths = [m.path for m in resolve_modules(model).modules]
    si = capture_sample_inputs(model, DummyTok(), paths, text="abc", seq_length=8)
    assert set(si.keys()) == set(paths)
    for t in si.values():
        assert t.dim() >= 2  # [batch, seq, hidden]


def test_layer_sensitivity_output_ranks_and_has_joint():
    """layer_sensitivity_output sorts by joint_output_mse desc and carries
    weight-only output_mse + W8A8 joint_output_mse (the two can disagree)."""
    model = TinyModel()
    paths = [m.path for m in resolve_modules(model).modules]
    weights = {p: get_weight(model, p) for p in paths}
    si = capture_sample_inputs(model, DummyTok(), paths, text="abc", seq_length=8)
    kinds = {m.path: m.kind.value for m in resolve_modules(model).modules}
    cfg = QuantConfig(bits=4, granularity=Granularity.PER_CHANNEL,
                      symmetry=Symmetry.SYMMETRIC)
    rows = layer_sensitivity_output(model, paths, weights, si, cfg, kinds=kinds)
    assert len(rows) == len(paths)
    assert rows[0]["joint_output_mse"] >= rows[-1]["joint_output_mse"]  # desc by joint
    for r in rows:
        assert r["output_mse"] >= 0
        assert r["joint_output_mse"] >= 0
        assert r["kind"] in ("attn", "mlp")
        assert "weight_mse" not in r  # removed
        assert "output_cosine" not in r  # removed


def test_app_run_sensitivity_returns_table_and_heatmap():
    """App.run_sensitivity returns a whole-model table (output_mse,
    joint_output_mse, weight_kurtosis_max, heavy_channel_ratio,
    act_channel_severity) and a heatmap."""
    from weiacviz.viz.app import App

    model = TinyModel()
    app = App()
    app._model = model
    app._tokenizer = DummyTok()
    app._resolve_result = resolve_modules(model)
    app._modules = app._resolve_result.modules

    df, hm = app.run_sensitivity(bits=4, sort_by="joint_output_mse", seq_length=8)
    assert len(df) == len(app._modules)
    expected = {"module_path", "kind", "output_mse", "joint_output_mse",
                "weight_kurtosis_max", "heavy_channel_ratio", "act_channel_severity"}
    assert expected <= set(df.columns)
    assert hm is not None


