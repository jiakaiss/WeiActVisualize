"""Tests for report module."""
import json

import pandas as pd

from weiacviz.report.export import export_csv, export_json, export_markdown
from weiacviz.report.recommend import (
    ModuleRecommendation,
    RecommendationReport,
    recommend,
)


def _fake_report():
    recs = [
        ModuleRecommendation(
            module_path="a", kind="attn", recommended_bits=4,
            recommended_granularity="per-channel", recommended_symmetry="symmetric",
            output_mse=1e-5, joint_output_mse=1e-5,
            weight_kurtosis_max=0.0, heavy_channel_ratio=0.0,
            reason="why: low | how: W4 per-channel symmetric | 激活 per-token"),
        ModuleRecommendation(
            module_path="b", kind="mlp", recommended_bits=8,
            recommended_granularity="per-group(128)", recommended_symmetry="asymmetric",
            output_mse=1e-2, joint_output_mse=1e-2,
            weight_kurtosis_max=50.0, heavy_channel_ratio=0.1,
            reason="why: high | how: W8 per-group(128) asymmetric | 激活 per-token"),
    ]
    return RecommendationReport(model="test", config={"bits": 4},
                                recommendations=recs, summary="2 modules.")


def _row(path, kind, out_mse, skew=0.0, joint=None,
         heavy_ratio=0.0, kurt_max=0.0):
    """Build a sensitivity row. ``joint`` defaults to ``out_mse`` (so weight-only
    and W8A8 are equal unless activation loss is being tested)."""
    return {"module_path": path, "kind": kind, "output_mse": out_mse,
            "joint_output_mse": out_mse if joint is None else joint,
            "weight_kurtosis_max": kurt_max, "heavy_channel_ratio": heavy_ratio,
            "weight_skewness": skew}


def test_recommend_low_vs_high_sensitivity():
    rows = [
        _row("a", "attn", 1e-6),
        _row("b", "mlp", 1e-2, heavy_ratio=0.1, kurt_max=50.0),  # high joint + heavy channels
    ]
    rep = recommend(rows, model_name="m")
    by = {r.module_path: r for r in rep.recommendations}
    assert by["a"].recommended_bits == 4                       # low sensitivity
    assert by["b"].recommended_bits == 8                       # high + heavy channels
    assert "per-group" in by["b"].recommended_granularity      # heavy channels -> per-group
    assert "modules analyzed" in rep.summary


def test_recommend_skew_asymmetric():
    rep = recommend([_row("a", "attn", 1e-6, skew=0.8)], model_name="m")
    assert rep.recommendations[0].recommended_symmetry == "asymmetric"


def test_recommend_down_proj_high_sens_w8_per_channel():
    # MLP down_proj, high sensitivity, no heavy channels -> W8 per-channel (not per-group)
    rows = [
        _row("layers.0.self_attn.q_proj", "attn", 1e-6),           # low sensitivity
        _row("layers.0.mlp.down_proj", "mlp", 1e-2),               # high, no heavy channels
    ]
    rep = recommend(rows, model_name="m")
    by = {r.module_path: r for r in rep.recommendations}
    rec = by["layers.0.mlp.down_proj"]
    assert rec.recommended_bits == 8
    assert rec.recommended_granularity == "per-channel"  # not per-group (no heavy channels)


def test_recommend_activation_loss_share_in_reason():
    # joint >> weight-only => activation quantization dominates the loss
    rep = recommend([_row("a", "mlp", out_mse=1e-3, joint=1e-2)], model_name="m")
    assert "激活量化损失占比" in rep.recommendations[0].reason


def test_export_markdown(tmp_path):
    rep = _fake_report()
    p = tmp_path / "report.md"
    text = export_markdown(rep, p)
    assert p.exists()
    assert "量化建议报告" in text
    assert "a" in text and "b" in text
    assert "joint_output_mse" in text


def test_export_csv(tmp_path):
    rows = [{"module": "a", "mse": 0.1}, {"module": "b", "mse": 0.2}]
    p = tmp_path / "data.csv"
    export_csv(rows, p)
    df = pd.read_csv(p)
    assert list(df["module"]) == ["a", "b"]


def test_export_json(tmp_path):
    rep = _fake_report()
    p = tmp_path / "report.json"
    export_json(rep, p)
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["recommendations"]) == 2
