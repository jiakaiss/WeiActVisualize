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
        ModuleRecommendation("a", 4, "per-tensor", "low", mse=1e-5, outlier_ratio=0.0),
        ModuleRecommendation("b", 8, "per-channel", "high", mse=1e-2, outlier_ratio=0.05),
    ]
    return RecommendationReport(model="test", config={"bits": 4},
                                recommendations=recs, summary="2 modules.")


def test_recommend_high_mse_gets_8bit():
    sens = [
        {"module_path": "a", "mse": 1e-6, "top": False},
        {"module_path": "b", "mse": 1e-2, "top": True},
    ]
    outliers = {"a": 0.0, "b": 0.05}
    rep = recommend(sens, outliers, model_name="m")
    by_path = {r.module_path: r for r in rep.recommendations}
    assert by_path["a"].recommended_bits == 4
    assert by_path["b"].recommended_bits == 8
    assert "modules analyzed" in rep.summary


def test_export_markdown(tmp_path):
    rep = _fake_report()
    p = tmp_path / "report.md"
    text = export_markdown(rep, p)
    assert p.exists()
    assert "量化建议报告" in text
    assert "a" in text and "b" in text


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
