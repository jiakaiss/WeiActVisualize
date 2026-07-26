"""Tests for shared.config."""
from weiacviz.shared.config import Settings, get_settings


def test_default_settings():
    s = Settings()
    assert s.dtype in ("fp32", "fp16", "bf16")
    assert s.calibration_samples > 0
    assert s.default_histogram_bins > 0


def test_override_via_kwargs():
    s = get_settings(model_name_or_path="test/model", dtype="fp32")
    assert s.model_name_or_path == "test/model"
    assert s.dtype == "fp32"


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("WEIACVIZ_MODEL_NAME_OR_PATH", "env/model")
    monkeypatch.setenv("WEIACVIZ_CALIBRATION_SAMPLES", "64")
    s = Settings()
    assert s.model_name_or_path == "env/model"
    assert s.calibration_samples == 64
