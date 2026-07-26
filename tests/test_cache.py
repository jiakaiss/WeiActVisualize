"""Tests for shared.cache."""
import pandas as pd

from weiacviz.shared.cache import ResultCache, compute_fingerprint


def test_fingerprint_stable():
    a = compute_fingerprint("x", 1, {"k": "v"})
    b = compute_fingerprint("x", 1, {"k": "v"})
    assert a == b


def test_fingerprint_differs_on_input():
    assert compute_fingerprint("a") != compute_fingerprint("b")
    assert compute_fingerprint(1) != compute_fingerprint(2)


def test_json_roundtrip(tmp_path):
    c = ResultCache(tmp_path)
    fp = compute_fingerprint("key")
    assert c.get_json(fp) is None
    c.put_json(fp, {"x": 1})
    assert c.get_json(fp) == {"x": 1}
    assert c.exists(fp)


def test_dataframe_roundtrip(tmp_path):
    c = ResultCache(tmp_path)
    fp = compute_fingerprint("df")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    c.put_dataframe(fp, df)
    got = c.get_dataframe(fp)
    assert list(got["a"]) == [1, 2]
    assert list(got["b"]) == [3, 4]


def test_config_change_invalidates(tmp_path):
    c = ResultCache(tmp_path)
    fp1 = compute_fingerprint("model-A", "layer.0", "weight")
    fp2 = compute_fingerprint("model-B", "layer.0", "weight")
    c.put_json(fp1, {"v": 1})
    assert c.get_json(fp2) is None  # different config -> miss
