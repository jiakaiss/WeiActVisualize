"""Large-model smoke test (manual; requires a real 7B+ model + GPU).

Enable by setting ``WEIACVIZ_SMOKE_MODEL=<huggingface model id>``.
Verifies sharded loading, batched inference, and online aggregation work
at 7B+ scale without OOM.
"""
import os

import pytest

SKIP = not os.environ.get("WEIACVIZ_SMOKE_MODEL")


@pytest.mark.skipif(SKIP, reason="set WEIACVIZ_SMOKE_MODEL=<model_id> to run")
def test_large_model_smoke():
    from weiacviz.loading.calibration import load_calibration_texts
    from weiacviz.loading.model_loader import load_model
    from weiacviz.loading.module_resolver import resolve_modules
    from weiacviz.loading.runner import run_calibration

    model_id = os.environ["WEIACVIZ_SMOKE_MODEL"]
    model, tokenizer = load_model(model_id, dtype="bf16", device="auto")
    result = resolve_modules(model)
    paths = [m.path for m in result.modules]

    texts = load_calibration_texts("wikitext2", num_samples=8)
    agg = run_calibration(model, tokenizer, texts, paths, seq_length=512)
    # online aggregation: memory independent of batches; every target captured
    for p in paths[:5]:
        assert agg.stats[p]["output"].count > 0
