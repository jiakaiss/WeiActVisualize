"""Calibration data loading and tokenization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


def load_calibration_texts(
    dataset: str = "wikitext2",
    split: str = "train",
    text_column: str = "text",
    num_samples: int = 128,
    local_path: Optional[str] = None,
) -> List[str]:
    """Load raw text samples for calibration.

    Supports WikiText2 / C4 from HuggingFace datasets, or a local text file
    (blank-line or newline separated documents).
    """
    if local_path is not None:
        return _load_local_texts(local_path, num_samples)

    name = dataset.lower()
    try:
        if name == "wikitext2":
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        elif name == "c4":
            from datasets import load_dataset
            ds = load_dataset("allenai/c4", split=split, streaming=True)
        else:
            from datasets import load_dataset
            ds = load_dataset(name, split=split)

        texts: List[str] = []
        for ex in ds:
            t = ex.get(text_column, "")
            if t and t.strip():
                texts.append(t)
            if len(texts) >= num_samples:
                break
        if texts:
            return texts
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to load dataset '%s' (%s); falling back to built-in sample "
            "texts. Set local_path for real calibration data.",
            dataset, type(e).__name__,
        )

    return _builtin_sample_texts(num_samples)


def _load_local_texts(path: str, num_samples: int) -> List[str]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
    if len(docs) <= 1:
        docs = [d.strip() for d in raw.splitlines() if d.strip()]
    return docs[:num_samples] if num_samples > 0 else docs


def _builtin_sample_texts(num_samples: int) -> List[str]:
    """Return built-in sample texts (fallback when dataset download fails)."""
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be quantized to reduce memory and inference cost.",
        "Large language models are typically composed of stacked transformer blocks.",
        "Weight and activation distributions inform quantization suitability decisions.",
        "Outlier channels in activations are a primary source of quantization error.",
        "Per-channel quantization often reduces error versus per-tensor schemes.",
        "Four bits can represent sixteen distinct levels on a symmetric grid.",
        "Calibration data should be representative of the model's target distribution.",
        "The cosine similarity between original and quantized outputs measures fidelity.",
        "Sensitivity analysis ranks layers by their quantization-induced loss.",
        "Attention projections include query, key, value and output matrices.",
        "MLP blocks use gate, up and down projections in modern architectures.",
        "Lower bit-widths generally increase quantization error non-linearly.",
        "Group-wise quantization trades overhead for finer granularity control.",
        "Online aggregation keeps memory usage independent of calibration size.",
    ]
    if num_samples <= 0:
        return samples
    reps = (num_samples // len(samples)) + 1
    return (samples * reps)[:num_samples]


def tokenize_batch(tokenizer, texts: List[str], seq_length: int = 2048):
    """Tokenize a list of texts into a padded batch."""
    return tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=seq_length,
    )


def iter_batches(tokenizer, texts: List[str], batch_size: int = 8,
                 seq_length: int = 2048) -> Iterator:
    """Yield tokenized batches of calibration data."""
    for i in range(0, len(texts), batch_size):
        yield tokenize_batch(tokenizer, texts[i:i + batch_size], seq_length=seq_length)
