"""Calibration data loading and tokenization."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional


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
    return texts


def _load_local_texts(path: str, num_samples: int) -> List[str]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
    if len(docs) <= 1:
        docs = [d.strip() for d in raw.splitlines() if d.strip()]
    return docs[:num_samples] if num_samples > 0 else docs


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
