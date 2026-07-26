"""Fingerprinted on-disk cache for statistics / quantization results.

Avoids re-running expensive model inference when inputs/config are unchanged.
Cache key is a stable SHA-256 fingerprint of the relevant config parts; any
config change naturally invalidates the entry (a miss).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


def compute_fingerprint(*parts: Any) -> str:
    """Stable hash of arbitrary JSON-serialisable parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ResultCache:
    """JSON / Parquet cache keyed by fingerprint."""

    def __init__(self, base_dir: Union[Path, str]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, fingerprint: str, ext: str) -> Path:
        return self.base_dir / f"{fingerprint}.{ext}"

    # --- JSON ---
    def get_json(self, fingerprint: str) -> Optional[Any]:
        p = self._path(fingerprint, "json")
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def put_json(self, fingerprint: str, data: Any) -> None:
        p = self._path(fingerprint, "json")
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, default=str, ensure_ascii=False)

    # --- DataFrame ---
    def get_dataframe(self, fingerprint: str) -> Optional[pd.DataFrame]:
        p = self._path(fingerprint, "parquet")
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def put_dataframe(self, fingerprint: str, df: pd.DataFrame) -> None:
        p = self._path(fingerprint, "parquet")
        df.to_parquet(p, index=False)

    # --- helpers ---
    def exists(self, fingerprint: str) -> bool:
        return self._path(fingerprint, "json").exists() or self._path(fingerprint, "parquet").exists()

    def clear(self) -> None:
        for p in self.base_dir.glob("*"):
            if p.is_file():
                p.unlink()
