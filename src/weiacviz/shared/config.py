"""Application configuration via pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Values can be overridden via environment variables prefixed with ``WEIACVIZ_``
    (e.g. ``WEIACVIZ_MODEL_NAME_OR_PATH``) or a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="WEIACVIZ_",
        env_file=".env",
        extra="ignore",
    )

    # --- Model ---
    model_name_or_path: str = "Qwen/Qwen2-0.5B"
    dtype: Literal["fp32", "fp16", "bf16"] = "fp16"
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # --- Calibration data ---
    calibration_dataset: str = "wikitext2"
    calibration_split: str = "train"
    calibration_text_column: str = "text"
    calibration_samples: int = 128
    calibration_batch_size: int = 8
    calibration_seq_length: int = 2048
    calibration_collect_histogram: bool = True
    calibration_histogram_bins: int = 256

    # --- Cache / output ---
    cache_dir: Path = Field(default_factory=lambda: Path(".weiacviz_cache"))
    output_dir: Path = Field(default_factory=lambda: Path("outputs"))

    # --- Defaults for analysis ---
    default_histogram_bins: int = 256
    default_outlier_percentile: float = 99.9
    default_outlier_zscore: float = 3.0
    default_topk_sensitive: int = 10

    # --- HuggingFace endpoint (mirror for regions where huggingface.co is unreachable) ---
    hf_endpoint: str = "https://hf-mirror.com"


def get_settings(**overrides) -> Settings:
    """Build a Settings instance, applying optional overrides."""
    return Settings(**overrides)
