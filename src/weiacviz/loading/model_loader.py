"""HuggingFace model loading with dtype control and sharded device_map."""
from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default to a HuggingFace mirror when HF_ENDPOINT is unset (helps regions where
# huggingface.co is unreachable). Override via the HF_ENDPOINT env var.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _resolve_model_path(model_name_or_path: str) -> str:
    """Return a local path to the model.

    Local directories are returned as-is; hub ids are downloaded via ModelScope
    (reliable in regions where huggingface.co is unreachable).
    """
    from pathlib import Path

    if Path(model_name_or_path).exists():
        return model_name_or_path
    from modelscope import snapshot_download

    return snapshot_download(model_name_or_path)


_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def load_model(
    model_name_or_path: str,
    dtype: str = "fp16",
    device: str = "auto",
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer.

    Args:
        model_name_or_path: HF hub id or local path.
        dtype: "fp32" | "fp16" | "bf16".
        device: "cpu" (CPU only), "cuda" (single GPU), or "auto"
            (device_map="auto": shard across devices with CPU offload as
            needed, enabling models larger than a single GPU's memory).

    Returns:
        (model, tokenizer) with model in eval mode.
    """
    torch_dtype = _DTYPE_MAP.get(dtype, torch.float16)
    resolved = _resolve_model_path(model_name_or_path)
    # Local load: stop transformers from reaching the Hub for metadata.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=trust_remote_code
    )

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch_dtype,
            device_map="cpu", trust_remote_code=trust_remote_code,
        )
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        model = model.to("cuda")
    else:  # auto: shard + offload
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch_dtype,
            device_map="auto", trust_remote_code=trust_remote_code,
        )

    model.eval()
    return model, tokenizer
