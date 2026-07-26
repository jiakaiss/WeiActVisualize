"""HuggingFace model loading with dtype control and sharded device_map."""
from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,
            device_map="cpu", trust_remote_code=trust_remote_code,
        )
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        model = model.to("cuda")
    else:  # auto: shard + offload
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,
            device_map="auto", trust_remote_code=trust_remote_code,
        )

    model.eval()
    return model, tokenizer
