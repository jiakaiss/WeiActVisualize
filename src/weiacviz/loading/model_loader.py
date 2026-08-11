"""HuggingFace model loading with dtype control and multi-backend device."""
from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import HFCausalLMAdapter

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


def detect_available_backend() -> str:
    """Detect the best available compute backend: cuda > npu > cpu.

    NPU detection guards the ``torch_npu`` import so environments without
    Ascend tooling do not crash.
    """
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _move_to_backend(model, device: str):
    """Move an already-loaded model onto ``device`` (npu)."""
    if device == "npu":
        try:
            import torch_npu  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "device='npu' 需要 Ascend NPU 后端，请安装 torch_npu"
            ) from e
    return model.to(device)


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
        device: "auto" | "cpu". "auto" detects the best available accelerator
            (cuda > npu > cpu); cuda uses device_map="auto" sharding (large-model
            friendly), npu uses model.to. "cpu" forces CPU.

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

    if device == "auto":
        device = detect_available_backend()

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch_dtype,
            device_map="cpu", trust_remote_code=trust_remote_code,
        )
    elif device == "cuda":
        # device_map="auto" shards across GPUs with CPU offload as needed,
        # supporting models larger than a single GPU's memory.
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch_dtype,
            device_map="auto", trust_remote_code=trust_remote_code,
        )
    else:  # npu: load on CPU then move (no device_map sharding)
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        model = _move_to_backend(model, device)

    model.eval()
    return model, tokenizer


def load_causal_lm_adapter(
    model_name_or_path: str,
    dtype: str = "fp16",
    device: str = "auto",
    trust_remote_code: bool = False,
    seq_length: int = 2048,
) -> HFCausalLMAdapter:
    """Load a causal LM via ``load_model`` and wrap it in an ``HFCausalLMAdapter``.

    Calibration texts are set later (``set_texts``) at calibration time, so
    ``sample_inputs`` (per-token slice view) works immediately after load.
    """
    model, tokenizer = load_model(
        model_name_or_path, dtype=dtype, device=device,
        trust_remote_code=trust_remote_code,
    )
    return HFCausalLMAdapter(model, tokenizer, seq_length=seq_length)
