"""Model adapter interface: decouples the quantization pipeline from any
specific model framework.

The pipeline (runner / sensitivity / charts) never calls ``from_pretrained``,
``tokenizer(...)``, or ``model(input_ids)`` directly. It talks to a
``ModelAdapter`` instead, which knows how to:

  - enumerate target modules (default: ``resolve_modules`` -- already falls
    back to enumerating all ``nn.Linear`` for unknown architectures),
  - yield calibration batches in whatever shape the model's forward expects,
  - run one forward pass (hooks capture activations),
  - produce one sample input per module (for ``output_diff`` sensitivity).

The Linear quantization main line (fake_quant / weight_stats / output_diff /
online aggregation) is unchanged and works for any model whose target layers
are ``nn.Linear``. To support a new model, subclass and override
``calib_batches`` + ``run_forward`` -- typically <10 lines. See
``adapters/README.md`` for the integration contract.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .hook import ActivationCapture
from .module_resolver import ResolveResult, resolve_modules

# A text long enough that a real tokenizer yields plenty of tokens for the
# on-demand per-token slice view (available right after model load, before any
# calibration). Overridable via ``HFCausalLMAdapter(sample_text=...)``.
_DEFAULT_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models are composed of stacked transformer blocks. "
    "Activation distributions inform quantization suitability decisions. "
    "Outlier channels dominate the activation range and drive quantization error. "
) * 8


class ModelAdapter:
    """Boundary between the pipeline and a concrete model framework.

    Subclasses must override :meth:`calib_batches` and :meth:`run_forward`.
    The defaults for :meth:`enumerate_modules` and :meth:`sample_inputs` work
    for any ``nn.Module`` whose target layers are ``nn.Linear``.
    """

    def __init__(self, model: nn.Module, device: Optional[Any] = None):
        self._model = model
        self._device = device or getattr(model, "device", torch.device("cpu"))

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self):
        return self._device

    def enumerate_modules(self) -> ResolveResult:
        """Enumerate target modules. Default reuses ``resolve_modules`` which
        already falls back to all ``nn.Linear`` for unknown architectures."""
        return resolve_modules(self._model)

    def calib_batches(self, n_samples: int, batch_size: int) -> Iterator[Any]:
        """Yield calibration batches in the shape the model's forward expects.

        Contract: **deterministic** -- the same ``(n_samples, batch_size)``
        must produce the same batch sequence, because two-pass calibration
        (Pass 1 fixes the range, Pass 2 fills the histogram) re-runs the same
        data. Use a seeded ``torch.Generator`` for random data.
        """
        raise NotImplementedError

    def run_forward(self, batch) -> None:
        """Run one forward pass; forward hooks capture activations. No return."""
        raise NotImplementedError

    def _sample_batch(self):
        """One batch for :meth:`sample_inputs` (sensitivity / slice view).

        Default: the first calibration batch. Override if a single fixed
        sample is preferable to the calibration stream (e.g. a fixed text).
        """
        return next(self.calib_batches(n_samples=1, batch_size=1))

    def sample_inputs(self, module_paths: List[str]) -> Dict[str, torch.Tensor]:
        """Run one forward and return one input activation per target module.

        Used by sensitivity analysis (``output_diff``): provides a real input
        tensor for each module so quantization error can be measured at the
        module output. Default implementation runs one forward via a capture
        hook; subclasses normally do NOT need to override this.
        """
        cap = ActivationCapture(module_paths, capture_inputs=True, capture_outputs=False)
        cap.attach(self._model)
        was_training = getattr(self._model, "training", False)
        self._model.eval()
        try:
            batch = self._sample_batch()
            with torch.no_grad():
                self.run_forward(batch)
            inputs: Dict[str, torch.Tensor] = {}
            for p in module_paths:
                entry = cap.buffer.get(p, {})
                if "input" in entry:
                    inputs[p] = entry["input"]
        finally:
            cap.detach()
            if was_training:
                self._model.train()
        return inputs


class HFCausalLMAdapter(ModelAdapter):
    """Adapter for HuggingFace causal LMs (Llama / Qwen / Mistral / DeepSeek /
    any ``AutoModelForCausalLM``). This is the default and reproduces the
    pre-adapter behavior: tokenize texts -> ``model(input_ids)``.

    ``texts`` may be set after construction (e.g. loaded at calibration time);
    ``sample_inputs`` only needs ``tokenizer`` + ``sample_text``, so the
    per-token slice view works right after load, before calibration.
    """

    def __init__(self, model, tokenizer, texts: Optional[List[str]] = None,
                 seq_length: int = 2048, sample_text: Optional[str] = None,
                 device: Optional[Any] = None):
        super().__init__(model, device)
        self._tokenizer = tokenizer
        self._texts: List[str] = texts or []
        self._seq_length = int(seq_length)
        self._sample_text = sample_text or _DEFAULT_SAMPLE_TEXT

    def set_texts(self, texts: List[str]) -> None:
        """Set (or replace) the calibration texts after construction."""
        self._texts = list(texts)

    def calib_batches(self, n_samples: int, batch_size: int) -> Iterator[torch.Tensor]:
        if not self._texts:
            raise RuntimeError(
                "HFCausalLMAdapter: calibration texts not set. Call set_texts() "
                "before running calibration."
            )
        texts = self._texts[:n_samples]
        bs = max(1, batch_size)
        for i in range(0, len(texts), bs):
            enc = self._tokenizer(
                texts[i:i + bs], return_tensors="pt", padding=True,
                truncation=True, max_length=self._seq_length,
            )
            yield enc["input_ids"].to(self._device)

    def run_forward(self, batch) -> None:
        self._model(batch)

    def _sample_batch(self) -> torch.Tensor:
        enc = self._tokenizer(
            [self._sample_text], return_tensors="pt", padding=True,
            truncation=True, max_length=self._seq_length,
        )
        return enc["input_ids"].to(self._device)


class DiTAdapter(ModelAdapter):
    """Reference adapter for Diffusion Transformers (DiT).

    DiT is a transformer of ``nn.Linear`` layers (qkv / proj / fc1 / fc2 /
    adaLN_modulation), so the entire Linear quantization main line applies
    unchanged. The only differences from a causal LM: no tokenizer, and the
    forward signature is ``model(x, t, y)`` (noisy latent, timestep, class
    label). Calibration data is (random or real) latents + timesteps.

    ``calib_batches`` is deterministic (seeded generator) so two-pass
    calibration sees the same data. For a DiT variant with a different
    forward signature, subclass and override :meth:`run_forward`.
    """

    def __init__(self, dit: nn.Module, latent_shape, num_classes: int,
                 device: Optional[Any] = None):
        super().__init__(dit, device)
        dit.eval()
        self._latent_shape = tuple(latent_shape)
        self._num_classes = int(num_classes)

    def calib_batches(self, n_samples: int, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        # Fresh seeded generator each call: deterministic AND non-polluting
        # (sample_inputs re-running calib_batches does not shift the stream
        # seen by a later calibration pass).
        gen = torch.Generator().manual_seed(0)
        bs = max(1, batch_size)
        produced = 0
        while produced < n_samples:
            n = min(bs, n_samples - produced)
            x = torch.randn(n, *self._latent_shape, generator=gen)
            t = torch.randint(0, 1000, (n,), generator=gen)
            y = torch.randint(0, self._num_classes, (n,), generator=gen)
            yield {"x": x.to(self._device), "t": t.to(self._device), "y": y.to(self._device)}
            produced += n

    def run_forward(self, batch) -> None:
        self._model(batch["x"], batch["t"], batch["y"])
