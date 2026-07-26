"""Non-invasive forward hooks to capture activations."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ActivationCapture:
    """Register forward hooks on target modules to capture input/output activations.

    Captured tensors are detached and moved to CPU so GPU memory is not held
    across batches. The buffer is cleared between batches by the runner so that
    only aggregated statistics (not raw activations) accumulate.
    """

    def __init__(self, module_paths: List[str],
                 capture_inputs: bool = True, capture_outputs: bool = True):
        self.module_paths = set(module_paths)
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    def attach(self, model: nn.Module) -> None:
        for path, mod in model.named_modules():
            if path in self.module_paths:
                handle = mod.register_forward_hook(self._make_hook(path))
                self._handles.append(handle)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, path: str):
        def hook(module, inputs, output):
            entry: Dict[str, torch.Tensor] = {}
            if self.capture_inputs and inputs:
                inp = inputs[0]
                if isinstance(inp, torch.Tensor):
                    entry["input"] = inp.detach().to("cpu")
            if self.capture_outputs:
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    entry["output"] = out.detach().to("cpu")
            self._buffer[path] = entry
        return hook

    def get(self, path: str) -> Optional[Dict[str, torch.Tensor]]:
        return self._buffer.get(path)

    @property
    def buffer(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self._buffer

    def clear(self) -> None:
        self._buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.detach()
