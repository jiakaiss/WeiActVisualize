"""Custom-model adapter template -- copy me, fill in two methods, verify.

Workflow (full guide in ``README.md`` next to this file):

1. Copy this class into your project and load your model.
2. Implement ``calib_batches`` (yield calibration batches; MUST be
   deterministic) and ``run_forward`` (one forward pass, your signature).
3. Run ``python -m weiacviz.loading.adapters.template`` and make sure the
   verification report is all PASS before wiring the adapter into the UI /
   pipeline.

The whole Linear main line (weight stats / activation capture / fake quant /
sensitivity / report) is reused automatically -- nothing else to implement.
"""
from __future__ import annotations

from typing import Iterator

import torch

from ..adapter import ModelAdapter


class CustomModelAdapter(ModelAdapter):
    """TODO: replace the two methods below with your model's data & forward.

    Keep these rules in mind:
      - the model should be ``.eval()``-ed and on ``device`` before you
        construct the adapter (pass ``device=`` if it has no ``.device``),
      - target layers must be ``nn.Linear`` for the quantization main line,
      - batches may be any structure (tensor, dict of tensors, ...) as long
        as YOUR ``run_forward`` knows how to consume them.
    """

    def calib_batches(self, n_samples: int, batch_size: int) -> Iterator[dict]:
        """Yield ``ceil(n_samples / batch_size)`` calibration batches.

        MUST be deterministic: the same ``(n_samples, batch_size)`` must
        produce the same batch sequence every call -- two-pass calibration
        (pass 1 fixes the range, pass 2 fills the histogram) replays this
        stream. For random data, create a seeded generator INSIDE the method
        (like below) so re-calls re-produce the stream without shifting it.
        """
        gen = torch.Generator().manual_seed(0)  # your fixed seed
        produced = 0
        while produced < n_samples:
            n = min(batch_size, n_samples - produced)
            yield {
                # TODO: replace with your model's real calibration inputs.
                # Example shapes below fit a DiT-like ``forward(x, t, y)``.
                "x": torch.randn(n, 64, 256, generator=gen),
                "t": torch.randint(0, 1000, (n,), generator=gen),
            }
            produced += n

    def run_forward(self, batch) -> None:
        """One forward pass; forward hooks capture activations, no return.

        TODO: replace with your model's signature, e.g.
            self._model(input_ids=batch)                    # causal LM
            self._model(batch["x"], batch["t"], batch["y"])  # DiT-like
            self._model(pixel_values=batch["pixels"])       # vision
        """
        self._model(batch["x"], batch["t"])


if __name__ == "__main__":
    # Self-verification entry point. With the TODO stubs above it cannot
    # run against a real model, so it demonstrates the check on the bundled
    # mini-DiT demo adapter instead -- replace these three lines with:
    #     my_model = ...  # load + .eval() + .to(device)
    #     report = verify_adapter(CustomModelAdapter(my_model, device=...))
    from .diagnose import verify_adapter
    from .dit_demo import build_demo_dit_adapter

    print(verify_adapter(build_demo_dit_adapter()))
