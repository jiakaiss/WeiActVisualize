"""Progress reporting bridge between runner and Gradio UI."""
from __future__ import annotations

from typing import Callable, Optional


class ProgressReporter:
    """Adapts runner's (done, total) progress callback to a UI sink."""

    def __init__(self, sink: Optional[Callable[[float, str], None]] = None):
        self.sink = sink
        self.last_done = 0
        self.last_total = 0

    def callback(self, done: int, total: int) -> None:
        self.last_done = done
        self.last_total = total
        frac = done / total if total else 0.0
        msg = f"{done}/{total} batches"
        if self.sink is not None:
            self.sink(frac, msg)

    def as_callback(self) -> Callable[[int, int], None]:
        return self.callback


def gradio_progress_adapter(gr_progress):
    """Wrap a gr.Progress into a sink(frac, desc) callable."""

    def sink(frac: float, desc: str):
        try:
            gr_progress(frac, desc=desc)
        except Exception:  # noqa: BLE001
            pass

    return sink
