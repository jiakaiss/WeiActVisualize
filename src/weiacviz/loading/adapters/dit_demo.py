"""Mini DiT (Diffusion Transformer) demo model + adapter builder.

A tiny but real DiT-shaped transformer -- ``nn.Linear`` qkv / proj / fc1 / fc2 /
adaLN_modulation with a ``forward(x, t, y)`` signature -- so the UI can
demonstrate the full DiT visualization path (weights / activations / fake
quant / sensitivity) without downloading a real diffusion model.

Real DiT models (or any diffusion transformer) should be wrapped directly via
``DiTAdapter`` from Python: subclass and override ``run_forward`` if the
forward signature differs. See ``README.md`` for the integration contract.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..adapter import DiTAdapter


class MiniDiTBlock(nn.Module):
    """One DiT block: (pseudo) attention + MLP + adaLN modulation path."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.attn_qkv = nn.Linear(d, 3 * d)
        self.attn_proj = nn.Linear(d, d)
        self.mlp_fc1 = nn.Linear(d, 4 * d)
        self.mlp_fc2 = nn.Linear(4 * d, d)
        self.adaLN_modulation = nn.Linear(d, 6 * d)

    def forward(self, x, c):
        q, k, v = self.attn_qkv(x).chunk(3, dim=-1)
        x = x + self.attn_proj(q + k + v)      # pseudo-attention (no heads)
        x = x + self.mlp_fc2(torch.relu(self.mlp_fc1(x)))
        _ = self.adaLN_modulation(c)           # adaLN path; weights still analyzed
        return x


class MiniDiT(nn.Module):
    """Minimal DiT.

    Args:
        d: hidden / embedding dim.
        n_blocks: number of DiT blocks.
        num_classes: class-label vocabulary size (for ``y``).
        seq_len: token sequence length T (latent shape is ``(T, d)``).

    Forward: ``x`` [B, T, d] noisy latent, ``t`` [B] timestep, ``y`` [B] label.
    """

    def __init__(self, d: int = 16, n_blocks: int = 2, num_classes: int = 10,
                 seq_len: int = 8):
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        self.x_embed = nn.Linear(d, d)
        self.t_embed = nn.Linear(1, d)
        self.y_embed = nn.Embedding(num_classes, d)
        self.blocks = nn.ModuleList([MiniDiTBlock(d) for _ in range(n_blocks)])
        self.final = nn.Linear(d, d)

    def forward(self, x, t, y):
        x = self.x_embed(x)
        c = self.t_embed(t.float().unsqueeze(-1)) + self.y_embed(y)  # [B, d]
        for blk in self.blocks:
            x = blk(x, c)
        return self.final(x)


def build_demo_dit_adapter(d: int = 16, n_blocks: int = 2,
                           num_classes: int = 10, seq_len: int = 8,
                           device: Optional[str] = None) -> DiTAdapter:
    """Build a MiniDiT wrapped in a DiTAdapter for UI demonstration."""
    model = MiniDiT(d=d, n_blocks=n_blocks, num_classes=num_classes, seq_len=seq_len)
    latent_shape = (seq_len, d)
    return DiTAdapter(model, latent_shape=latent_shape,
                      num_classes=num_classes, device=device)
