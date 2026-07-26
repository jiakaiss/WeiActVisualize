"""Tests for viz.structure."""
import torch
import torch.nn as nn

from weiacviz.loading.module_resolver import resolve_modules
from weiacviz.viz.structure import build_module_table, build_overview, _layer_index


class TinyAttn(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class TinyMLP(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2)
        self.up_proj = nn.Linear(d, d * 2)
        self.down_proj = nn.Linear(d * 2, d)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * torch.sigmoid(self.gate_proj(x)))


class TinyBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.self_attn = TinyAttn(d)
        self.mlp = TinyMLP(d)

    def forward(self, x):
        return x + self.mlp(self.self_attn(x))


class TinyModel(nn.Module):
    def __init__(self, d=8, n=3):
        super().__init__()
        self.config = type("C", (), {"architectures": ["LlamaForCausalLM"]})()
        self.embed = nn.Linear(d, d)
        self.layers = nn.ModuleList([TinyBlock(d) for _ in range(n)])

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


def test_layer_index_parsing():
    assert _layer_index("model.layers.0.self_attn.q_proj") == 0
    assert _layer_index("model.layers.5.mlp.gate_proj") == 5
    assert _layer_index("model.embed_tokens") == -1


def test_module_table_params_and_sort():
    model = TinyModel(n=3)
    result = resolve_modules(model)
    rows = build_module_table(result, model)
    # each layer: 4 attn + 3 mlp = 7 modules; 3 layers -> 21
    assert len(rows) == 21
    # q_proj params: 8*8 weight + 8 bias = 72
    qrow = next(r for r in rows if r["path"].endswith("q_proj"))
    assert qrow["params"] == 8 * 8 + 8
    # sorted by layer index ascending
    layers = [r["layer"] for r in rows]
    assert layers == sorted(layers)
    assert layers[0] == 0
    # within a layer, attn comes before mlp
    layer0 = [r for r in rows if r["layer"] == 0]
    kinds = [r["kind"] for r in layer0]
    assert kinds.index("attn") < kinds.index("mlp")


def test_overview_content():
    model = TinyModel(n=3)
    result = resolve_modules(model)
    ov = build_overview(result, model)
    assert "架构族" in ov
    assert "`llama`" in ov
    assert "transformer 层数" in ov
    assert "目标 module 数" in ov
    assert "21" in ov
