# Model Adapters — 接入自定义模型

WeiActVisualize 的量化主线（权重分布 / 激活捕获 / fake quant / 敏感性 / 报告）针对
`nn.Linear` 层。要接入一个新模型，只需实现一个 `ModelAdapter` 子类，告诉管道「怎么喂校准
数据、怎么跑 forward」。**Linear 主线自动复用，无需改管道代码。**

## 何时需要自定义 adapter

- 模型不是 HuggingFace causal LM（DiT / encoder / vision / 多模态 / 自定义 `nn.Module`）。
- forward 签名不是 `model(input_ids)`（如 DiT 的 `model(x, t, y)`）。
- 校准数据不是文本（如图像 latent）。

如果模型能被 `AutoModelForCausalLM.from_pretrained` 加载、`model(input_ids)` 前向，直接用
`HFCausalLMAdapter`（UI 默认）即可，无需自定义。

## 接入契约

子类化 `ModelAdapter`，override 两个方法：

```python
from weiacviz.loading.adapter import ModelAdapter

class MyAdapter(ModelAdapter):
    def calib_batches(self, n_samples, batch_size):
        """产出 n_samples 个校准 batch（按 batch_size 分批）。
        必须**确定性**：相同 (n_samples, batch_size) 产出相同序列——
        两遍校准（Pass 1 定 range、Pass 2 填直方图）要跑相同数据。
        随机数据用 seeded torch.Generator（见 DiTAdapter）。"""
        for ...:
            yield batch  # 任意形状，你的 run_forward 能消费即可

    def run_forward(self, batch):
        """跑一次 forward。激活由 forward hook 自动捕获，无需返回。"""
        self._model(batch["x"], batch["t"], ...)  # 你的 forward 签名
```

其余方法用基类默认：

- `enumerate_modules()`：默认调 `resolve_modules`，枚举所有 `nn.Linear`（未知架构族降级为
  OTHER）。若目标层不是 Linear，override 此方法。
- `sample_inputs(paths)`：默认跑一次 forward + input hook，返回每 module 的输入（给
  `output_diff` 敏感性）。一般不用 override。
- `_sample_batch()`：默认取 `calib_batches` 第一个 batch。若想用固定样本（如固定文本），
  override。

## 构造

```python
adapter = MyAdapter(my_model, device="cuda")  # model 已加载并 .eval()
```

`ModelAdapter.__init__(model, device=None)`；`device` 默认从 `model.device` 推断。

## 参考实现

- `HFCausalLMAdapter`（`loading/adapter.py`）：HF causal LM，tokenize 文本 -> `model(input_ids)`。
- `DiTAdapter`（`loading/adapter.py`）：DiT，随机 latent + timestep + label ->
  `model(x, t, y)`，seeded `torch.Generator` 保证确定性。
- `dit_demo.py`：mini-DiT 演示模型 + `build_demo_dit_adapter()`（UI「DiT 演示」入口用）。

## 限制

- 量化主线只处理 `nn.Linear` 的 2D 权重。`Conv2d` / 非 Linear 层的 per-channel / per-group
  量化轴假设不成立，会降级。
- `output_diff` 敏感性要求目标层可脱离模型独立前向（`module(x)`）。带残差绑定的层 / 融合算子
  可能不满足，会降级为 NaN（不崩），但权重分布 / 激活捕获 / fake quant 仍可用。
- DiT forward 签名变体多（`(x,t,y)` / `(x,t,c)` / `(x,timesteps,context)`）；`DiTAdapter` 是
  参考，按你的签名 override `run_forward`。
