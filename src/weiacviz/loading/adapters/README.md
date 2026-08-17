# Model Adapters - 新模型适配指南

WeiActVisualize 的量化主线(权重分布 / 激活捕获 / fake quant / 敏感性 / 报告)针对
`nn.Linear` 层。要接入一个新模型,只需实现一个 `ModelAdapter` 子类,告诉管道「怎么喂校准
数据、怎么跑 forward」。**Linear 主线自动复用,无需改管道代码。**

## 第一步:判断走哪条路

```
你的模型是什么?
├─ HF causal LM(AutoModelForCausalLM 能加载,forward 是 model(input_ids))
│    -> 零适配。HFCausalLMAdapter 直接用(UI 默认入口)
├─ DiT 且 forward 签名恰好是 model(x, t, y)
│    -> 零适配。DiTAdapter 直接用;签名不同则子类化,只 override run_forward
└─ 其他(encoder / vision / 多模态 / 自定义 nn.Module / forward 签名变体)
     -> 用 template.py 模板写一个子类(通常 <10 行),见第二步
```

特殊情形:**HF 但非 causal**(BERT / encoder 类)——`load_model` 走
`AutoModelForCausalLM` 会加载失败,需自己 `AutoModel.from_pretrained` 加载后手工构造
adapter(tokenize 后喂 `input_ids` 的套路与 `HFCausalLMAdapter` 相同,可直接子类化它并
override `run_forward`)。

## 第二步:实现两个方法

复制 `template.py`(同目录)改两个方法,或按下面的骨架写:

```python
from weiacviz.loading.adapter import ModelAdapter

class MyAdapter(ModelAdapter):
    def calib_batches(self, n_samples, batch_size):
        """产出 n_samples 个校准样本,按 batch_size 分批 yield。
        batch 可以是任意结构(tensor / dict of tensors),只要你的
        run_forward 能消费。"""
        gen = torch.Generator().manual_seed(0)   # 见下方确定性契约
        produced = 0
        while produced < n_samples:
            n = min(batch_size, n_samples - produced)
            yield {"x": torch.randn(n, ..., generator=gen),
                   "t": torch.randint(0, 1000, (n,), generator=gen)}
            produced += n

    def run_forward(self, batch):
        """跑一次 forward,你的签名。激活由 forward hook 自动捕获,无需返回。"""
        self._model(batch["x"], batch["t"])
```

构造:`adapter = MyAdapter(model, device="cuda")`。模型先 `.eval()` 并放到目标 device
(`device` 参数缺省从 `model.device` 推断)。

### 确定性契约(最重要的规则)

`calib_batches(n_samples, batch_size)` 每次**调用**必须产出**相同**的 batch 序列。
为什么:校准分两遍(Pass 1 定 range,Pass 2 填直方图),两遍要跑完全相同的数据;
多样本敏感度也会重放这个流。怎么做:随机数据在**方法内部**新建 seeded
`torch.Generator`(照抄 `DiTAdapter.calib_batches`)——每次调用从头 seed,既确定,
又不会因某次调用消耗了随机数而让后续调用错位。

## 第三步:自验证(接好必跑)

```python
from weiacviz.loading.adapters.diagnose import verify_adapter

print(verify_adapter(adapter))
```

逐项体检并输出 PASS/FAIL 报告:

| 检查项 | 内容 |
|---|---|
| `enumerate_modules` | family、Linear 模块数、attention/MLP 分类、是否降级 |
| `calib_batches` | 流可产出,且**两遍调用逐 batch 比对一致**(确定性) |
| `input_capture` | 一次 forward 后 hook 捕获到多少个 module 的输入 |
| `output_diff` | 抽样跑量化误差,确认敏感性可用(NaN = 该层降级,见「坑」) |

全 PASS 再接入管道/UI。也可以 `python -m weiacviz.loading.adapters.template`
直接跑演示。

## 适配 checklist

- [ ] 模型已 `.eval()`、已在目标 device
- [ ] 目标层是 `nn.Linear`(主线只处理 2D 权重;Conv2d 等不适用)
- [ ] `calib_batches` 确定性(方法内 seeded Generator)
- [ ] batch 结构与 `run_forward` 的消费方式一致
- [ ] `verify_adapter` 全 PASS(至少 calib_batches + input_capture 两项)

## 适配完自动获得的能力

权重分布(per-tensor/channel/group)、激活捕获与在线聚合、per-token 分析、
fake quant(W4/W8 × 粒度 × 对称性)、单样本 + 多校准样本敏感度、量化建议报告。
基类的 `enumerate_modules` / `sample_inputs` / `sample_inputs_batched` /
`_sample_batch`(想换固定样本时 override)默认实现通常够用,无需动。

## 常见坑

1. **模块命名不在 `_ARCH_PATTERNS`**(只登记了 llama/qwen/mistral/deepseek/gpt2/bert/dit
   族的常见后缀)-> 降级为 OTHER、枚举全部 Linear。**功能完整可用**,只是 kind 分类为空;
   想要分类就往 `module_resolver.py` 加一族(纯数据)。
2. **层号正则**只认 `layers.N.` / `blocks.N.` / `h.N.` / `layer.N.`;其他命名(如
   `encoder.layers.0.`)提取不到层号,只影响热力图按层分组展示。
3. **多样本敏感度的 `max_tokens` 截断假设序列轴在倒数第 2 维**。语言模型/DiT 的
   `(B, S, H)` 成立;vision 的 `(B, C, H, W)` 4D 激活截的是空间 H 维——内存照样受控,
   但要知道截的不是 token。
4. **`output_diff` 要求目标层可脱离模型独立前向**(纯 `nn.Linear` 成立)。残差绑定 /
   融合算子 / 依赖外部 buffer 的层会降级为 NaN(不崩);权重分布、激活捕获、fake quant
   仍可用。
5. **校准数据想要真实分布**:随机数据(如 DiT 的随机 latent)能跑通全流程,但敏感度排序
   反映的是随机输入下的误差;有真实校准数据(真实 latent / 真实图像 batch)尽量用真实的。

## 接入 UI / 管道

UI 目前内置「HF CausalLM」与「DiT 演示」两个入口;自定义 adapter 走 Python API:

```python
from weiacviz.viz.app import App

app = App()
app._adapter = MyAdapter(my_model, device="cuda")
app._model = app._adapter.model
app._resolve_result = app._adapter.enumerate_modules()
app._modules = app._resolve_result.modules
# 之后正常调用 app.run_calib / app.run_sensitivity / app.build_report
```

(UI 侧自定义 adapter 注册入口是后续工作;`tests/test_integration.py` 里的
`test_app_run_sensitivity_multi_sample_averages` 有可运行的注入示例。)

## 参考实现

- `HFCausalLMAdapter`(`loading/adapter.py`):HF causal LM,tokenize -> `model(input_ids)`。
- `DiTAdapter`(`loading/adapter.py`):DiT,seeded 随机 latent+timestep+label ->
  `model(x, t, y)`。
- `dit_demo.py`:mini-DiT 演示模型 + `build_demo_dit_adapter()`(UI「DiT 演示」入口)。
- `template.py`:可复制模板,`python -m` 可跑自验证演示。
- `diagnose.py`:`verify_adapter` 体检工具。
