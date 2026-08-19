# WeiActVisualize

大模型**权重与激活数值分布可视化**工具 —— 量化前的"体检"，判断模型 / 各层是否适合量化、用哪种方案量化。

## 功能

- **模型加载与张量捕获**：内置 HF causal LM（Llama/Qwen/Mistral/DeepSeek）与 DiT；通过 `ModelAdapter` 接口可接入任意 `nn.Linear` 模型，`device_map=auto` 分块加载支持大模型，forward hook 非侵入式捕获激活
- **在线聚合**：激活逐批聚合为 running statistics，不保留原始张量，内存与模型规模解耦
- **分布统计**：权重/激活 per-tensor/channel/group 统计、直方图、离群值检测（百分位 / Z-score）、分布距离（KL / Wasserstein）
- **量化模拟**：fake quantization（W4/W8、per-tensor/channel/group、对称/非对称），MSE / cosine 误差度量，层级敏感性分析（输出误差可在多个校准样本上平均，避免单样本误排层序）
- **交互式可视化**：Gradio 界面，直方图 / 热力图 / 量化前后对比，按层 / 模块 / bit 筛选，推理队列 + 进度条
- **报告与导出**：量化建议报告（Markdown）+ 数据（CSV/JSON）+ 图表（PNG）

## 安装

推荐用 conda 隔离环境：

```bash
conda create -n weiacviz python=3.11 -y
conda activate weiacviz
pip install -e .[dev]
```

> 国内网络：conda 可用清华源（`-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main`），pip 用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 启动

```bash
weiacviz          # 启动 Gradio 界面（控制台脚本）
# 或
python -m weiacviz.viz.app
```

## 配置

通过环境变量（前缀 `WEIACVIZ_`）或 `.env` 文件：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WEIACVIZ_MODEL_NAME_OR_PATH` | `Qwen/Qwen2-0.5B` | 模型 ID 或本地路径 |
| `WEIACVIZ_DTYPE` | `fp16` | fp32 / fp16 / bf16 |
| `WEIACVIZ_DEVICE` | `auto` | cpu / cuda / auto（分块加载） |
| `WEIACVIZ_CALIBRATION_DATASET` | `wikitext2` | 校准数据集 |
| `WEIACVIZ_CALIBRATION_SAMPLES` | `128` | 校准样本数 |
| `WEIACVIZ_CALIBRATION_BATCH_SIZE` | `8` | 校准批大小 |
| `WEIACVIZ_CACHE_DIR` | `.weiacviz_cache` | 统计结果缓存目录 |
| `WEIACVIZ_DEFAULT_OUTLIER_PERCENTILE` | `99.9` | 离群值百分位阈值 |
| `WEIACVIZ_DEFAULT_OUTLIER_ZSCORE` | `3.0` | 离群值 Z-score 阈值 |

## 自定义模型接入

量化主线针对 `nn.Linear` 层。内置 adapter：

- `HFCausalLMAdapter`：HF causal LM（默认），tokenize 文本 -> `model(input_ids)`
- `DiTAdapter`：Diffusion Transformer，随机 latent+timestep -> `model(x, t, y)`

接入其他模型（vision / 多模态 / 自定义 `nn.Module`），复制 `adapters/template.py`
实现两个方法即可，Linear 主线自动复用；适配完用 `verify_adapter` 一键体检
（模块枚举 / 校准流确定性 / 激活捕获 / 敏感度可用性）：

```python
from weiacviz.loading.adapter import ModelAdapter
from weiacviz.loading.adapters.diagnose import verify_adapter

class MyAdapter(ModelAdapter):
    def calib_batches(self, n_samples, batch_size):
        for ...:
            yield batch  # 确定性：相同参数产出相同序列（两遍校准依赖）

    def run_forward(self, batch):
        self._model(batch["x"], batch["t"], ...)  # 你的 forward 签名

print(verify_adapter(MyAdapter(my_model)))  # 全 PASS 再接入
```

UI 模型加载 tab 选「DiT 演示」可在界面跑通 DiT 全流程；真实 DiT 用 Python API（`DiTAdapter`）。
完整适配指南（决策树 / 确定性契约 / checklist / 常见坑）见
`src/weiacviz/loading/adapters/README.md`。限制：`output_diff` 敏感性要求目标层可独立前向
（纯 Linear 成立）；非 Linear 层降级，权重/激活/fake quant 仍可用。

## 典型工作流

1. **加载模型**：CPU 验证用小模型（Qwen2-0.5B），大模型选 `device=auto`
2. **权重分布**：观察各层权重分布与离群情况，定位异常层
3. **校准与激活**：跑校准推理捕获激活统计（在线聚合，大模型不 OOM）
4. **量化模拟**：对比量化前后分布与误差，多方案对比
5. **敏感性分析**：量化输出误差按层排序；「样本数」滑块（1–32，默认 8）控制在校准数据上平均的批次数——单样本会受个别难样本影响，多样本平均排序更稳
6. **报告**：基于敏感性分析生成量化建议（哪些层适合 W4、哪些需保 W8）

## 测试

```bash
pytest
```

大模型冒烟测试（手动，需真实模型 + GPU）：

```bash
WEIACVIZ_SMOKE_MODEL=<model_id> pytest tests/test_smoke_large.py
```

## 架构与决策

详见 `openspec/changes/add-quant-analysis-core/design.md`。

核心设计：
- **在线聚合**使内存与模型规模解耦（小模型开发 → 大模型可用）
- **fake quantization** 隔离 bit/粒度/对称性变量，便于方案对比
- **Gradio 队列 + 进度回调**解决大模型推理阻塞 UI

## 状态

第一版（add-quant-analysis-core）覆盖：分布可视化 + 量化模拟 + 敏感性分析 + Gradio 界面 + 报告导出。
自定义模型接入（`ModelAdapter` 抽象，内置 DiT 适配）已支持：任意 `nn.Linear` 模型实现两个方法即可接入。
后续迭代可加：SmoothQuant 离群值迁移、完整量化算法、多卡并行、HF Spaces 部署。
