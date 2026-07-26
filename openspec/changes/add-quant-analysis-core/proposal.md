## Why

大模型量化（W8A8 / W4A16 / W4A8 等）能显著降低显存占用与推理成本，但量化精度高度依赖权重和激活的数值分布特征 —— 激活离群值、权重范围、敏感层与通道决定了哪些部分适合低比特量化、哪些必须保留高精度。当前缺少一个轻量的"量化前体检"工具，让算法工程师在动手量化前快速观察分布、模拟量化误差、定位敏感层，避免盲目量化导致精度崩塌后反复返工。

## What Changes

- 新增**模型加载与张量捕获**：支持 HuggingFace 模型多架构加载（Llama / Qwen / Mistral / DeepSeek 等），通过 forward hook 捕获中间激活，支持 dtype 控制（fp16/bf16/fp32）与分块加载/分批推理（兼顾小模型开发迭代与大模型可用）
- 新增**校准数据加载**：支持 WikiText2 / C4 等标准数据集与自定义语料，可配置样本数与批次
- 新增**分布统计计算**：权重/激活的 per-tensor / per-channel / per-group 统计量（min/max/mean/std/percentile）、直方图分桶、离群值检测（百分位 / Z-score）、分布距离度量（KL 散度 / Wasserstein）
- 新增**量化模拟**：fake quantization（W8A8 / W4A16 / W4A8 / INT3），per-tensor / per-channel / per-group 与对称/非对称方案对比，误差度量（MSE / cosine similarity / 层输出 diff），敏感性分析（定位高损失层与通道）
- 新增**交互式可视化**：基于 Gradio 的界面，提供直方图、热力图、量化前后对比图，支持按层 / 模块类型 / bit-width 筛选，推理任务队列与进度条（避免长任务阻塞 UI）
- 新增**分析报告**：量化建议报告（哪些层适合低比特、哪些需保高精度）与图表/数据导出
- 新增**工程基础**：Python 包结构（src 布局）、pyproject.toml、配置管理、统计结果缓存、测试骨架

## Capabilities

### New Capabilities
- `model-tensor-capture`: 模型多架构加载、校准数据接入、权重与激活张量的分层捕获，支持 dtype 控制与分块加载/分批推理
- `distribution-statistics`: 权重与激活数值分布的统计计算、直方图分桶、离群值检测、分布距离度量
- `quantization-simulation`: fake quantization 模拟、量化方案对比、误差度量、敏感性分析
- `interactive-visualization`: Gradio 交互式可视化界面，图表渲染、筛选交互、推理队列与进度
- `analysis-report`: 量化适配性建议报告生成与图表/数据导出

### Modified Capabilities
<!-- 新项目，无现有 specs，无需求变更 -->

## Impact

- **新增代码**：新建 Python 包（建议 `src/weiacviz/` 布局），下分 `loading` / `stats` / `quant` / `viz` / `report` 子模块及共享 `config` / `cache`
- **依赖**：torch、transformers、datasets、gradio、plotly、numpy、scipy（见 design.md 详细版本约束）
- **入口**：新增 CLI/Gradio 启动入口
- **数据与缓存**：统计结果落盘缓存以避免重复推理；本地模型与校准数据路径通过配置管理
- **测试**：新增单元测试骨架（统计计算、量化模拟的正确性优先）
- 本变更为项目初始搭建，无破坏性变更（**BREAKING**: 无）
