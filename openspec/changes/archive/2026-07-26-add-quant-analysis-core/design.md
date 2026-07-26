## Context

项目从零开始，目标是构建大模型量化前的"分布体检"工具：加载模型 → 捕获权重/激活张量 → 统计数值分布 → 模拟量化并度量误差 → 可视化 → 生成量化建议。

约束与现状：
- 后端必须为 Python（PyTorch / Transformers 加载模型、forward hook 捕获激活），前端已选定 Gradio。
- 需兼顾小模型（0.5B-1B，CPU 开发迭代）与大模型（7B-70B，GPU），架构须支持分块加载与分批推理。
- 权重张量可离线从模型参数直接读取；激活张量只能在校准数据上在线推理时通过 hook 捕获。
- 激活数据量极大（layer × batch × seq × hidden），无法全量驻留内存，必须在线聚合。
- 统计与量化模拟可能耗时，重复推理成本高，需要结果缓存。

利益相关者：算法工程师（量化决策者）、模型部署工程师（关注精度/成本权衡）。

## Goals / Non-Goals

**Goals:**
- 提供端到端的分析流水线：加载 → 捕获 → 统计 → 量化模拟 → 可视化 → 报告。
- 架构上支持从小模型到大模型的平滑扩展：device_map 分块加载、分批推理、CPU/GPU 自适应、激活在线聚合。
- 模块解耦：`loading` / `stats` / `quant` / `viz` / `report` 可独立复用与测试。
- 长推理任务不阻塞 UI（Gradio 队列 + 进度回调）。
- 统计结果按配置指纹缓存，避免重复推理。
- 量化方案可对比（bit-width / 粒度 / 对称性），输出敏感性排序。

**Non-Goals:**
- 不实现 kernel 级量化加速推理引擎（只做 fake quantization 数值模拟，关注误差估计而非真实加速）。
- 不做模型训练 / 微调 / QAT。
- 不做量化权重的压缩存储（如写入量化后的 safetensors）。
- 第一版不做分布式 / 多卡并行（预留接口，单卡或 CPU offload）。
- 不实现 SmoothQuant / AWQ / GPTQ 等完整量化算法（仅提供离群值与敏感性分析作为算法决策输入；实际算法后续迭代）。
- 第一版不做前后端分离部署（Gradio 单体；但保留内部 API 边界以便后续拆分）。

## Decisions

### 1. 技术栈：Python + PyTorch + Transformers + Gradio + Plotly + NumPy/SciPy
- **备选**：Streamlit（长任务与队列能力弱）、FastAPI + React（工程量大 3-5 倍，第一版不必要）、Plotly Dash（学习曲线与交互复杂度高于 Gradio）。
- **理由**：Gradio 原生队列与进度条直接解决"大模型推理阻塞 UI"的核心痛点；Plotly 图表可直接嵌入；与 HuggingFace 生态天然契合；迭代速度最快。等核心价值验证后再视情况升级到前后端分离。

### 2. 包结构：src 布局 `src/weiacviz/`，按能力域分子模块
```
src/weiacviz/
  loading/    # 模型加载、module 解析、hook 注册、校准数据
  stats/      # 分布统计、直方图、离群值、分布距离
  quant/      # fake quant、方案对比、误差度量、敏感性
  viz/        # Gradio 界面、图表渲染、筛选
  report/     # 建议报告、导出
  shared/     # config、cache、types
```
- **备选**：扁平结构（耦合高）、按技术层分（不利于能力演进）。
- **理由**：src 布局可安装、测试隔离好；按能力域分模块与 spec 一一对应，便于演进与归档。

### 3. 张量捕获：非侵入式 forward hook
- 在目标 module 上注册 forward hook，捕获输入/输出激活；权重通过 `module.named_parameters()` 离线读取。
- **备选**：改写 model.forward（侵入式、难维护）、依赖 nnsight/transformer-lens（额外依赖且架构覆盖受限）。
- **理由**：hook 非侵入、通用、可控；与具体架构解耦，未知架构也能降级工作。

### 4. 大模型支持：分块加载 + 分批推理 + 激活在线聚合
- `device_map="auto"` 分块加载，支持 CPU offload；校准数据分 batch 推理。
- **关键**：激活逐 batch 在线聚合为 running statistics（min/max/sum/sumsq/直方图），不保留原始激活，控制内存。
- **备选**：全量缓存激活（7B 模型即可能 OOM）。
- **理由**：在线聚合使内存与模型规模解耦，是小模型到大模型可扩展的核心。

### 5. 量化模拟：fake quantization（round-then-dequant）
- `q = round(x/scale) → clamp → dq = q * scale`，支持 per-tensor / per-channel / per-group、对称/非对称、W8A8/W4A16/W4A8/INT3。
- **备选**：调用 bitsandbytes 真实量化（依赖重、且测量的是引擎实现而非方案本身）。
- **理由**：纯数值模拟可隔离"量化方案"变量，便于对比；可控、可复现。

### 6. 缓存：统计结果按配置指纹落盘
- 缓存键 = hash(model_id, module_path, stat_type, params, calibration_config)；结果存 parquet/json。
- **理由**：避免重复推理；配置变更自动失效。

### 7. 配置与异步
- 配置：YAML + pydantic-settings dataclass。
- 异步：Gradio `@app.launch(queue=True)` + tqdm/yield 进度回调。

## Risks / Trade-offs

- [激活内存爆炸] → 在线聚合只存统计量；校准样本数设上限且可配。
- [大模型加载慢 / OOM] → device_map 自动分块 + CPU offload 选项；分批推理；默认先在小模型验证。
- [多架构适配差异] → 抽象 module resolver，按架构族（Llama-like / Qwen-like）归类目标 module；未知架构降级为枚举全部 Linear 层。
- [Gradio 复杂交互受限] → 第一版接受；保留 stats/quant 与 viz 之间的纯函数 API 边界，便于后续拆分前后端。
- [fake quant 与真实量化精度偏差] → 文档明确为"误差估计"，关注相对趋势与层间排序而非绝对值。
- [校准数据代表性不足] → 支持自定义数据集；UI 提示样本数与代表性的权衡。

## Migration Plan

初始搭建，无存量迁移。按以下顺序增量交付（与 tasks.md 对应）：
1. 包骨架 + 配置 + 缓存基础设施 + 测试骨架。
2. 模型加载 + module 解析 + hook 捕获 + 校准数据（小模型验证）。
3. 分布统计（权重 + 激活在线聚合）。
4. 量化模拟 + 误差度量 + 敏感性分析。
5. Gradio 可视化界面 + 图表 + 筛选 + 异步队列。
6. 报告生成 + 导出。
7. 端到端集成测试 + 大模型（7B）冒烟验证。

回滚策略：各模块独立、无破坏性变更；任意阶段可停在已交付能力。

## Open Questions

- 是否需要多 GPU 支持？第一版单卡 / CPU offload，预留 `device` 抽象接口。
- 默认校准样本数？建议 128-512，可配；需在精度代表性与推理耗时间权衡。
- 是否对接 HuggingFace Spaces 部署？后续迭代考虑。
- 离群值检测阈值默认值（百分位 99.9% / Z-score 3）需在真实模型上校准。
