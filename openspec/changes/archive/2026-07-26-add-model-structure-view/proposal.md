## Why

当前加载模型后，UI 只返回一行状态文字（架构族 / 是否降级 / module 数），用户看不到模型的具体结构、各层 module 的 shape 和参数量。要分析某层时只能在 module 下拉框里盲目选择，层数多的大模型里难以定位目标层。加载后直接展示模型结构概览与逐层 module 表格（含 shape / dtype / 参数量），能让用户快速了解模型构成并定位要分析的层。

## What Changes

- 加载模型后展示**模型概览**：架构族、是否降级、总参数量、transformer 层数、目标 module 数
- 加载模型后展示**逐层 module 表格**：每个目标 module 的 path、kind（attn/mlp/other）、权重 shape、dtype、参数量
- 表格按 transformer 层（layer index）组织 / 排序，便于逐层浏览
- 在 Gradio 界面新增「模型结构」展示区（加载模型 tab 内，加载完成后自动填充）

## Capabilities

### New Capabilities
<!-- 无新能力，本次为现有 interactive-visualization 的需求扩展 -->

### Modified Capabilities
- `interactive-visualization`: 新增「模型结构视图」需求 -- 加载模型后展示概览（架构族 / 降级状态 / 总参数量 / 层数 / module 数）与逐层 module 表格（path / kind / shape / dtype / 参数量），按层组织

## Impact

- **viz/app.py**：`App.load()` 在加载完成后返回结构信息；新增模型结构展示组件（概览文本 + module 表格）
- **viz/structure.py**（新增）：从 `resolve_modules` 的 `ModuleInfo` 列表构建表格数据，计算每个 module 的参数量（含偏置），提取 layer index 排序
- 复用现有 `loading/module_resolver` 的 `ModuleInfo`（已有 path / kind / shape / dtype），不改动模型加载逻辑
- 无新外部依赖
