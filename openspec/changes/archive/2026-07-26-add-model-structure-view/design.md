## Context

当前 `App.load()` 加载模型后只返回一行状态字符串（架构族 / 降级 / module 数）。`resolve_modules` 已返回 `ModuleInfo`（path / kind / shape / dtype），但 UI 未展示这些信息。用户加载后看不到模型结构与各层 shape，定位要分析的层只能在下拉框里盲目选。本次在加载完成后展示模型结构概览与逐层 module 表格。

## Goals / Non-Goals

**Goals:**
- 加载模型后展示**模型概览**：架构族、降级状态、全模型总参数量、transformer 层数、目标 module 数。
- 展示**逐层 module 表格**：path / kind / shape / dtype / 参数量，按 layer index 排序。
- 复用现有 `ModuleInfo` 与 `resolve_modules`，不改模型加载核心逻辑。

**Non-Goals:**
- 不做「点击表格行跳转到该 module 分布查看」的交互（留后续增强）。
- 不展示非目标 module（只展示 resolve_modules 返回的 attn/mlp 目标）。
- 不修改 model loading / module_resolver 的核心逻辑。

## Decisions

### 1. 展示形式：gr.Dataframe 表格 + gr.Markdown 概览
- **备选**：gr.HTML 渲染树形结构、gr.Json、gr.Dataframe。
- **理由**：Dataframe 可排序、清晰、Gradio 原生；Markdown 概览简洁。树形 HTML 美观但实现复杂且不可排序。

### 2. 参数量计算
- `params = out_features * in_features + (has_bias ? out_features : 0)`，从 `module.weight.shape` 与 `module.bias is not None` 判断。
- 概览的「总参数量」用 `sum(p.numel() for p in model.parameters())` 统计全模型；表格的参数量是单个 module 的 weight+bias。纯算术，无新依赖。

### 3. Layer index 提取与排序
- 从 module path 用正则 `layers\.(\d+)\.` 解析 layer index。
- 无法解析的（如 embed）标 `-1`，排在表格最前。
- 表格按 layer index 升序，同层按 attn -> mlp 顺序。

### 4. 表格位置
- 放在「模型加载」tab 内，加载按钮与状态文字下方。
- `load()` 返回 (状态文本, 概览 markdown, 表格 dataframe)；加载完成后自动填充。

## Risks / Trade-offs

- [大模型表格过长] -> 7B 有几百个目标 module；Dataframe 可滚动且可排序，暂不分页，后续可加。
- [layer index 解析对非标准命名失效] -> 无 layer index 的标 -1 排在前，不报错。
- [参数量统计范围] -> 概览展示全模型总参数量（直观），表格展示各目标 module 参数量（不含 embedding/norm 等），两者口径不同需在概览文字说明。

## Migration Plan

增量 UI 改动，无迁移。「模型加载」tab 新增概览与表格组件，不影响其他 tab。

## Open Questions

- 是否需要点击表格行跳转到该 module 的权重分布查看？（第一版不做，留后续）
- 概览「总参数量」口径：决定展示全模型总参数量，并在文字标注「目标 module 参数量」单独一行。
