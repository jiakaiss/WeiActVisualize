## Context

`interactive-visualization` 的“模型加载” tab 当前组件顺序为：状态 → 概览 → 逐层 module 表格 → 完整网络结构（`print(model)`）。用户希望先看全局结构再看逐层细节。“权重分布” tab 的 `view_weight` 固定按 `PER_CHANNEL` 计算统计（`viz/app.py:62`），无法查看 per-group；而 per-group 切片数量巨大（`out * in / group_size`，7B 级可达数十万），若与 per-channel 同时计算会非常耗时。`weight_stats` 与 `histogram_sliced` 已支持 `PER_GROUP`（上个 change 落地），`channel_violin` 目前只按 axis 0 切（per-channel），per-group 需要按 group 切。

## Goals / Non-Goals

**Goals:**
- “模型加载” tab 调整顺序：完整网络结构展示在逐层 module 表格之前（概览仍最前）。
- “权重分布” tab 新增粒度下拉（per-channel / per-group）与 group_size 输入。
- `view_weight` 按选定粒度计算统计与图表，per-channel 与 per-group 不同时计算。
- `channel_violin` 支持 per-group 切片（复用 `slice_weight`），保持 top-k 采样防卡顿。

**Non-Goals:**
- 不改 `distribution-statistics`（per-group 统计/直方图计算能力已具备）。
- 不对 per-group 大切片数做向量化性能优化（本次接受按需触发；大模型耗时风险见 Risks）。
- 不改激活侧、不改 `recommend` / 量化逻辑。
- 不改变 `load` 回调的返回值顺序（仅调 UI 组件定义顺序）。

## Decisions

### D1: 模型加载顺序调整只动 UI 组件定义顺序
`load` 回调返回 `(status, overview, table, str(model))`，`load_btn.click(..., outputs=[load_out, overview_md, structure_df, model_code])` 的 outputs 列表按返回值顺序映射，与 UI 组件定义顺序无关。因此只需在 `build_app` 中把 `model_code`（完整结构）的组件定义移到 `structure_df`（逐层表格）之前，outputs 列表保持不变。零行为风险。

### D2: 权重分布粒度切换控件
“权重分布” tab 新增 `gr.Dropdown(["per-channel", "per-group"], value="per-channel")` 与 `gr.Number(value=128, label="group_size")`（per-group 时生效）。`view_weight` 签名扩展为 `view_weight(module_path, num_bins, granularity, group_size)`，内部把字符串映射为 `Granularity` 枚举传给 `weight_stats`。

### D3: 仅计算选定粒度
`view_weight` 一次调用 `weight_stats(w, path, granularity=选, group_size=选, num_bins=nb)`，得到该粒度的 `StatResult` 列表；`channel_stats_heatmap(stats)` 粒度无关（基于 StatResult），直接复用；`channel_violin` 按同粒度切片。per-channel 与 per-group 绝不同时计算，满足“不用同时展示 per-channel”。

### D4: channel_violin 支持粒度
`channel_violin(weight, granularity=PER_CHANNEL, group_size=None, max_channels=64)` 内部改用 `slice_weight` 切片（替代当前手写 `arr[0..n]`），per-channel 与 per-group 统一路径；切片数超 `max_channels` 时按 |max| top-k 采样。per-group 切片数大时仍只画 top-k 个 violin，不致卡死。

## Risks / Trade-offs

- **per-group 大模型耗时**：`weight_stats` 的 per-group 循环对 7B 级（26 万 group）会非常慢（分钟级），`channel_violin` 切片循环亦然。本次不向量化优化，依赖“用户按需选 per-group 才触发”规避；小模型（0.5B，约 6 千 group）可接受。后续可向量化 `weight_stats` 的 per-group 路径。
- **per-group 形态热图密集**：切片数多时热图列密集，但 plotly Heatmap 可承载不崩，可读性下降属可接受折中。
- **group_size 合法性**：per-group 时 group_size 需 > 0；`view_weight` 对非法值用 `gr.Error` 友好提示。

## Migration Plan

无破坏性变更。`view_weight` 签名扩展（新增可选参数），UI 控件新增。测试更新 `view_weight` 调用签名。回滚即 revert 本 change。

## Open Questions

- per-group 大模型性能是否需要本次向量化 `weight_stats`？（建议否：按需触发已规避，向量化属独立优化）
