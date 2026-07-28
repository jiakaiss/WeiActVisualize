## Why

“模型加载” tab 当前把逐层 module 表格放在前、完整网络结构（print model）放在后，用户想先看全局网络结构、再看逐层细节，由全局到局部的浏览顺序更自然。“权重分布” tab 当前固定按 per-channel 计算，无法查看 per-group 分布；而 per-group 切片数量庞大（可达数十万），若与 per-channel 同时计算会非常耗时。需要增加粒度切换，只计算用户选定的那一个粒度，避免无谓开销。

## What Changes

- **模型加载显示顺序**：“模型加载” tab 调整组件顺序，完整网络结构（`print(model)` 文本）展示在逐层 module 表格**之前**；架构概览仍置于最前。
- **权重分布粒度可选**：“权重分布” tab 新增粒度下拉（per-channel / per-group）与 group_size 输入（仅 per-group 时生效）。
- **按需计算单一粒度**：`view_weight` 按用户选定的粒度计算统计与图表，per-channel 与 per-group **不同时计算**，只算选定的一个，避免 per-group 全量计算带来的耗时。
- **per-group 展示复用现有能力**：per-group 下沿用整张量直方图 + per-group 形态指标热力图 + per-group violin（top-k 采样），与 per-channel 视图结构一致。

## Capabilities

### New Capabilities
<!-- 无新 capability -->

### Modified Capabilities
- `interactive-visualization`: 调整模型加载 tab 的显示顺序（完整网络结构置于逐层表格之前）；扩展分布图表渲染，支持 per-channel / per-group 粒度切换且仅计算选定粒度。

## Impact

- `src/weiacviz/viz/app.py`：“模型加载” tab 组件顺序调整；“权重分布” tab 新增粒度下拉与 group_size 控件；`view_weight` 签名增加 granularity / group_size 参数并按选定粒度计算。
- `src/weiacviz/viz/charts.py`：无需大改（形态热图 / violin 已基于 `StatResult` 列表，粒度无关）。
- `tests/test_stats.py`：更新 `view_weight` 测试以适配新签名，新增粒度切换测试。
- 无新增运行时依赖。
