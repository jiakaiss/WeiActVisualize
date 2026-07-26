## Why

当前“权重分布”视图把整张权重张量展平成一张 per-tensor 直方图，而 per-channel 统计虽然已计算，UI 却只露出每通道的 `mean` 一个标量。这种整体视角看不出量化难度：少数离群通道会被平均掩盖，而它们正是把 per-tensor 量化 `scale = amax/qmax` 撑大、压垮主体精度的元凶。同时，分布**形态**（重尾 vs 正态）直接决定 scale 利用率——重尾意味着极端值浪费量化级，是量化难度的核心信号，但当前统计只有 min/max/mean/std/百分位，没有任何形态度量。实际量化普遍采用 per-channel / per-group 粒度，因此分布与形态都应在该粒度上呈现，才能与量化决策对齐。

## What Changes

- **直方图支持粒度**：`histogram()` 不再只对整张量展平，支持 per-channel / per-group 切片分桶，返回每切片的直方图。
- **新增分布形态指标**：per-channel / per-group 粒度计算 excess kurtosis（超额峰度）、skewness（偏度）、robust tail-ratio（`(p99.9 − p0.1) / (2·std)`），纯 numpy 实现，不引入新依赖。
- **StatResult 扩展**：填充当前恒为 `None` 的 `histogram` 字段，并新增形态指标字段。
- **形态分类标签**：按 kurtosis 阈值给出可读标签（正态 / 重尾 / 轻尾），偏度非零时提示非对称量化可能更优。
- **可视化升级**：新增 per-channel 分布 violin / 叠加直方图（肉眼看出尾部轻重），以及 per-channel 形态指标热力图（kurtosis / skewness / tail-ratio / outlier-ratio 并排），替换当前只展示 `mean` 的 heatmap。
- **UI 入口**：“权重分布” tab 在选定 module 后同时展示整张量直方图、per-channel 形态热力图与 violin。

## Capabilities

### New Capabilities
<!-- 无新 capability，均为对现有 capability 的需求变更 -->

### Modified Capabilities
- `distribution-statistics`: 新增“分布形态指标”需求（kurtosis/skewness/tail-ratio，per-channel/per-group）；扩展“直方图分桶”需求以支持按粒度切片分桶。
- `interactive-visualization`: 扩展“分布图表渲染”需求，新增 per-channel 分布展示（violin / 叠加直方图）与 per-channel 形态指标热力图。

## Impact

- `src/weiacviz/stats/histogram.py`：`histogram()` 增加 `granularity` / `group_size` 参数。
- `src/weiacviz/stats/shape.py`（新建）：kurtosis / skewness / robust_tail_ratio 计算与形态标签。
- `src/weiacviz/stats/weight_stats.py`：每切片填充 `histogram` 与形态指标到 `StatResult`。
- `src/weiacviz/shared/types.py`：`StatResult` 新增 `kurtosis` / `skewness` / `tail_ratio` / `shape_label` 字段。
- `src/weiacviz/viz/charts.py`：新增 violin 与 per-channel 形态热力图渲染。
- `src/weiacviz/viz/app.py`：`view_weight` 扩展返回形态热力图与 violin。
- `tests/`：新增形态指标与粒度直方图的单测；更新既有 stats/viz 测试。
- 无新增运行时依赖（纯 numpy + 已有 plotly）。
