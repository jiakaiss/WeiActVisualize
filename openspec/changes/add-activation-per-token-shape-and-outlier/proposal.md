## Why

激活侧的可视化深度明显弱于权重侧。权重有「整张量直方图 + per-channel 形态热力图 + top-k by kurtosis 的 violin + 术语说明」的完整形态分析链路；激活侧只有柱状直方图 + 一行文字 advice + 一张 per-channel abs_mean bar。结果是在激活侧用户无法一眼看出「哪些 token 重尾、离群多严重、per-tensor 量化够不够」，而这恰恰是 W8A8 per-token 激活量化的核心判断。同时 UI 多处存在冗长或过时的说明文本（如敏感性 docstring 提到未实际接入的 `act_cv`）。需把 per-token 激活分析补齐到权重侧的形态可视化深度并补上离群幅度量化，同时清理冗余说明。

## What Changes

- **per-token 切片分布视图（复用权重侧图表）**：按需捕获当前选中 module 的一次激活样本，对每个 token 的 hidden 维分布计算形态指标（kurtosis / skewness / tail_ratio / shape_label），渲染 top-k by kurtosis 的 violin + per-token 形态热力图（复用 `channel_violin` / `channel_stats_heatmap`）。这是「类似权重部分」的逐切片形态可视化在 per-token 粒度的落地。
- **per-token 离群幅度分析**：在已在线聚合的 per-token abs_max 分布上量化离群 token（百分位 / Z-score），报告离群比例、severity（max/median）、最大幅值，并在分布图上标出。
- **per-token abs_max 分布图升级为 violin**：由当前柱状图改为 violin（KDE + IQR + 中位数），直观显示重尾形态；离群 token 在 violin 上标记。
- **扩展 `activation_stats_per_token`**：补齐 kurtosis / skewness / tail_ratio / shape_label（当前只有 min/max/mean/std/百分位）。
- **清理冗余 / 过时说明**：移除敏感性总览、报告、校准与激活 tab 中冗长的 markdown 说明段，修正引用未接入信号（如 `act_cv`）的过时 docstring；保留有助于解读的术语说明。
- **移除 per-channel（hidden 维）激活分析**：per-channel 激活量化无推理引擎支持，作为量化粒度无意义；其唯一价值（离群通道诊断）不足以维持一条独立链路。删除 `RunningChannelStats` / `activation_outlier_channels` / `channel_absmean_bar` / `collect_channel_stats` 采集开关 / 敏感性与报告中的 `act_channel_severity` 列及对应 reason。激活分析聚焦 per-token。
- **不做**：激活量化前后对比；结果落盘缓存。

## Capabilities

### New Capabilities

（无——均扩展现有 capability。）

### Modified Capabilities

- `distribution-statistics`：分布形态指标与离群值检测扩展到 per-token 粒度——per-token 切片 SHALL 计算 kurtosis / skewness / tail_ratio / shape_label；per-token abs_max 分布 SHALL 量化离群 token 的比例、severity 与最大幅值。同时移除 per-channel 激活离群通道检测场景。
- `interactive-visualization`：激活分布可视化升级到权重侧深度——per-token 切片 violin（top-k by kurtosis）+ per-token 形态热力图 + per-token abs_max 分布 violin + 离群幅度标记；移除 per-channel abs_mean 离群通道视图；并 SHALL 移除冗余 / 过时的解释文本。
- `model-tensor-capture`：新增按需单 module 激活样本捕获——为 per-token 切片视图提供一次前向的 output 激活样本，有界内存，仅对当前查看的 module 触发。移除 per-channel 激活在线聚合（`RunningChannelStats` / `collect_channel_stats`）。

## Impact

- **代码**：`stats/activation_stats.py`（`activation_stats_per_token` 补形态指标、新增 per-token 离群幅度函数）、`viz/charts.py`（per-token abs_max 分布 violin、复用 `channel_violin` / `channel_stats_heatmap`）、`viz/app.py`（`view_activation` 重构：per-token 切片视图 + 离群幅度 + 清理说明文本 + 按需样本捕获调用）、`loading/runner.py` 或 `loading/hook.py`（按需单 module 输出样本捕获，参考 `capture_sample_inputs`）。
- **测试**：`tests/test_stats.py`（per-token 形态指标、离群幅度）、`tests/test_loading.py`（按需样本捕获）、`tests/test_integration.py`（激活视图端到端）。
- **依赖**：无新增（plotly / numpy 已有）。
- **兼容性**：纯增量；`view_activation` 返回值结构可能调整（UI 内部，非公开 API）；现有校准流程与在线聚合不变。
