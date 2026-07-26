## Context

`distribution-statistics` 已能按 per-tensor / per-channel / per-group 计算 min/max/mean/std/百分位，`slice_weight` (`loading/weights.py:18`) 也已支持切片。但分布可视化仍停在整张量：`histogram()` (`stats/histogram.py:12`) 内部直接 `flatten()`，没有粒度参数；`StatResult.histogram` 字段 (`shared/types.py:66`) 预留却恒为 `None`；UI 的 `view_weight` (`viz/app.py:49`) 只把 per-channel 的 `mean` 画成单标量热力图，埋没了 range/std/outlier 等更能反映难度的量。同时没有任何分布形态指标--而重尾正是量化难度的核心形态信号（极端值撑大 `scale = amax/qmax`，压垮主体精度）。本设计把分布与形态都下沉到 per-channel / per-group 粒度，与实际量化决策对齐。

## Goals / Non-Goals

**Goals:**
- 直方图支持 per-channel / per-group 切片分桶，每切片独立返回。
- per-channel / per-group 粒度计算 excess kurtosis、skewness、robust tail-ratio，并给出形态标签。
- `StatResult` 真正填充 `histogram` 并携带形态指标。
- UI 同时展示整张量直方图、per-channel 形态热力图、per-channel violin。
- 纯 numpy 实现，不引入新运行时依赖。

**Non-Goals:**
- 不改动 `fake_quant` / `sensitivity` / `recommend` 的量化逻辑（形态指标后续可接入推荐，但不在本次范围）。
- 不做激活侧的形态指标（本次聚焦权重；接口对称设计，激活可后续复用）。
- 不做分布拟合检验（如 KS 检验），仅用经验阈值的形态度量。
- 不改 `recommend.py` 的 symmetry 选择逻辑（偏度仅作提示展示，不自动改 config）。

## Decisions

### D1: `histogram()` 保持签名不变，新增 `histogram_sliced()`
保留 `histogram(t, num_bins, value_range) -> HistogramResult`（向后兼容现有 `distribution_histogram` 等调用）。新增 `histogram_sliced(t, granularity, group_size, num_bins, value_range) -> List[HistogramResult]`，内部复用 `slice_weight` 对每切片分桶。
- **备选**：让 `histogram()` 直接支持粒度并返回列表 -- 否决，会改变返回类型、破坏现有调用点。
- **value_range 语义**：`None` 时每切片各自 auto（看自身形状）；调用方可传统一 range 做跨通道对比。形态指标本身 range 无关，不强制对齐。

### D2: 形态指标用 biased 矩估计，纯 numpy
- `excess_kurtosis = m4/m2² − 3`，`skewness = m3/m2^1.5`，`m_k` 为中心矩。
- `robust_tail_ratio = (p99.9 − p0.1) / (2·std)`，`std==0` 时返回 `nan`。
- **备选**：用 scipy 的 bias-corrected 估计 -- 否决，会引入新依赖且对大张量无显著收益。
- **理由**：biased 估计对小样本有偏，但量化关心的是“有没有重尾”的相对排序，biased 已足够；robust_tail_ratio 作为分位鲁棒兜底，双指标互证。

### D3: 形态标签用经验阈值，可后续配置
- `kurtosis > 3` → “重尾”（拉普拉斯 ≈ 3 为参照）
- `kurtosis < -0.5` → “轻尾”（均匀 ≈ -1.2 为参照）
- 其余 → “正态”
- `|skewness| > 0.5` → 附注“偏态，非对称量化可能更优”
- 阈值在 `shape.py` 内以模块常量定义，便于后续提到 config。

### D4: StatResult 新增字段带默认值
新增 `kurtosis: float = nan`、`skewness: float = nan`、`tail_ratio: float = nan`、`shape_label: str = ""`。默认值保证既有构造点不破。`histogram` 字段在 `weight_stats` 中按切片填充。

### D5: 可视化新增 violin 与多指标形态热力图
- 新增 `channel_stats_heatmap(stats, metrics)` 渲染 [metric × channel] 矩阵，一次并排展示 kurtosis/skewness/tail_ratio/outlier_ratio/max_abs。
- 新增 `channel_violin(weight, max_channels=64)` 渲染 per-channel violin；通道数超过上限时按 `outlier_ratio`（或 `max_abs`）取 top-k，避免大模型（数千通道）卡顿。
- 整张量直方图沿用 `distribution_histogram`。
- **备选**：用叠加直方图代替 violin -- 否决，通道数多时叠加不可读；violin 更紧凑且直接显示尾部。

### D6: UI 在“权重分布” tab 串接
`view_weight` 返回 `(整张量直方图, 形态热力图, violin)` 三图。形态热力图替换原 per-channel mean heatmap。

## Risks / Trade-offs

- **kurtosis 4 阶矩小样本不稳** → 用 robust_tail_ratio 兜底，UI 同时展示两者供互证；文档注明 biased 估计。
- **大模型 per-channel violin 卡顿** → `max_channels` 上限 + top-k 采样；热力图为 1D 数组，plotly 可承载全量通道。
- **StatResult 填充 histogram 增内存** → 默认 256 bins × 数千通道约数百 KB，可接受；`num_bins` 可配，超大模型可调小。
- **形态标签阈值为主观经验值** → 作为模块常量集中定义，便于调参；不自动改量化 config，仅作展示与提示。
- **per-group 直方图对齐** → `slice_weight` 的 per-group 沿展平维度切，与 `fake_quant` 的 per-group（沿 input dim）语义略有差异，文档注明，本次按 `slice_weight` 既有语义实现。

## Migration Plan

无破坏性变更：`histogram()` 签名不变，`StatResult` 新字段有默认值，现有调用与测试不受影响。回滚即 revert 本 change，无数据迁移、无配置格式变更。

## Open Questions

- 形态标签阈值是否需要提升为 `QuantConfig`/settings 可配项？（建议先硬编码，待用户反馈再外露）
- violin 通道上限 64 是否合适？是否应改为按层自适应？（建议默认 64，后续按模型规模调）
