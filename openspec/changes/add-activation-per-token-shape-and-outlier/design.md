## Context

激活侧的可视化深度落后于权重侧。权重侧 (`app.view_weight`, `app.py:70-88`) 已有完整链路：整张量直方图 + per-channel 形态热力图 (`channel_stats_heatmap`) + top-k by kurtosis 的 violin (`channel_violin`) + 术语说明。激活侧 (`view_activation`, `app.py:124-175`) 只有：全局柱状直方图 + per-token abs_max 柱状直方图 + per-channel abs_mean bar + 一行文字 advice。

数据层已就绪但未被充分利用：
- `RunningTokenStats` (`runner.py:98-221`) 单遍聚合 per-token abs_max 的 raw moments（mean/std/cv/kurtosis/skewness）+ 两遍可选直方图，O(1) 内存。
- `activation_stats_per_token` (`activation_stats.py:20-36`) 能对原始激活算 per-token StatResult，但**只有 min/max/mean/std/百分位，缺形态指标**，且**未被 UI 调用**。
- `capture_sample_inputs` (`runner.py:471-501`) 已有「按需单次前向 + hook 捕获」的范式，可复用。
- 权重侧的 `channel_violin` / `channel_stats_heatmap` (`charts.py:139-221`) 接受任意 `StatResult` 列表，与角色无关，可直接复用于 per-token 切片。

约束：面向 7B+ 大模型，新增可视化不得破坏在线聚合的内存解耦特性；per-token 切片需要原始 hidden 向量，必须用有界样本而非全量保留。

## Goals / Non-Goals

**Goals:**
- per-token 切片形态视图：复用 `channel_violin` / `channel_stats_heatmap`，按需捕获单 module 一次激活样本，渲染 top-k by kurtosis violin + per-token 形态热力图。
- per-token abs_max 分布升级为 violin（KDE + IQR + 中位数），并叠加离群 token 标记。
- per-token 离群幅度量化：比例、severity（max/median）、最大幅值。
- 扩展 `activation_stats_per_token` 补齐 kurtosis/skewness/tail_ratio/shape_label。
- 清理 UI 冗余 / 过时说明文本。

**Non-Goals:**
- per-channel（hidden 维）激活形态在线矩（保持现状 SmoothQuant bar 诊断）。
- 激活量化前后对比、`distance.py` 改造、结果落盘缓存。
- input/output 对齐（当前 `view_activation` 看 output；W8A8 实际量化 input。本变更不改这一点，留作后续）。

## Decisions

### D1: per-token 切片视图的数据源 = 按需单 module 样本捕获（非校准期保留）
- **决策**：用户点「查看激活分布」时，对选中 module 跑一次前向、hook 捕获其 output 激活，计算 per-token 形态指标后释放原始张量。
- **备选 A**：校准期为所有 module 保留 per-token hidden 向量 -- 否决，破坏 O(1) 内存（7B × hidden × tokens 不可接受）。
- **备选 B**：校准期保留 top-k token 向量 -- 否决，"哪些 k" 依赖完整分布且增加 aggregator 复杂度。
- **理由**：复用 `capture_sample_inputs` 范式，有界内存（单条样本 × 有界 seq_length），仅对查看的 module 触发，不污染校准 aggregator。

### D2: 按需样本捕获 = output 激活，单条样本 + 有界 seq_length
- **决策**：新增 `capture_module_output_sample(model, tokenizer, module_path, text, seq_length=512)`，返回该 module 的 output 激活张量。单条文本、seq_length 上界 512（≈512 token，足够 top-64 violin 采样），内存 ≈ 512 × hidden × 2B（7B 下约 4MB）。
- **理由**：与 `view_activation` 现有 output 语义一致；有界；top-k violin 只需数十 token。
- **取舍**：W8A8 量化的是 input，此处看 output（见 Non-Goals / Open Questions）。

### D3: per-token 切片形态指标 = 扩展 `activation_stats_per_token`
- **决策**：在 `activation_stats_per_token` (`activation_stats.py:20`) 中，对每个 token 的 hidden 行调用 `shape.excess_kurtosis` / `skewness` / `robust_tail_ratio` / `shape_label`，填入 `StatResult` 的 kurtosis/skewness/tail_ratio/shape_label 字段。
- **理由**：复用 `shape.py` 既有实现；产出 `StatResult` 列表后可直接喂给 `channel_violin` / `channel_stats_heatmap`（它们本就接受 `StatResult` 列表，与角色无关）。
- **备选**：新写一套 per-token violin 渲染 -- 否决，重复代码。

### D4: per-token abs_max 分布 violin = 由聚合直方图采样重建
- **决策**：`RunningTokenStats.abs_max_hist` 只存 (counts, bin_edges)。渲染 violin 时按 counts 对 bin 中心加权采样（上限 4000 点）喂给 `go.Violin`。
- **备选 A**：保留原始 per-token abs_max 标量 -- 否决，破坏在线 O(1)。
- **备选 B**：用直方图密度曲线代替 violin -- 否决，用户要"类似权重部分"的 violin。
- **理由**：采样 violin 视觉忠实；精确形态指标（kurtosis 等）来自在线 raw moments，独立显示，不依赖采样精度。

### D5: per-token 离群幅度 = 新增 `activation_token_outliers`
- **决策**：在 `activation_stats.py` 新增 `activation_token_outliers(token_stats, method="percentile"|"zscore", ...)`，返回 `{outlier_ratio, severity, max_abs, threshold, method}`。severity = max_abs_max / median_abs_max（类比 `activation_outlier_channels` 的 severity）。百分位由 `_hist_percentile` 从直方图近似，z-score 用在线 mean/std。
- **理由**：全部来自 `RunningTokenStats` 已有数据，无需额外捕获；与 `activation_outlier_channels` 形成对称 API。
- **标记**：离群 token 在 D4 的 violin 尾部以红色标记点叠加。

### D6: `view_activation` 返回结构扩展
- **决策**：返回 `(advice, global_fig, token_absmax_violin, slice_violin, slice_heatmap, channel_fig)`。advice 精简为形态标签 + 关键指标 + 离群幅度一行；global_fig 保留柱状直方图；token_absmax_violin 替代原 per-token 柱状图；slice_violin/slice_heatmap 为新增；channel_fig 保留 per-channel bar 诊断。
- **理由**：UI 内部契约，非公开 API；gradio outputs 列表同步更新。

### D7: 清理范围
- **决策**：
  - 精简「校准与激活」tab 说明段 (`app.py:415-419`) 为一行。
  - 精简「敏感性总览」tab markdown (`app.py:468-475`)：保留列含义，删冗长算法提示。
  - 精简「报告」tab markdown (`app.py:494-501`) 同理。
  - 修正 `run_sensitivity` docstring (`app.py:311-318`)：移除引用未接入的 `act_cv`，对齐实际列。
  - 保留「权重分布」tab 的术语说明 accordion（有助解读）。
- **理由**：用户要求"去掉没必要的说明"；仅删冗余 / 过时，保留有用 glossary。

### D8: per-token 切片视图可用时机 = 模型加载后即可（无需校准）
- **决策**：切片视图依赖按需样本捕获（D1），不依赖校准聚合；故模型加载后即可用。per-token abs_max violin + 离群幅度依赖校准 `RunningTokenStats`，未校准时显示提示。
- **理由**：最大化可用性；切片形态不要求跨批聚合。

## Risks / Trade-offs

- [每次「查看激活分布」多一次前向] -> 单次前向、有界 seq_length，可接受；后续可按 module 缓存样本避免重算（Open Question）。
- [采样 violin 与真实分布有偏差] -> 形态指标来自精确在线矩，独立显示；violin 仅作视觉，可接受。
- [per-token 切片 kurtosis 在短 hidden 上噪声大] -> LLM hidden 通常 ≥ 数百（7B 为 4096），kurtosis 稳定；小模型仍可用。
- [清理文本误删有用信息] -> 仅删冗长 / 过时段，保留术语 glossary；低风险。
- [`view_activation` 返回签名变更] -> UI 内部，同步 gradio outputs 列表即可；无外部 API 消费者。

## Migration Plan

纯增量。新增 `capture_module_output_sample`、`activation_token_outliers`、扩展 `activation_stats_per_token`、`view_activation` 返回结构与 gradio outputs。校准流程与在线聚合不变。回滚 = revert 本 change，无数据 / 配置格式变更。

## Open Questions

- **按需样本捕获 input 还是 output？** 当前选 output（与 `view_activation` 一致）。W8A8 量化 input，若后续要对齐，可加 role 参数。本变更不处理。
- **按需样本是否按 module 缓存？** 第一版不缓存（每次重算一次前向，简单）。若实测卡顿，按 `(module_path, sample_text)` 缓存。
- **per-token abs_max violin 是否保留原柱状图？** 第一版替换为 violin（global 激活直方图仍是柱状）。若需并列可后续加。
