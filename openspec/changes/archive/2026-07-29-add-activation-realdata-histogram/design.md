## Context

当前 `loading/runner.py` 的 `RunningStats` 仅在线聚合标量(count/min/max/sum/sumsq/abs_max,`runner.py:21-26`)。`add-quant-analysis-core` 的 `design.md` Decision 4 本承诺"在线聚合直方图",但未实现。激活 hook 已非侵入捕获 input/output 并 `detach().to("cpu")`(`hook.py:37-49`),每批 `clear()`。校准数据分批推理已就绪(`runner.py:99-113`)。`OnlineAggregator` 按 `module × role` 组织,但结果在 `app.run_calib` 存入 `self._aggregator`(`app.py:79`)后无 UI 消费,「校准与激活」tab 是空壳(`app.py:153-159`)。

约束:面向 7B+ 大模型,内存必须与批数/序列长/模型规模解耦(不能保留原始激活)。校准为离线分析,可接受两遍推理的一次性成本。

## Goals / Non-Goals

**Goals:**
- 在真实大模型校准数据上在线聚合全局激活直方图,内存与模型规模解耦。
- 闭合「校准与激活」tab,使激活分布可见。
- 向后兼容:`run_calibration` 新参数均有默认值,现有调用与测试不受影响。

**Non-Goals:**
- per-channel 在线矩(per-channel kurtosis/skewness)--下一档。
- 代表样本保留与激活量化前后对比--下一档。
- 激活量化(`quantize_activations` 接入 `fake_quant`)--下一档。
- `distance.py` 对聚合直方图的 KL/Wasserstein 改造--随激活量化档。
- 结果落盘缓存--独立工作(见 Open Questions)。

## Decisions

### D1: 两遍校准定 range(核心)
在线直方图要求 range 事先固定,而准确 range 需先看完全部数据。
- Pass 1:现有 `RunningStats`,产出每 `module/role` 全局 `min/max`。
- Pass 2:用 Pass 1 的 `[min, max]` 作固定 range,跨批在线分桶累加。
- **备选 A**:单遍 + 对称 `[-abs_max, abs_max]`--否决,极端离群值撑开主体,且单遍内 amax 实时变化,已分桶 count 无法回填。
- **备选 B**:单遍 + t-digest--否决,实现复杂、引入依赖,过度设计。
- **备选 C**:单遍 + 保守预设 range--否决,跨模型激活幅值差异大,难定统一上界。
- **理由**:两遍最简单可靠,range 准确使直方图与后续 KL 有意义;校准离线、可缓存,一次性成本可接受。

### D2: range 退化与超界处理
- `min == max`:`hi = lo + 1.0`(与 `histogram.py:26-29` 一致),不抛异常。
- 模型须 `eval()` 模式,保证 Pass 2 值域不超 Pass 1;超界值由 `np.histogram` 默认丢弃(eval 下可忽略,文档注明)。

### D3: `RunningHistogram` 聚合器
与 `RunningStats` 平级,放 `loading/runner.py`。
- `__init__(value_range, num_bins=256)`:预算 `bin_edges`。
- `update(t)`:`np.histogram` 累加到 `counts`(int64 数组)。
- `to_result() -> HistogramResult`。
- 内存 `bins × O(1)`。复用 `stats._util.to_numpy`(已处理 bf16/fp16 -> fp32)。

### D4: `OnlineAggregator` 扩展
加 `histograms: Dict[path][role] -> Optional[RunningHistogram]`,Pass 2 后填充,未启用为 `None`。`to_dict()` 含直方图(`counts` 为 list,可 JSON 序列化,为后续缓存铺路)。

### D5: `run_calibration` 两遍编排
- 抽 `_run_pass(model, tok, texts, paths, config, progress_cb, seq_length, done_offset, total)` 复用单遍推理循环(现 `runner.py:99-113`)。
- 新增参数 `collect_histogram=False, num_bins=256`。
- Pass 1 完成后,从 `aggregator.stats` 取每 `path/role` 的 `[min, max]` 构造 `RunningHistogram`;Pass 2 重挂 hook、再跑一遍、仅 `histogram.update(capture)`。
- `update_from_capture` 分离 `update_stats`/`update_histogram`:Pass 1 仅更新标量统计,Pass 2 仅更新直方图--避免标量统计被两遍重复累加导致 `count` 翻倍。
- 进度:`total = n_batches * (2 if collect_histogram else 1)`,复用现有 `ProgressReporter`(frac 自然反映两遍)。
- **顺带修 bug**:`app.run_calib` 当前 `config=None`(`app.py:81`),`batch_size` 滑块无效--改为构造 `CaptureConfig(max_samples, batch_size)`。

### D6: UI 渲染
- `viz/charts.py` 新增 `render_histogram_result(h: HistogramResult, name)`,**不改** `distribution_histogram` 签名(避免破坏 `app.py:68` 等现有调用)。
- 「校准与激活」tab 加 module 下拉(复用 `module_choices()`)+「采集直方图」checkbox + bins 滑块;运行校准后选 module 渲染其 output 全局激活直方图。

### D7: 配置
`Settings`(`shared/config.py`)加 `calibration_collect_histogram: bool = True`、`calibration_histogram_bins: int = 256`。

## Risks / Trade-offs

- [两遍推理慢(大模型)] -> 一次性成本,可接受;后续可按指纹落盘缓存(`shared/cache.py` 已就绪)。
- [range 被 Pass 1 极端离群值撑开,主体压扁] -> 这正是要观察的现象,可接受;下一档 per-channel 可缓解。
- [Pass 2 超界] -> `eval()` + `np.histogram` 丢弃,影响可忽略。
- [`to_dict` 含直方图增大序列化体积] -> bins 默认 256、count int64,每 `module×role` ~2KB,可接受。

## Migration Plan

纯增量,无存量迁移。`run_calibration` 新参数均有默认值,现有调用与测试不受影响。回滚即 revert 本 change,无数据/配置格式变更。

## Open Questions

- 第一版是否接入 `ResultCache` 落盘?两遍更慢,缓存价值大,但 `runner`/`app` 当前均未接缓存,接入是独立工作。**建议第一版不做**,直方图链路闭合后单独加。
