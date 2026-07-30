## Why

当前激活处理是半成品:`design.md` 承诺的"在线聚合直方图"未实现(`RunningStats` 只存标量 min/max/sum/sumsq),「校准与激活」tab 无任何可视化输出,`self._aggregator` 存下后从不被 UI 消费。结果是在真实大模型上跑完校准,激活分布完全不可见——而激活离群值恰恰是量化误差的主因(README 自述)。需在 7B+ 大模型上跑真实校准数据并在线聚合出全局激活直方图,内存与模型规模解耦,闭合"激活在 UI 不可见"的缺口。

## What Changes

- 新增 `RunningHistogram` 在线直方图聚合器:固定 range、跨批累加 counts,内存 `bins × O(1)`,与批数/序列长/模型规模解耦。
- `run_calibration` 支持两遍校准:Pass 1 用现有 `RunningStats` 确定每 module/role 的全局 `[min, max]`,Pass 2 以该固定 range 在线分桶。
- `OnlineAggregator` 携带每 module/role 的全局激活直方图(`HistogramResult`)。
- 「校准与激活」tab 增加可视化:运行校准后可选 module 查看其 output 全局激活直方图。
- 修复 `app.run_calib` 的 `batch_size` 滑块无效 bug(当前 `config=None` 导致批次大小参数被忽略)。
- 新增配置项 `calibration_collect_histogram` / `calibration_histogram_bins`。
- 第一版**不做**:per-channel 在线矩、代表样本保留、激活量化(`quantize_activations`)、`distance.py` 改造、结果落盘缓存。

## Capabilities

### New Capabilities

(无——均扩展现有 capability。)

### Modified Capabilities

- `model-tensor-capture`:「激活在线聚合」需求扩展——除标量 running statistics 外,系统 SHALL 支持在线聚合全局激活直方图(固定 range 跨批分桶,不保留原始张量),并 SHALL 支持两遍校准以确定直方图 range(Pass 1 取全局极值,Pass 2 分桶)。
- `interactive-visualization`:新增激活分布可视化——「校准与激活」tab SHALL 在校准完成后展示选定 module 的全局激活直方图。

## Impact

- **代码**:`loading/runner.py`(`RunningHistogram`、`OnlineAggregator.histograms`、`run_calibration` 两遍编排、`_run_pass` 抽取)、`viz/charts.py`(`render_histogram_result`)、`viz/app.py`(校准与激活 tab 可视化、`batch_size` 修复、`collect_histogram`/`num_bins` 透传)、`shared/config.py`(两个新配置项)。
- **测试**:`tests/test_loading.py`(`RunningHistogram` 单测、两遍校准集成测试、range 退化)。
- **依赖**:无新增(numpy 已有)。
- **兼容性**:`run_calibration` 新参数均有默认值,向后兼容;现有 UI 行为不变,仅「校准与激活」tab 增加内容。
