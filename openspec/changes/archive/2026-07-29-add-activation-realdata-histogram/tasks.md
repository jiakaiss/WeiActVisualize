## 1. 激活在线直方图聚合

- [x] 1.1 在 `loading/runner.py` 新增 `RunningHistogram(value_range, num_bins=256)`：预算 `bin_edges`、`update(t)` 用 `np.histogram` 累加到 int64 `counts`、`to_result() -> HistogramResult`；复用 `stats._util.to_numpy`（已处理 bf16/fp16）
- [x] 1.2 `OnlineAggregator` 加 `histograms: Dict[path][role] -> Optional[RunningHistogram]`（初始化为 `None`）；`to_dict()` 含直方图 `counts`/`bin_edges`
- [x] 1.3 抽 `_run_pass(model, tok, texts, paths, config, progress_cb, seq_length, done_offset, total)` 复用现 `runner.py:99-113` 单遍推理循环
- [x] 1.4 `run_calibration` 加参数 `collect_histogram=False, num_bins=256`：Pass 1 完成后从 `aggregator.stats` 取每 `path/role` 的 `[min, max]`（退化时 `hi = lo + 1.0`）构造 `RunningHistogram`；Pass 2 重挂 hook 再跑、`histogram.update(capture)`；进度 `total = n_batches * (2 if collect_histogram else 1)`
- [x] 1.5 确保 `model.eval()`（防御 Pass 2 超界）；超界值由 `np.histogram` 默认丢弃，在 docstring 注明

## 2. 配置

- [x] 2.1 `shared/config.py` 的 `Settings` 加 `calibration_collect_histogram: bool = True`、`calibration_histogram_bins: int = 256`

## 3. 可视化与 UI

- [x] 3.1 `viz/charts.py` 新增 `render_histogram_result(h: HistogramResult, name) -> go.Figure`（不改 `distribution_histogram` 签名，避免破坏 `app.py:68` 等现有调用）
- [x] 3.2 `viz/app.py` 的 `run_calib` 修复 `config=None`（`app.py:81`）：构造 `CaptureConfig(max_samples, batch_size)` 使批次大小滑块生效；透传 `collect_histogram` / `num_bins`
- [x] 3.3 「校准与激活」tab 加 module 下拉（复用 `module_choices()`）+ 刷新按钮、「采集直方图」checkbox、bins 滑块、激活直方图 `gr.Plot` 输出
- [x] 3.4 新增查看激活方法：选 module -> 从 `self._aggregator.histograms[path]["output"]` 渲染；未运行校准或未采集直方图时以 `gr.Error` 提示

## 4. 测试

- [x] 4.1 `tests/test_loading.py`：`RunningHistogram` 单测（固定 range 分桶、跨批累加正确、`to_result` 字段齐全）
- [x] 4.2 `tests/test_loading.py`：两遍校准集成测试（合成 `TinyLlamaLike` + `DummyTok`，`collect_histogram=True` 后 `histograms[p]["output"]` 非空，`sum(counts)` 接近总元素数，允差边界丢弃）
- [x] 4.3 `tests/test_loading.py`：range 退化（常量张量 `min == max`）不抛异常
- [x] 4.4 回归：现有 `test_running_stats_online_aggregation`、`test_online_aggregator_memory_independent_of_batches` 通过

## 5. 验证

- [x] 5.1 `C:/Users/wjk/.conda/envs/weiacviz/python.exe -m pytest tests/ -q` 全绿
- [x] 5.2 `openspec validate add-activation-realdata-histogram --strict` 通过
