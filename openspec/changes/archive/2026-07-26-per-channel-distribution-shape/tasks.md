## 1. 类型与形态指标基础

- [x] 1.1 在 `shared/types.py` 的 `StatResult` 新增 `kurtosis: float = nan`、`skewness: float = nan`、`tail_ratio: float = nan`、`shape_label: str = ""` 字段（带默认值，保证既有构造点不破）
- [x] 1.2 新建 `stats/shape.py`：实现 `excess_kurtosis(a)`、`skewness(a)`、`robust_tail_ratio(a, lo=0.1, hi=99.9)`，纯 numpy；定义重尾/轻尾/偏态阈值常量（3 / -0.5 / 0.5）
- [x] 1.3 在 `stats/shape.py` 实现 `shape_label(kurtosis, skewness)`：返回“正态/重尾/轻尾”并在偏态超阈值时附注非对称量化提示
- [x] 1.4 零方差切片：`robust_tail_ratio` 返回 `nan`，`excess_kurtosis/skewness` 返回 `nan`，不抛异常

## 2. 直方图粒度

- [x] 2.1 在 `stats/histogram.py` 新增 `histogram_sliced(t, granularity, group_size=None, num_bins=256, value_range=None) -> List[HistogramResult]`，复用 `slice_weight` 对每切片分桶；`histogram()` 签名保持不变
- [x] 2.2 `value_range=None` 时每切片各自 auto；传入统一 range 时跨切片对齐

## 3. 形态指标接入统计

- [x] 3.1 在 `stats/weight_stats.py` 每切片计算 kurtosis / skewness / tail_ratio / shape_label 并填入 `StatResult`
- [x] 3.2 在 `stats/weight_stats.py` 每切片调用 `histogram` 填充 `StatResult.histogram`（当前恒为 None）
- [x] 3.3 保留 `activation_stats.py` 接口对称（如适用），本次仅权重侧填充，不破坏激活侧

## 4. 可视化

- [x] 4.1 在 `viz/charts.py` 新增 `channel_stats_heatmap(stats, metrics=("kurtosis","skewness","tail_ratio","outlier_ratio"))`，渲染 [metric × channel] 矩阵
- [x] 4.2 在 `viz/charts.py` 新增 `channel_violin(weight, max_channels=64)`，通道数超限时按 `outlier_ratio`（fallback `max_abs`）取 top-k
- [x] 4.3 形态标签在热力图/图例中以可区分方式标示重尾通道

## 5. UI 串接

- [x] 5.1 修改 `viz/app.py` 的 `view_weight` 返回 `(整张量直方图, 形态热力图, violin)` 三图
- [x] 5.2 更新“权重分布” tab 的 `gr.Plot` 输出控件与回调绑定，移除原仅展示 mean 的 heatmap

## 6. 测试

- [x] 6.1 `tests/test_stats.py` 新增 shape 单测：正态分布 kurtosis≈0、拉普拉斯≈3、均匀≈-1.2；零方差返回 nan；tail-ratio 数值正确
- [x] 6.2 新增 `histogram_sliced` 单测：per-channel / per-group 切片数与每切片 bin 数正确
- [x] 6.3 新增 `weight_stats` 单测：`StatResult` 含形态字段与非 None 的 `histogram`
- [x] 6.4 新增 charts 单测：`channel_stats_heatmap` / `channel_violin` 返回 `go.Figure`，top-k 采样在通道超限时生效
- [x] 6.5 更新既有 stats/viz 测试以兼容 `StatResult` 新字段
- [x] 6.6 新增 `view_weight` 集成/smoke 测试：返回三个 Figure

## 7. 验证

- [x] 7.1 运行 `pytest` 全量通过
- [x] 7.2 运行 `openspec validate per-channel-distribution-shape --strict` 通过
