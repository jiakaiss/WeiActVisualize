## 1. Stats 层：per-token 形态指标与离群幅度

- [x] 1.1 扩展 `stats/activation_stats.py` 的 `activation_stats_per_token`：对每个 token 的 hidden 行调用 `shape.excess_kurtosis` / `skewness` / `robust_tail_ratio` / `shape_label`，填入 `StatResult` 的 kurtosis/skewness/tail_ratio/shape_label 字段
- [x] 1.2 新增 `activation_token_outliers(token_stats, method="percentile"|"zscore", k/percentile)`：返回 `{outlier_ratio, severity(max/median), max_abs, threshold, method}`；百分位经 `_hist_percentile`，z-score 用在线 mean/std
- [x] 1.3 单测 `tests/test_stats.py`：per-token 形态指标正确性（重尾/正态合成数据）、零方差 token 鲁棒、离群幅度计算（severity/max_abs）

## 2. 捕获层：按需单 module 激活样本

- [x] 2.1 在 `loading/runner.py` 新增 `capture_module_output_sample(model, tokenizer, module_path, text, seq_length=512)`：复用 `ActivationCapture`（仅 output），单条样本，hook 用后卸载，返回 output 激活张量
- [x] 2.2 单测 `tests/test_loading.py`：捕获返回 output 张量、shape 与 seq_length 对应、hook 已卸载、独立 hook 不污染校准 aggregator

## 3. 图表层：per-token abs_max 分布 violin 与切片复用

- [x] 3.1 在 `viz/charts.py` 新增 `render_token_absmax_violin(token_stats, outlier_info=None, max_points=4000)`：由 `abs_max_hist` 按 counts 加权采样 bin 中心重建 `go.Violin`，离群 token 以红色标记点叠加
- [x] 3.2 验证 `channel_violin` / `channel_stats_heatmap` 直接接受 per-token `StatResult` 列表；必要时补 `name` 参数透传，不新增渲染逻辑
- [x] 3.3 单测空数据 / 无直方图（单遍未采集）时的鲁棒占位

## 4. UI 集成：view_activation 重构与文本清理

- [x] 4.1 重构 `viz/app.py` 的 `view_activation`：调用 `capture_module_output_sample` -> `activation_stats_per_token` -> `channel_violin` + `channel_stats_heatmap` 产出 per-token 切片视图
- [x] 4.2 在 `view_activation` 接入 `render_token_absmax_violin` + `activation_token_outliers`（依赖校准 `token_stats`，未采集时占位提示）
- [x] 4.3 调整「校准与激活」tab 的 gradio outputs 列表与布局：全局直方图 + per-token abs_max violin + per-token 切片 violin + per-token 形态热力图 + per-channel bar
- [x] 4.4 精简 `view_activation` 的 advice 文本为：形态标签 + mean/std/kurtosis/skewness + 离群比例/severity 一行
- [x] 4.5 清理冗余/过时说明：`app.py:415-419`（校准与激活）、`app.py:468-475`（敏感性总览）、`app.py:494-501`（报告）markdown 段；修正 `run_sensitivity` docstring (`app.py:311-318`) 移除 `act_cv` 引用并对齐实际列；保留权重分布术语说明

## 5. 端到端验证

- [x] 5.1 集成测试 `tests/test_integration.py`：加载 Qwen2-0.5B -> `view_activation` 返回全部图；未校准时切片视图可用、abs_max violin 占位；校准后 abs_max violin + 离群标记可用
- [x] 5.2 `pytest` 全量通过
- [ ] 5.3 手动启动 Gradio（Qwen2-0.5B, CPU）确认 per-token 切片 violin / 形态热力图 / abs_max violin 离群标记正常渲染
