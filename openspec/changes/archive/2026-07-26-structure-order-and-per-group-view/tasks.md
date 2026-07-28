## 1. 模型加载显示顺序

- [x] 1.1 在 `viz/app.py` 的 `build_app` “模型加载” tab 中，将 `model_code`（`gr.Code` 完整结构）的组件定义移到 `structure_df`（逐层表格）之前；保持 `load_btn.click` 的 `outputs` 列表顺序不变（仍对应 `load` 返回值）

## 2. 权重分布粒度控件

- [x] 2.1 在“权重分布” tab 新增 `gr.Dropdown(["per-channel", "per-group"], value="per-channel", label="粒度")` 与 `gr.Number(value=128, label="group_size")`
- [x] 2.2 更新 `w_btn.click` 的 `inputs` 加入粒度与 group_size 两个控件

## 3. view_weight 按粒度计算

- [x] 3.1 扩展 `view_weight` 签名为 `(module_path, num_bins, granularity, group_size)`，把字符串映射为 `Granularity` 枚举
- [x] 3.2 调 `weight_stats(w, module_path, granularity=选定, group_size=选定, num_bins=nb)`，仅计算选定粒度
- [x] 3.3 per-group 时校验 `group_size > 0`，否则 `raise gr.Error` 给友好提示
- [x] 3.4 调 `channel_violin(w, granularity=选定, group_size=选定, name=...)` 按同粒度画 violin

## 4. channel_violin 支持粒度

- [x] 4.1 在 `viz/charts.py` 的 `channel_violin` 签名加 `granularity=Granularity.PER_CHANNEL, group_size=None`，import `slice_weight` 与 `Granularity`
- [x] 4.2 内部改用 `slice_weight(weight, granularity, group_size)` 切片，替代当前手写 `arr[0..n]`，per-channel / per-group 统一路径
- [x] 4.3 切片数超 `max_channels` 时按 `|max|` top-k 采样（沿用现有逻辑）

## 5. 测试

- [x] 5.1 更新 `tests/test_stats.py::test_view_weight_returns_three_figures` 适配新签名（传 `granularity="per-channel"`）
- [x] 5.2 新增 per-group view_weight 测试：`granularity="per-group", group_size=4`，返回三个 Figure
- [x] 5.3 新增 `channel_violin` per-group 测试：`granularity=Granularity.PER_GROUP, group_size=4`，返回 `go.Figure`
- [x] 5.4 新增 per-group 非法 group_size 测试：`group_size=0` 触发 `gr.Error`

## 6. 验证

- [x] 6.1 运行 `pytest` 全量通过
- [x] 6.2 运行 `openspec validate structure-order-and-per-group-view --strict` 通过
