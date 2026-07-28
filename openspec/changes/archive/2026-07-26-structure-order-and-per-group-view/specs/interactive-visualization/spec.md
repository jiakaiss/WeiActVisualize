## MODIFIED Requirements

### Requirement: 完整网络结构展示
系统 SHALL 在加载模型后展示完整的模型结构文本（等价于 print(model)），且该完整结构 SHALL 展示在逐层 module 表格之前，使用户先看到全局网络再浏览逐层细节。

#### Scenario: 展示完整网络结构
- **WHEN** 模型加载完成
- **THEN** 界面展示模型的完整结构文本（包含所有子模块与层级关系）

#### Scenario: 完整结构置于逐层表格之前
- **WHEN** 模型加载完成后展示加载结果
- **THEN** 完整网络结构文本出现在逐层 module 表格之前

### Requirement: 分布图表渲染
系统 SHALL 使用 Plotly 渲染权重/激活的整张量分布直方图、per-channel 或 per-group 形态指标热力图（kurtosis / skewness / tail-ratio / outlier-ratio 并排）与对应粒度的分布小提琴图（violin）。系统 SHALL 支持用户在 per-channel 与 per-group 粒度间切换并指定 group_size，且 SHALL 仅计算当前选定粒度的统计与图表，per-channel 与 per-group 不同时计算。

#### Scenario: 查看 per-channel 权重分布
- **WHEN** 用户选择某层权重分布视图且粒度为 per-channel
- **THEN** 界面展示整张量直方图、per-channel 形态指标热力图与 per-channel 分布 violin

#### Scenario: 查看 per-group 权重分布
- **WHEN** 用户选择粒度为 per-group 并指定 group_size
- **THEN** 界面展示整张量直方图、per-group 形态指标热力图与 per-group 分布 violin，且不计算 per-channel 统计

#### Scenario: 仅计算选定粒度
- **WHEN** 用户在 per-channel 与 per-group 间切换
- **THEN** 系统仅计算当前选定粒度的统计与图表，不同时计算两种粒度

#### Scenario: 查看形态指标热力图
- **WHEN** 用户查看某层权重的形态热力图
- **THEN** 界面并排展示当前选定粒度下每切片的 excess kurtosis、skewness、tail-ratio 与 outlier-ratio

#### Scenario: 大切片数 violin 采样
- **WHEN** 当前粒度下切片数超过 violin 展示上限（默认 64）
- **THEN** 界面按 outlier-ratio 取 top-k 切片绘制 violin，避免渲染卡顿

#### Scenario: 形态标签可视化
- **WHEN** 某切片被标记为“重尾”
- **THEN** 形态热力图或图例中对该切片的 kurtosis 值以可区分方式标示
