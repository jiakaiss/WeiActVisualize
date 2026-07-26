## MODIFIED Requirements

### Requirement: 分布图表渲染
系统 SHALL 使用 Plotly 渲染权重/激活的整张量分布直方图、per-channel/per-layer 统计量热力图、per-channel 形态指标热力图（kurtosis / skewness / tail-ratio / outlier-ratio 并排）与 per-channel 分布小提琴图（violin）。

#### Scenario: 查看某层权重分布
- **WHEN** 用户选择某层的权重分布视图
- **THEN** 界面同时展示该层权重的整张量直方图、per-channel 形态指标热力图与 per-channel 分布 violin

#### Scenario: 查看形态指标热力图
- **WHEN** 用户查看某层权重的 per-channel 形态热力图
- **THEN** 界面并排展示每通道的 excess kurtosis、skewness、tail-ratio 与 outlier-ratio

#### Scenario: 大通道数 violin 采样
- **WHEN** 某层权重输出通道数超过 violin 展示上限（默认 64）
- **THEN** 界面按 outlier-ratio 取 top-k 通道绘制 violin，避免渲染卡顿

#### Scenario: 形态标签可视化
- **WHEN** 某通道被标记为“重尾”
- **THEN** 形态热力图或图例中对该通道的 kurtosis 值以可区分方式标示
