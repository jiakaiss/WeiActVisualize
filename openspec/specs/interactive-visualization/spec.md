# interactive-visualization Specification

## Purpose
TBD - created by archiving change add-quant-analysis-core. Update Purpose after archive.
## Requirements
### Requirement: Gradio 界面入口
系统 SHALL 提供 Gradio 启动入口，组织模型加载、分析、可视化、报告的功能区，并以队列模式启动以支持并发长任务。

#### Scenario: 启动界面
- **WHEN** 用户执行启动命令
- **THEN** 系统在本地端口启动 Gradio 应用，开启队列模式

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
- **THEN** 界面按 excess kurtosis 取 top-k 切片绘制 violin，且每个 violin 标注其 kurtosis 值

#### Scenario: 形态标签可视化
- **WHEN** 某切片被标记为“重尾”
- **THEN** 形态热力图或图例中对该切片的 kurtosis 值以可区分方式标示

### Requirement: 量化对比视图
系统 SHALL 展示量化前后的分布对比与多方案误差对比图。

#### Scenario: 对比量化前后
- **WHEN** 用户对某层执行量化模拟并查看对比
- **THEN** 界面并排展示量化前后分布与误差数值

### Requirement: 交互筛选
系统 SHALL 支持按层、模块类型（attn/mlp）、bit-width 筛选与下钻。

#### Scenario: 按模块类型筛选
- **WHEN** 用户筛选模块类型为 attn
- **THEN** 界面仅展示 attention 相关 module 的统计与图表

### Requirement: 异步队列与进度
系统 SHALL 将耗时推理与统计任务放入队列执行，并向界面推送进度，不阻塞 UI。

#### Scenario: 长任务进度反馈
- **WHEN** 用户触发一次大模型校准推理
- **THEN** UI 不冻结，界面实时显示批进度百分比直至完成

### Requirement: 模型结构概览
系统 SHALL 在模型加载完成后展示模型概览，包括架构族、是否降级、全模型总参数量、transformer 层数、目标 module 数。

#### Scenario: 加载后展示概览
- **WHEN** 用户成功加载模型
- **THEN** 界面展示架构族、降级状态、全模型总参数量、transformer 层数、目标 module 数

### Requirement: 逐层 module 结构表格
系统 SHALL 展示目标 module 的表格，每行包含 path、kind（attn/mlp/other）、权重 shape、dtype、参数量。

#### Scenario: 查看模块表格
- **WHEN** 模型加载完成
- **THEN** 界面展示每个目标 module 的 path / kind / shape / dtype / 参数量

#### Scenario: 计算单模块参数量
- **WHEN** 展示某 module 的参数量
- **THEN** 系统按 weight 参数量加上偏置参数量（若有 bias）计算并显示

### Requirement: 按层排序展示
系统 SHALL 按 transformer layer index 组织与排序 module 表格，便于逐层浏览。

#### Scenario: 按 layer index 排序
- **WHEN** 展示 module 表格
- **THEN** 表格按 layer index 升序排列，无法解析 layer index 的 module 排在最前

### Requirement: 完整网络结构展示
系统 SHALL 在加载模型后展示完整的模型结构文本（等价于 print(model)），且该完整结构 SHALL 展示在逐层 module 表格之前，使用户先看到全局网络再浏览逐层细节。

#### Scenario: 展示完整网络结构
- **WHEN** 模型加载完成
- **THEN** 界面展示模型的完整结构文本（包含所有子模块与层级关系）

#### Scenario: 完整结构置于逐层表格之前
- **WHEN** 模型加载完成后展示加载结果
- **THEN** 完整网络结构文本出现在逐层 module 表格之前

### Requirement: 激活分布可视化
系统 SHALL 在「校准与激活」功能区展示选定 module 的**输入**激活分布（即 W8A8 中被量化的 Linear 输入 x），达到与权重分布视图同等的形态分析深度：全局激活直方图、per-token abs_max 分布的 violin（KDE + IQR + 中位数，离群 token 标记）、per-token 切片分布的 top-k by kurtosis violin 与 per-token 形态热力图。per-token 切片视图 SHALL 基于按需单 module 输入激活样本（复用 `capture_sample_inputs`，模型加载后即可用，无需校准），且 SHALL 将 `[batch, seq, hidden]` 展平为 `[N, hidden]` 使每个 token 成为一个切片；per-token abs_max 分布 violin 与离群标记 SHALL 依赖校准聚合的 per-token 输入统计。系统 SHALL 在未满足查看条件时给出明确提示，并 SHALL 移除冗余 / 过时的解释文本，仅保留有助于解读的术语说明。

#### Scenario: 查看激活全局直方图
- **WHEN** 用户运行校准（启用直方图采集）完成后选择某目标 module
- **THEN** 界面展示该 module 输入激活的全局分布直方图

#### Scenario: 查看 per-token 切片形态视图
- **WHEN** 用户在模型加载后选择某 module 并查看激活分布
- **THEN** 界面展示该 module 一次输入激活样本的 per-token 切片 violin（top-k by kurtosis，每 token 一个切片）与 per-token 形态热力图，无需先运行校准

#### Scenario: 查看 per-token abs_max 分布与离群标记
- **WHEN** 用户运行校准（启用 per-token 统计）后查看激活分布
- **THEN** 界面以 violin 展示 per-token abs_max 分布，并标记离群 token、显示离群比例与 severity

#### Scenario: 未采集所需数据时的提示
- **WHEN** 用户未运行校准或未启用对应采集项即请求依赖校准的视图
- **THEN** 系统提示需先运行校准并启用对应采集项，不渲染空图；per-token 切片视图仍可用

#### Scenario: 切换 module 查看不同层激活
- **WHEN** 用户切换所选 module
- **THEN** 界面更新为所选 module 的激活视图，无需重新运行校准

#### Scenario: 移除冗余与过时解释文本
- **WHEN** 渲染激活 / 敏感性 / 报告功能区
- **THEN** 界面不包含冗长或引用未实际接入指标（如 `act_cv`）的说明段，仅保留有助于解读的术语说明
