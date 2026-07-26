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
系统 SHALL 渲染权重/激活的分布直方图与 per-channel/per-layer 统计量热力图，使用 Plotly。

#### Scenario: 查看某层权重分布
- **WHEN** 用户选择某层的权重分布视图
- **THEN** 界面展示该层权重的直方图与 per-channel 统计热力图

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
系统 SHALL 在加载模型后展示完整的模型结构文本（等价于 print(model)），让用户直观看到整个网络而不仅是目标 Linear 层。

#### Scenario: 展示完整网络结构
- **WHEN** 模型加载完成
- **THEN** 界面展示模型的完整结构文本（包含所有子模块与层级关系）

