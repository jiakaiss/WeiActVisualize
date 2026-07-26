## ADDED Requirements

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
