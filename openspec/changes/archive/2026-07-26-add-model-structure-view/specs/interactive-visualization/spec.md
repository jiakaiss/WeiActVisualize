## ADDED Requirements

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
