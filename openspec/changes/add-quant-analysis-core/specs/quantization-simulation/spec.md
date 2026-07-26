## ADDED Requirements

### Requirement: Fake quantization
系统 SHALL 对权重/激活执行 fake quantization（round-clamp-dequant），支持配置 bit-width（W8A8/W4A16/W4A8/INT3）、对称/非对称、per-tensor / per-channel / per-group 粒度。

#### Scenario: 对称 per-channel W4 量化
- **WHEN** 请求对某层权重做 4-bit 对称 per-channel 量化
- **THEN** 系统返回量化-反量化后的权重张量，数值落在量化网格上

#### Scenario: 非对称 per-tensor 激活量化
- **WHEN** 请求对某激活做 8-bit 非对称 per-tensor 量化
- **THEN** 系统返回量化-反量化后的激活张量

### Requirement: 量化方案对比
系统 SHALL 在同一张量上对比不同量化方案（bit-width × 粒度 × 对称性）的误差，输出对比表。

#### Scenario: 对比多方案
- **WHEN** 请求某层在 W4/W8、per-tensor/per-channel 下的量化误差
- **THEN** 系统返回各方案的误差度量对比表

### Requirement: 量化误差度量
系统 SHALL 计算量化前后的误差，包括 MSE、cosine similarity 与层输出 diff（在样本输入上比较 module 输出）。

#### Scenario: 计算层输出误差
- **WHEN** 对某 module 做权重 fake quant 后在样本输入上前向
- **THEN** 系统返回原始输出与量化输出的 MSE 与 cosine similarity

### Requirement: 敏感性分析
系统 SHALL 按层与通道排序量化损失，识别高损失层与通道，作为量化决策输入。

#### Scenario: 层级敏感性排序
- **WHEN** 请求全模型的层级量化敏感性
- **THEN** 系统返回各层量化误差排序，标注最高损失的 N 个层
