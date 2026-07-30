## MODIFIED Requirements

### Requirement: 激活在线聚合
系统 SHALL 在分批推理过程中以 running statistics 方式在线聚合激活标量统计量（min/max/mean/std/abs_max），不保留原始激活张量，以控制内存并与模型规模解耦。系统 SHALL 支持在线聚合全局激活直方图（固定值域、跨批累加桶计数），且 SHALL 支持两遍校准以确定直方图值域：第一遍聚合标量统计量取全局极值，第二遍以该极值为固定值域分桶。直方图聚合的内存占用 SHALL 与批数与序列长度无关。

#### Scenario: 多批标量聚合
- **WHEN** 校准数据被分为多批依次推理
- **THEN** 系统逐批更新标量聚合统计量且不累积原始激活，内存占用与批数无关

#### Scenario: 在线直方图聚合
- **WHEN** 启用直方图采集并执行多批校准推理
- **THEN** 系统以固定值域跨批累加桶计数，产出每 module/role 的全局激活直方图，且不保留原始激活

#### Scenario: 两遍校准确定值域
- **WHEN** 启用直方图采集
- **THEN** 系统第一遍产出每 module/role 的全局 min/max，第二遍以 [min, max] 为固定值域分桶，使直方图覆盖全局值域

#### Scenario: 值域退化鲁棒处理
- **WHEN** 某 module/role 激活的全局 min 等于 max
- **THEN** 系统以该值为中心扩展为非零宽度值域并正常分桶，不抛出异常

#### Scenario: 关闭直方图采集
- **WHEN** 未启用直方图采集
- **THEN** 系统仅执行单遍标量聚合，不产生直方图，行为与既有流程一致
