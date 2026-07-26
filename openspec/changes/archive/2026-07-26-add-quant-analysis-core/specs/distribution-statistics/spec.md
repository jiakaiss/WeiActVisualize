## ADDED Requirements

### Requirement: 权重分布统计
系统 SHALL 计算权重张量的 per-tensor / per-channel / per-group 统计量，包括 min/max/mean/std 与指定百分位。

#### Scenario: 计算权重 per-channel 统计
- **WHEN** 请求某 Linear 层权重的 per-channel 统计
- **THEN** 系统返回每个输出通道的 min/max/mean/std 与百分位

### Requirement: 激活分布统计
系统 SHALL 计算激活的 per-token / per-channel 统计量，反映激活幅值在不同位置与通道的分布。

#### Scenario: 计算激活 per-token 范围
- **WHEN** 请求某层激活的 per-token 统计
- **THEN** 系统返回每个 token 位置上的幅值范围统计

### Requirement: 直方图分桶
系统 SHALL 将权重/激活数值分桶为直方图，支持配置桶数与值域范围（自动或指定）。

#### Scenario: 生成分布直方图
- **WHEN** 请求某张量的分布直方图与桶数 256
- **THEN** 系统返回 256 个桶的计数与边界

### Requirement: 离群值检测
系统 SHALL 基于百分位或 Z-score 检测权重/激活中的离群值，并报告离群值的比例与位置。

#### Scenario: 检测激活离群通道
- **WHEN** 请求某层激活的离群值分析（百分位 99.9%）
- **THEN** 系统返回超过阈值的通道索引、离群比例与幅值

### Requirement: 分布距离度量
系统 SHALL 计算两个分布之间的 KL 散度与 Wasserstein 距离，用于量化前后或层间对比。

#### Scenario: 对比量化前后分布
- **WHEN** 给定量化前与量化后的张量分布
- **THEN** 系统返回两者间的 KL 散度与 Wasserstein 距离
