# distribution-statistics Specification

## Purpose
TBD - created by archiving change add-quant-analysis-core. Update Purpose after archive.
## Requirements
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
系统 SHALL 将权重/激活数值分桶为直方图，支持配置桶数、值域范围（自动或指定）与粒度（per-tensor / per-channel / per-group）。per-channel / per-group 粒度下，系统对每个切片分别分桶并返回每切片的直方图。

#### Scenario: 生成分布直方图
- **WHEN** 请求某张量的分布直方图与桶数 256
- **THEN** 系统返回 256 个桶的计数与边界

#### Scenario: 生成 per-channel 直方图
- **WHEN** 请求某 Linear 权重的 per-channel 分布直方图与桶数 128
- **THEN** 系统返回每个输出通道各自的 128 桶计数与边界

#### Scenario: 生成 per-group 直方图
- **WHEN** 请求某权重以 group_size=64 的 per-group 分布直方图
- **THEN** 系统返回每个 group 各自的直方图计数与边界

### Requirement: 分布形态指标
系统 SHALL 在 per-channel / per-group / per-token 粒度上计算权重/激活的分布形态指标，包括 excess kurtosis（超额峰度）、skewness（偏度）与 robust tail-ratio（`(p99.9 − p0.1) / (2·std)`），并基于 excess kurtosis 给出可读形态标签（正态 / 重尾 / 轻尾）。per-token 粒度针对激活：对一次激活样本的每个 token 的 hidden 维分布计算形态指标，用于识别重尾 / 离群 token。指标实现 SHALL 仅依赖 numpy，不引入新运行时依赖。

#### Scenario: 计算 per-channel 形态指标
- **WHEN** 请求某 Linear 权重的 per-channel 形态指标
- **THEN** 系统返回每个输出通道的 excess kurtosis、skewness、tail-ratio 与形态标签

#### Scenario: 计算 per-token 激活形态指标
- **WHEN** 请求某 module 一次激活样本的 per-token 形态指标
- **THEN** 系统返回每个 token 的 hidden 维分布的 excess kurtosis、skewness、tail-ratio 与形态标签

#### Scenario: 识别重尾通道
- **WHEN** 某通道权重的 excess kurtosis 超过重尾阈值（默认 3）
- **THEN** 系统为该通道标记形态标签为“重尾”

#### Scenario: 识别轻尾通道
- **WHEN** 某通道权重的 excess kurtosis 低于轻尾阈值（默认 -0.5）
- **THEN** 系统为该通道标记形态标签为“轻尾”

#### Scenario: 偏态提示非对称量化
- **WHEN** 某切片权重的 |skewness| 超过偏态阈值（默认 0.5）
- **THEN** 系统提示该切片分布偏移，非对称量化可能更优

#### Scenario: 零方差切片的鲁棒处理
- **WHEN** 某切片权重的标准差为 0
- **THEN** 系统将其 tail-ratio 返回为 NaN，且不抛出异常

### Requirement: 离群值检测
系统 SHALL 基于百分位或 Z-score 检测权重/激活中的离群值，并报告离群值的比例与位置。对 per-token abs_max 分布，系统 SHALL 量化离群 token 的比例、severity（最大幅值 / 中位幅值）与最大幅值。

#### Scenario: 量化 per-token 离群幅度
- **WHEN** 请求某 module 激活的 per-token abs_max 分布离群分析
- **THEN** 系统返回离群 token 比例、severity（最大 abs_max / 中位 abs_max）与最大 abs_max 幅值
### Requirement: 分布距离度量
系统 SHALL 计算两个分布之间的 KL 散度与 Wasserstein 距离，用于量化前后或层间对比。

#### Scenario: 对比量化前后分布
- **WHEN** 给定量化前与量化后的张量分布
- **THEN** 系统返回两者间的 KL 散度与 Wasserstein 距离

