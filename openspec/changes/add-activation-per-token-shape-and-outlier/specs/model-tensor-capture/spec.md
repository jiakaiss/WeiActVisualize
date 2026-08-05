## ADDED Requirements

### Requirement: 按需单 module 激活样本捕获
系统 SHALL 支持按需捕获单个目标 module 的一次前向 output 激活样本，用于 per-token 切片形态可视化。捕获 SHALL 通过非侵入式 forward hook 完成、用后即卸载，且 SHALL 限定为单条样本与有界序列长度以控制内存，仅对当前查看的 module 触发，不影响校准在线聚合。

#### Scenario: 捕获单 module 输出样本
- **WHEN** 用户查看某 module 的 per-token 切片视图
- **THEN** 系统对该 module 运行一次前向并返回其 output 激活张量，hook 用后卸载，内存占用与序列长度成比例且不累积

#### Scenario: 捕获不影响校准聚合
- **WHEN** 按需样本捕获与校准在线聚合共存
- **THEN** 两者使用独立 hook 注册，按需捕获不污染校准 aggregator 的统计量
