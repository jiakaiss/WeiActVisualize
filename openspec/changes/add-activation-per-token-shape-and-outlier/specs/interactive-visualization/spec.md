## MODIFIED Requirements

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
