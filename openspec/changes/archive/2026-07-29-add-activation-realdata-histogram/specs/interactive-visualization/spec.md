## ADDED Requirements

### Requirement: 激活分布可视化
系统 SHALL 在「校准与激活」功能区于校准完成后展示选定 module 的全局激活直方图，使用户可观察激活数值分布。系统 SHALL 支持选择目标 module、配置是否采集直方图及桶数，并 SHALL 在未满足查看条件时给出明确提示。

#### Scenario: 查看激活全局直方图
- **WHEN** 用户运行校准（启用直方图采集）完成后选择某目标 module
- **THEN** 界面展示该 module output 的全局激活分布直方图

#### Scenario: 未采集直方图时的提示
- **WHEN** 用户未启用直方图采集或尚未运行校准即请求查看激活分布
- **THEN** 系统提示需先运行校准并启用直方图采集，不渲染空图

#### Scenario: 切换 module 查看不同层激活
- **WHEN** 用户在校准完成后切换所选 module
- **THEN** 界面更新为所选 module 的全局激活直方图，无需重新运行校准
