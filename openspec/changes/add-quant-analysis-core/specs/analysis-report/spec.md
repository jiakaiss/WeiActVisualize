## ADDED Requirements

### Requirement: 量化建议报告
系统 SHALL 基于分布统计与敏感性分析生成量化建议报告，指出哪些层/模块适合低比特量化、哪些需保留高精度，并给出理由（离群值、误差量级）。

#### Scenario: 生成全模型报告
- **WHEN** 用户请求全模型量化建议报告
- **THEN** 系统输出按层分级的量化建议（推荐 bit-width/粒度）与理由说明

### Requirement: 报告与数据导出
系统 SHALL 支持将报告与图表/统计数据导出为常见格式（Markdown 报告、CSV/JSON 数据、PNG 图表）。

#### Scenario: 导出统计数据
- **WHEN** 用户请求导出某次分析的统计数据
- **THEN** 系统生成包含统计表的 CSV/JSON 文件供下载

#### Scenario: 导出图表
- **WHEN** 用户请求导出某图表
- **THEN** 系统生成 PNG 图片供下载
