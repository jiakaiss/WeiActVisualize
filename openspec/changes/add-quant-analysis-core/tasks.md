# Implementation Tasks

## 1. 工程骨架与基础设施

- [x] 1.1 创建 src 布局包结构 `src/weiacviz/` 及子模块 `loading/` `stats/` `quant/` `viz/` `report/` `shared/`
- [x] 1.2 创建 `pyproject.toml`，声明依赖（torch/transformers/datasets/gradio/plotly/numpy/scipy/pydantic-settings）与包元数据
- [x] 1.3 实现 `shared/config.py`：基于 pydantic-settings 的配置 dataclass（模型路径/dtype/device/校准数据/缓存路径）
- [x] 1.4 实现 `shared/cache.py`：按配置指纹（hash）落盘缓存（parquet/json），支持读写与配置变更失效
- [x] 1.5 定义 `shared/types.py` 共享数据类型（StatResult / QuantConfig / CaptureConfig 等）
- [x] 1.6 搭建 pytest 测试骨架与目录结构，编写 config 与 cache 的单元测试

## 2. 模型加载与张量捕获

- [x] 2.1 实现 `loading/model_loader.py`：HF 模型加载（ID/本地路径）、dtype 控制、device_map=auto 分块加载与 CPU offload
- [x] 2.2 实现 `loading/module_resolver.py`：按架构族（Llama-like / Qwen-like）归类目标 module，未知架构降级枚举全部 Linear
- [x] 2.3 实现 `loading/hook.py`：非侵入式 forward hook 注册/卸载，捕获目标 module 输入/输出激活
- [x] 2.4 实现 `loading/calibration.py`：加载 WikiText2/C4 与自定义语料，tokenize，配置样本数与批次
- [x] 2.5 实现 `loading/runner.py`：分批推理 + 激活在线聚合（running stats），每批释放显存
- [x] 2.6 实现 `loading/weights.py`：通过 named_parameters 读取权重，支持 per-tensor/channel/group 切片
- [x] 2.7 编写加载与捕获单元测试（用 TinyLlama / Qwen2-0.5B 小模型验证）

## 3. 分布统计

- [x] 3.1 实现 `stats/weight_stats.py`：权重 per-tensor/channel/group 的 min/max/mean/std/percentile
- [x] 3.2 实现 `stats/activation_stats.py`：激活 per-token/channel 统计
- [x] 3.3 实现 `stats/histogram.py`：直方图分桶（可配桶数与值域，自动或指定）
- [x] 3.4 实现 `stats/outliers.py`：百分位 / Z-score 离群值检测，返回比例与位置
- [x] 3.5 实现 `stats/distance.py`：KL 散度与 Wasserstein 距离
- [x] 3.6 编写统计模块单元测试（已知输入验证数值正确性）

## 4. 量化模拟

- [x] 4.1 实现 `quant/fake_quant.py`：round-clamp-dequant，支持 bit-width / 对称非对称 / per-tensor/channel/group
- [x] 4.2 实现 `quant/scheme_compare.py`：同张量多方案（bit × 粒度 × 对称性）误差对比表
- [x] 4.3 实现 `quant/error_metrics.py`：MSE、cosine similarity、层输出 diff
- [x] 4.4 实现 `quant/sensitivity.py`：按层/通道排序量化损失，标注最高损失 N 个层
- [x] 4.5 编写量化模拟单元测试（验证量化网格与误差计算正确性）

## 5. 交互式可视化

- [x] 5.1 实现 `viz/app.py`：Gradio 启动入口、功能区布局、队列模式启动
- [x] 5.2 实现 `viz/charts.py`：Plotly 直方图与 per-channel/layer 统计热力图渲染
- [x] 5.3 实现 `viz/comparison.py`：量化前后分布对比与多方案误差对比视图
- [x] 5.4 实现 `viz/filters.py`：按层 / 模块类型(attn/mlp) / bit-width 筛选与下钻
- [x] 5.5 实现 `viz/progress.py`：长任务进度回调，确保 UI 不阻塞
- [x] 5.6 手动验证界面交互（小模型端到端跑通）

## 6. 报告与导出

- [x] 6.1 实现 `report/recommend.py`：基于统计与敏感性生成量化建议报告（分层级 bit-width/粒度建议 + 理由）
- [x] 6.2 实现 `report/export.py`：导出 Markdown 报告、CSV/JSON 数据、PNG 图表
- [x] 6.3 编写报告与导出的单元测试

## 7. 集成与验证

- [x] 7.1 编写端到端集成测试：加载小模型 -> 捕获 -> 统计 -> 量化模拟 -> 报告全流程
- [ ] 7.2 大模型（7B）冒烟验证：分块加载 + 分批推理 + 在线聚合的内存占用与正确性
- [x] 7.3 编写 README 与使用示例（启动命令、配置说明、典型工作流）
- [x] 7.4 梳理并确定 Open Questions 中可决议项（默认校准样本数、离群值阈值默认值）
