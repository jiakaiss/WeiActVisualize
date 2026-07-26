# Implementation Tasks

## 1. 模型结构数据处理

- [x] 1.1 实现 `viz/structure.py`：`build_module_table(resolve_result, model)` 返回表格行（path / kind / shape / dtype / params / layer），含参数量计算（weight + bias）与 layer index 提取
- [x] 1.2 实现 `build_overview(resolve_result, model)` 返回概览（架构族 / 降级状态 / 全模型总参数量 / transformer 层数 / 目标 module 数）

## 2. UI 集成

- [x] 2.1 改 `viz/app.py` 的 `App.load()`：返回 (状态文本, 概览 markdown, 表格 dataframe)
- [x] 2.2 「模型加载」tab 新增 gr.Markdown（概览）+ gr.Dataframe（module 表格）组件并绑定 load 输出
- [x] 2.3 手动验证：加载 Qwen2-0.5B，确认概览与逐层表格正确展示（含 shape / 参数量）
- [x] 2.4 在「模型加载」tab 加 gr.Code 展示完整网络结构（str(model)），load() 返回该文本

## 3. 测试

- [x] 3.1 编写 `tests/test_structure.py`：参数量计算、layer index 解析、按层排序的正确性
