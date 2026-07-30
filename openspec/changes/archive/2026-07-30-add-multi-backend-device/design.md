## Context

`load_model`（`model_loader.py:37-81`）按 `device` 参数加载：`cpu`（`device_map="cpu"`）、`cuda`（`model.to("cuda")`）、`auto`（`device_map="auto"`）。`Settings.device` 默认 `"auto"`，Literal `cpu/cuda/auto`，UI 下拉同。激活 hook 把激活搬 CPU、统计走 numpy（有意，内存解耦，见 `add-activation-realdata-histogram`）。

约束：只动推理设备，不碰统计。`torch_npu` 可选依赖（不强装）。`auto` 须向后兼容（无加速器落 CPU）。用户面向仅 `auto/cpu`，不暴露 cuda/npu/mps 显式选项（自动检测即可）。

## Goals / Non-Goals

**Goals:**
- device 用户面向 `auto / cpu`。
- `auto` 检测可用加速器并优先（cuda > npu > cpu）。
- cuda 保留 `device_map="auto"` 分块（大模型）；npu 用 `model.to("npu")`。
- `torch_npu` 可选，未装时不崩。
- 向后兼容：`cpu` 行为不变。

**Non-Goals:**
- 统计聚合搬加速器（仍 CPU/numpy）。
- npu 上的 `device_map` 分块（仅 cuda 保留；npu 单设备 `to`，大模型 OOM 风险由 auto 检测承担）。
- 多卡并行（CUDA/NPU 多卡）。
- 暴露 cuda/npu/mps 显式选项（自动检测足够）。

## Decisions

### D1: 后端检测 `detect_available_backend()`
- 顺序探测 `cuda > npu > cpu`：
  - cuda：`torch.cuda.is_available()`
  - npu：`try: import torch_npu; torch.npu.is_available()`，import 失败则不可用（容错）
  - 均不可用则 cpu
- 返回设备字符串。
- **理由**：容错 import 避免 `torch_npu` 未装时崩溃。用户面向仅 auto/cpu，不暴露更多后端。

### D2: `auto` 语义
- `auto` = `detect_available_backend()` 的结果：
  - cuda -> `device_map="auto"`（保留分块，大模型支持）
  - npu -> `model.to("npu")`（单设备）
  - cpu -> `device_map="cpu"`
- **备选**：`auto` 始终 `device_map="auto"`--否决，transformers 的 `device_map="auto"` 不识别 npu 后端。
- **理由**：cuda 保留分块价值；npu 用 `to` 是 torch 标准方式。

### D3: device 分支（内部）
- 用户面向仅 `auto` / `cpu`；`auto` 内部解析为 cuda/npu/cpu：
  - cpu -> `device_map="cpu"`
  - cuda -> `device_map="auto"` 分块
  - npu -> `model.to("npu")`（需 `torch_npu` 已装，否则清晰报错）
- **理由**：简化用户选择；cuda 分块健壮；npu 用 `to`。不暴露 cuda/npu/mps 显式选项。

### D4: `torch_npu` 可选依赖
- 不写入 `pyproject` 必需依赖。
- `load_model` 的 npu 分支尝试 `import torch_npu`，失败给清晰错误（"NPU 需安装 torch_npu"）。
- **理由**：多数用户无 NPU，不强装。

### D5: 配置与 UI
- `Settings.device` Literal 为 `auto/cpu`，默认 `"auto"`。
- UI「模型加载」tab 的 device 下拉为 `auto / cpu`。
- **理由**：默认 `auto` 在 GPU/NPU 机器上开箱即用加速器；`cpu` 供强制 CPU 调试。

## Risks / Trade-offs

- [npu 大模型 OOM] -> 仅 `to("npu")` 无分块；auto 检测到 npu 即承担；文档提示大模型优先用 cuda+auto 分块。
- [torch_npu 未装] -> 容错 import + 清晰错误信息。
- [auto 检测 cuda 优先] -> 合理（NVIDIA 最常见）；npu 次之。

## Migration Plan

纯增量。`auto` 语义增强（原仅 cuda 分块，现 cuda 分块 + npu `to`）。回滚即 revert，无数据/配置格式变更。

## Open Questions

- npu 上是否需要 `device_map` 分块？第一版不做（Ascend 多卡分块需 huggingface accelerate 适配，后续迭代）。
