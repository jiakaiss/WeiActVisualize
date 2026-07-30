## Why

当前 `load_model` 仅支持 `cpu / cuda / auto` 三个 device 选项（`model_loader.py:48-50`，`Settings.device` Literal 为 `cpu/cuda/auto`，默认 `auto`）。`auto` 用 `device_map="auto"`，在有 NVIDIA GPU 的机器上会用 GPU，但**不支持华为 Ascend NPU（torch_npu）与 Apple MPS**；且 `auto` 在无 CUDA 的 NPU/MPS 机器上无法自动走加速器（仅做 CPU offload）。用户希望在 GPU 或 NPU 环境下默认走加速器，且后端可配置。本变更只动推理设备，不碰统计聚合（仍 CPU/numpy，保持 `add-activation-realdata-histogram` 的内存解耦设计）。

## What Changes

- 扩展 device 选项为 `cpu / cuda / npu / mps / auto`。
- `load_model` 新增 `npu`（torch_npu）、`mps` 分支：`model.to("npu")` / `model.to("mps")`。
- `auto` 语义扩展：检测可用加速器（cuda > npu > mps > cpu），cuda 时保留 `device_map="auto"` 分块，其余后端用 `model.to(device)`。
- 新增后端检测工具 `detect_available_backend()`（`is_available` 探测，容错 import）。
- `Settings.device` Literal 扩展；UI「模型加载」tab 的 device 下拉增加 `npu / mps`。
- `torch_npu` 为**可选**依赖（仅用 NPU 时需要），不强制安装。

## Capabilities

### New Capabilities

（无。）

### Modified Capabilities

- `model-tensor-capture`：「多架构模型加载」需求扩展--device 支持 `cpu / cuda / npu / mps / auto`，`auto` SHALL 检测可用加速器并优先使用，cuda 保留分块加载能力。

## Impact

- **代码**：`loading/model_loader.py`（npu/mps 分支 + `detect_available_backend`）、`shared/config.py`（device Literal 扩展）、`viz/app.py`（device 下拉选项）。
- **依赖**：`torch_npu` 可选（NPU 场景）；MPS 随 PyTorch 自带（Apple）。无强制新增依赖。
- **兼容性**：device 新增值均有默认；现有 `cpu/cuda/auto` 行为不变（`auto` 在无加速器时仍落 CPU）。
- **不触及**：统计聚合路径（`runner.py` / `stats/*` 仍 CPU/numpy）。
