## MODIFIED Requirements

### Requirement: 多架构模型加载
系统 SHALL 支持通过 HuggingFace 模型 ID 或本地路径加载模型，覆盖 Llama / Qwen / Mistral / DeepSeek 等主流架构。系统 MUST 支持指定 dtype（fp16/bf16/fp32）与 device（auto / cpu）。device=auto 时系统 SHALL 检测可用加速器（cuda > npu > cpu）并优先使用：检测到 cuda 时用 device_map="auto" 分块加载，检测到 npu 时以 model.to("npu") 加载，无加速器时落 CPU。系统 SHALL 在 npu 后端缺失 torch_npu 时给出清晰错误而非崩溃。

#### Scenario: 加载小模型到 CPU
- **WHEN** 用户指定一个 0.5B-1B 模型 ID 与 device=cpu
- **THEN** 系统成功加载模型并返回可推理的模型实例，dtype 按指定设置

#### Scenario: 大模型分块加载
- **WHEN** 用户指定一个 7B+ 模型且显存不足以单卡容纳
- **THEN** 系统通过 device_map=auto 自动分块加载，必要时 CPU offload，不抛出 OOM

#### Scenario: auto 检测到 GPU
- **WHEN** 用户指定 device=auto 且环境存在可用 CUDA GPU
- **THEN** 系统以 device_map="auto" 将模型加载到 GPU

#### Scenario: auto 检测到 NPU
- **WHEN** 用户指定 device=auto 且环境无 CUDA 但有 Ascend NPU（torch_npu 已装）
- **THEN** 系统以 model.to("npu") 将模型加载到 NPU

#### Scenario: auto 无加速器落 CPU
- **WHEN** 用户指定 device=auto 且环境无可用加速器
- **THEN** 系统将模型加载到 CPU

#### Scenario: NPU 后端缺失的容错
- **WHEN** auto 检测到 npu 但未安装 torch_npu
- **THEN** 系统报清晰错误提示需安装 torch_npu，不抛出 ImportError 崩溃
