## ADDED Requirements

### Requirement: 多架构模型加载
系统 SHALL 支持通过 HuggingFace 模型 ID 或本地路径加载模型，覆盖 Llama / Qwen / Mistral / DeepSeek 等主流架构。系统 MUST 支持指定 dtype（fp16/bf16/fp32）与 device（CPU/单卡/auto 分块加载）。

#### Scenario: 加载小模型到 CPU
- **WHEN** 用户指定一个 0.5B-1B 模型 ID 与 device=cpu
- **THEN** 系统成功加载模型并返回可推理的模型实例，dtype 按指定设置

#### Scenario: 大模型分块加载
- **WHEN** 用户指定一个 7B+ 模型且显存不足以单卡容纳
- **THEN** 系统通过 device_map=auto 自动分块加载，必要时 CPU offload，不抛出 OOM

### Requirement: Module 解析与分层索引
系统 SHALL 解析模型结构，为目标 module（q/k/v/o proj、gate/up/down proj 等）建立分层索引，并按架构族归类。未知架构 MUST 降级为枚举全部 Linear 层。

#### Scenario: 识别 Llama 架构目标层
- **WHEN** 加载一个 Llama-like 模型
- **THEN** 系统返回每层的 q_proj/k_proj/v_proj/o_proj 与 MLP 的 gate/up/down_proj 路径列表

#### Scenario: 未知架构降级
- **WHEN** 加载一个不在已知架构族列表中的模型
- **THEN** 系统降级枚举所有 nn.Linear 子模块作为目标，并记录降级提示

### Requirement: 权重张量读取
系统 SHALL 通过 named_parameters 离线读取目标 module 的权重张量，支持按 per-tensor / per-channel / per-group 切片访问。

#### Scenario: 读取某层权重
- **WHEN** 请求某 module 路径的权重
- **THEN** 系统返回该 module 的权重张量及其形状信息

### Requirement: 激活张量捕获
系统 SHALL 通过 forward hook 捕获目标 module 的输入/输出激活。hook MUST 非侵入式注册且可在推理后卸载。

#### Scenario: 捕获校准推理激活
- **WHEN** 在校准数据上执行一次 forward
- **THEN** 所有已注册 hook 的 module 的输入/输出激活被捕获并可被统计模块消费

### Requirement: 校准数据加载
系统 SHALL 支持加载 WikiText2 / C4 等标准数据集与自定义语料作为校准数据。系统 MUST 支持配置样本数与批次大小，并对文本进行 tokenize。

#### Scenario: 加载标准校准集
- **WHEN** 用户指定 WikiText2 与样本数 128
- **THEN** 系统返回 tokenized 的校准批次，可分批送入推理

#### Scenario: 自定义语料
- **WHEN** 用户提供本地文本文件路径
- **THEN** 系统加载并 tokenize 该语料作为校准数据

### Requirement: 激活在线聚合
系统 SHALL 在分批推理过程中以 running statistics 方式在线聚合激活统计量，不保留原始激活张量，以控制内存并与模型规模解耦。

#### Scenario: 多批聚合
- **WHEN** 校准数据被分为多批依次推理
- **THEN** 系统逐批更新聚合统计量且不累积原始激活，内存占用与批数无关

### Requirement: 分批推理
系统 SHALL 支持将校准数据分批送入模型推理，并在每批后释放中间激活显存。

#### Scenario: 分批执行
- **WHEN** 校准样本数超过单批容量
- **THEN** 系统自动分批推理，每批结束后释放激活显存，全部完成后输出聚合统计
