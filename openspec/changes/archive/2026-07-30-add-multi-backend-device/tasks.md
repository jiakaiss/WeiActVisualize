## 1. 后端检测与 load_model

- [x] 1.1 `loading/model_loader.py` 新增 `detect_available_backend() -> str`：cuda（`torch.cuda.is_available`）> npu（容错 `import torch_npu; torch.npu.is_available`）> cpu
- [x] 1.2 `load_model` 加 npu 分支：`import torch_npu`（失败给清晰错误）+ `model.to("npu")`
- [x] 1.3 `auto` 分支：detect 后 cuda -> `device_map="auto"`；npu -> `model.to("npu")`；cpu -> `device_map="cpu"`
- [x] 1.4 保持 `model.eval()` 与 tokenizer 逻辑不变

## 2. 配置与 UI

- [x] 2.1 `shared/config.py`：`Settings.device` Literal 为 `auto/cpu`，默认 `"auto"`
- [x] 2.2 `viz/app.py`「模型加载」tab device 下拉为 `auto / cpu`

## 3. 测试

- [x] 3.1 `tests/test_loading.py`：`detect_available_backend` 返回合法 device 字符串（`cpu/cuda/npu`）
- [x] 3.2 `tests/test_loading.py`：`load_model(device="cpu")` 分支正确（mock）
- [x] 3.3 `tests/test_loading.py`：npu 缺失 torch_npu 时 `_move_to_backend` raise RuntimeError 不崩

## 4. 验证

- [x] 4.1 `C:/Users/wjk/.conda/envs/weiacviz/python.exe -m pytest tests/ -q` 全绿
- [x] 4.2 `openspec validate add-multi-backend-device --strict` 通过
