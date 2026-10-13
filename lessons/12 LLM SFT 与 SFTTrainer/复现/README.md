# 第 12 讲复现记录

状态：**部分成功**。完成标准库数学验证、packing 边界验证和官方旧版类的受控 fixture 验证。未运行 Xgen 权重、真实 tokenizer、CUDA、反向传播、SFTTrainer 初始化或 Alpaca 全量训练。

在本目录执行：

```powershell
python scripts/verify_sft.py --output outputs/教学验证.json
```

`--output` 指定报告位置。可选 `--sources` 指向本地官方源码快照目录，其中需有 `sft-v0.4.7.py` 与 `utils-v0.4.7.py`。该模式仅提取 `ConstantLengthDataset` 类，注入整数 tokenizer fixture 和 `LongTensor=list`，执行其真实迭代逻辑；这不是实际 PyTorch tensor 测试。

本次命令：

```powershell
python scripts/verify_sft.py --sources "G:/zzz/Personal ChatGPT 系列/12 LLM SFT 与 SFTTrainer/资料" --output outputs/教学验证.json
```

Windows；Python 3.7.6。本机 torch 是缺少 Tensor/tensor API 的不完整 namespace，不能运行模型。GPU 为 RTX 4060 Laptop 8 GiB；没有下载 7B 权重。报告验证两个 4-token 块、一个丢弃尾 token、CLM 一次 shift、EOS 标签与 LM collator 的差异，以及 8,388,608 个默认 q/v LoRA 参数。

历史 notebook 未锁定依赖版本。本次参考 TRL v0.4.7、PEFT v0.4.0、Transformers v4.30.0，不能据此声称恢复了作者完整环境。实际训练还需锁定 CUDA/PyTorch/bitsandbytes/Accelerate、模型与数据 revision，确认单卡可承载，并检查训练后 adapter 能否重新加载；这些未验证。
