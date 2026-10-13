# 第 15 讲复现记录

**部分成功**：从官方固定源码用 AST 提取 Tokenizer.encode/decode、Llama.__init__/text_completion/chat_completion 和常量，直接执行官方方法。真实 SentencePiece 0.2.0 编码；generate 用常量答案 fixture 替代。它只验证包装层，不执行 torch/CUDA/Transformer，也不生成真实答案。

Python 3.12.14，位于本地共享隔离环境；默认 Python 的 torch 缺少 Tensor API。本机8 GiB单卡没有原始7B FP16权重。原实现build使用NCCL与FairScale，不在当前Windows环境尝试启动。没有下载模型或执行签名URL脚本。

工作目录为本目录，实际命令：

```powershell
& "G:/zzz/Personal ChatGPT 系列/共享运行环境/datasets/Scripts/python.exe" scripts/verify_completion.py --sources "G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料" --tokenizer "G:/zzz/Llama 入门教程/代码/personal_chatgpt/tutorials/llama/llama2_tokenizer.model" --output outputs/教学验证.json
```

`--sources` 目录需要 `llama-generation.py`、`llama-tokenizer.py`、`example_chat_completion.py`，取自 Meta 仓库 commit `7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e`，下载后按上述名字保存；`--tokenizer` 是已有模型词表文件。中文文件路径通过pathlib读取bytes传model_proto，避开native文件名问题。

检查内容：BOS/EOS边界、system合并、多轮角色、原dialog未修改、非法角色断言、空/只有system的IndexError、特殊标签错误替换、6个官方dialog与batch4/8的差异。fixture里logprobs是合成-0.1，不能当模型置信度；输出切片为整数例子，未执行真实generate循环。

真正推理需对应权限下的原始 `.pth` 分片、params.json、tokenizer、兼容NCCL/PyTorch/FairScale的设备与环境，MP和分片数相同，再验证真实生成、EOS和缓存。作者精确commit及全部依赖未找到，本次固定源码只是明确的参考版本。
