# 第 14 讲复现记录

状态：**部分成功**。真实运行 SentencePiece 0.2.0，加载课程仓库的 `tutorials/llama/llama2_tokenizer.model`，验证 32000 vocab、BOS/EOS、英文和中文原创文本往返、对话格式标记与长度。读取 Guanaco 固定 revision 的前五行，检查角色序列与 512-token 截断。用 AST 读取官方 TRL v0.5.0 示例默认配置，推导参数量及更新步数。

本次 Python 为共享隔离环境的 3.12.14，只新增 sentencepiece==0.2.0，没有更改默认 Python 的框架。没有加载 Llama 2 权重、运行 Transformers pipeline、SFTTrainer、4-bit CUDA kernel 或生成回答。

从本目录运行（将 tokenizer 路径换成自己取得的对应文件）：

```powershell
python scripts/verify_llama2.py --tokenizer "path/to/llama2_tokenizer.model" --output outputs/教学验证.json
```

实际命令：

```powershell
& "G:/zzz/Personal ChatGPT 系列/共享运行环境/datasets/Scripts/python.exe" scripts/verify_llama2.py --tokenizer "G:/zzz/Llama 入门教程/代码/personal_chatgpt/tutorials/llama/llama2_tokenizer.model" --samples "G:/zzz/Personal ChatGPT 系列/14 Llama 2 introduction 与 Guanaco/资料/guanaco-five-rows.json" --script "G:/zzz/Personal ChatGPT 系列/14 Llama 2 introduction 与 Guanaco/资料/sft-script-v0.5.0.py" --output outputs/教学验证.json
```

`--samples` 是 JSON 字典列表，每个有 text；`--script` 是待审计的旧版官方脚本。两者可省略；省略就不声称完成相应检查。数据 revision 为 `831dabac2283d99420cda0b673d7a2a43849f17a`，只读取前五行，未全量训练或评估。

本机 RTX 4060 Laptop 8 GiB；7B FP16 权重约12.55 GiB，超过单卡容量。默认 torch 缺少实际 Tensor API，没有下载权重或尝试全量训练。课程推理可跨设备，但内存/速度未验证。旧版训练 script 未锁定作者原始修改，命令可解析的静态证据不证明现代依赖兼容。

真正训练还需具有对应模型访问权限、固定 checkpoint/依赖/数据版本，确保 GPU 可运行，并检查 adapter 保存/重载与相同 prompt 的前后评估。当前文档是教学复现与源码审计，未复现“五小时”运行。
