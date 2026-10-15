# 第 12 讲：LLM SFT 与 SFTTrainer

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：追踪数据、packing、监督标签、LoRA 和训练流程。
2. [纯文本 Python 阅读版](代码/tutorials/llm_training_sft_代码注释.py)：直接看代码和中文注释。
3. [需要输出时看注释 notebook](代码/tutorials/代码注释.ipynb)：全部 27 格说明、22 个代码格注释，保留原历史输出。

另有 [监督标签教学补充](代码/监督标签_教学补充.py)，可对照真实 collator 读懂 EOS 与 -100。这是独立补充，不是原作者单元。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/llm_training_sft.ipynb)：固定提交原字节，保留历史输出。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：不改原单元，只插入说明。
- [按原编号查单元说明](代码单元解释.md)。
- [实际验证范围、环境和重做命令](复现/README.md)。
- `来源清单.json` 用于追溯来源，日常阅读无需打开。

已核对 schema、AST、历史输出与全部原语句。原版和注释版各 16 格在真实 CPU 教学上下文执行，两条明确配置变体各训练 2 步，adapter 与输出精确一致；保存重载输出在记录的 FP32 容差内一致。未运行原 Xgen-7B/CUDA 4bit/Alpaca 全量训练。

来源：[课程固定 notebook](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/llm_training_sft.ipynb)。原 notebook 的代理与 GPU 设置仍保留，阅读版不是可直接照搬到 CPU 的训练脚本。

</details>
