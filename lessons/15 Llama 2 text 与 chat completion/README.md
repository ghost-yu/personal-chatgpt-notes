# 第 15 讲：Llama 2 text 与 chat completion

[返回总目录](../../README.md)

1. [中文讲解](解释.md)：输入模板、原始权重、text/chat包装与真实生成结果的边界。
2. 注释 notebook：[主教程](代码/tutorials/代码注释.ipynb) → [tokenizer与接口补充](代码/tutorials/llama2_src_text_chat_代码注释.ipynb)。两份共28格、12个代码格，Markdown源码和命令也有说明。
3. Python 阅读版：[主教程](代码/tutorials/llama2_src_01_代码注释.py) → [接口补充](代码/tutorials/llama2_src_text_chat_代码注释.py)。

原torchrun、NCCL与真实7B/13B配置未运行。新增验证用官方源码的明确CPU设备变体、随机小Transformer和真实FairScale/Gloo，不代表原权重的回答能力。

<details>
<summary>原版、单元说明与验证记录</summary>

- 原notebook：[主教程](代码/tutorials/llama2_src_01.ipynb) · [接口补充](代码/tutorials/llama2_src_text_chat.ipynb)。两张PNG在代码/imgs，历史输出保留。
- 原代码带说明：[主](代码/tutorials/原代码_带说明.ipynb) · [补充](代码/tutorials/llama2_src_text_chat_原代码_带说明.ipynb)。
- [全部单元说明](代码单元解释.md) · [验证范围、环境与重做方式](复现/README.md)。
- `来源清单.json` 用于追溯来源，日常阅读无需打开。

</details>
