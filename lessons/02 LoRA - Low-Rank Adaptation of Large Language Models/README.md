# 第 02 讲：LoRA 原理与矩阵代码

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：先手算矩阵乘法、低秩和参数量，再对照原论文截图读LoRA；图片与中文解释直接在笔记中。
2. [纯文本 Python 阅读版](代码/tutorials/02_lora_baics_代码注释.py)：直接阅读代码与注释。
3. [需要输出时看注释 notebook](代码/tutorials/代码注释.ipynb)：原课程代码旁有中文注释，每格前有说明。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/02_lora_baics.ipynb)：保留原单元及历史输出。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原代码不改，只插入说明。
- [按编号查单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。

来源为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/02_lora_baics.ipynb)，固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。`baics` 是原文件名，不改写成 `basics`。新增中文说明、注释与验证脚本由本仓库编写。

运行图片展示时，从本讲 `代码/tutorials` 启动内核，图片位于 `代码/imgs`。原代码的7格实际只是导入Image和展示图片；LoRA训练不是这些单元的运行结果。Markdown13有未导入torch/nn/math、W占位、alpha未定义和return缩进错误，本注释版保留并标明；可运行教学例另列，未运行BLOOM大模型或论文评测。

</details>
