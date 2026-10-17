# 第 04 讲：PEFT 怎样把 LoRA 注入原模型

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：先理解对象引用、层与参数，再看目标层查找、替换和双分支；论文原图与公式已直接内嵌。
2. [纯文本 Python 阅读版](代码/tutorials/04_PEFT_lora_代码注释.py)：直接阅读代码与注释。
3. [需要输出时看注释 notebook](代码/tutorials/代码注释.ipynb)：原课程代码旁有中文注释，每格前有说明。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/04_PEFT_lora.ipynb)：保留原单元及历史输出。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原代码不改，只插入说明。
- [按编号查单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。

来源为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/04_PEFT_lora.ipynb)，固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。新增中文解释、注释与验证脚本由本仓库编写。原notebook未引用配套图片文件；本轮从LoRA v2论文直接截图Figure1和公式3，放在正文对应讲解旁。

实际验证直接使用[官方PEFT v0.3.0固定源码](https://github.com/huggingface/peft/tree/7120c9a2636f93c8db3fc4ae466e02f338ecead8)及CPU随机小型BLOOM；老师的0.4.0.dev0没有SHA，仍未精确重建。原版和注释版的模块匹配、真实注入、冻结、前向/梯度、disable及merge/unmerge结果一致。未运行原7B量化加载，也没有训练模型。Markdown7缺默认字段的dataclass节选仍按预期报错，注释版只标明、不静默修复。

</details>
