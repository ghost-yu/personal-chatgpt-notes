# 第 13 讲：Gradient checkpointing

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：理解重算、激活、梯度与训练配置。
2. [再看详细代码注释](代码/tutorials/代码注释.ipynb)：全部23格说明、12代码格和Markdown21开关类详注。
3. [纯文本 Python 阅读版](代码/tutorials/gradient_checkpointing_代码注释.py)：直接阅读代码和注释。

另有 [重算与梯度教学补充](代码/重算与梯度_教学补充.py)，展示真实Torch分段调用、dropout重放和冻结embedding边界。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/gradient_checkpointing.ipynb)：固定提交原字节，保留历史结果。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原代码不改，只插说明。
- [按编号查单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。
- `来源清单.json` 用于追溯来源，日常阅读无需打开。

原字节/schema/AST/历史输出核对通过。两版各10原格在CPU教学上下文执行；实际四组小BERT各两步训练，固定batch只切checkpoint的参数一致，原/注释对应四组状态精确一致。另验证真实重算、梯度、RNG与评估失败边界。未复现原BERT-large/GPU峰值与速度。

来源：[课程固定 notebook](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/gradient_checkpointing.ipynb)。阅读版保留原CUDA加载与训练配置，CPU执行改动见复现说明。

</details>
