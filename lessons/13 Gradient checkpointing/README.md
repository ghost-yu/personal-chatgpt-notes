# 第 13 讲：Gradient checkpointing

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从几个数的前向/求导讲到边界保存、分段、RNG重放和BERT开关；原论文Figure1/Figure3/Algorithm1/公式1直接截图就地解释，四段正文代码实际执行，旧完整讲解折叠保留。
2. [纯文本 Python 阅读版](代码/tutorials/gradient_checkpointing_代码注释.py)：直接阅读代码和注释。
3. [需要输出时看注释 notebook](代码/tutorials/代码注释.ipynb)：全部23格说明、12代码格和Markdown21开关类详注。

另有 [重算与梯度教学补充](代码/重算与梯度_教学补充.py)，展示真实Torch分段调用、dropout重放和冻结embedding边界。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/gradient_checkpointing.ipynb)：固定提交原字节，保留历史结果。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原代码不改，只插说明。
- [按编号查单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。
- `来源清单.json` 用于追溯来源，日常阅读无需打开。

历史交付已核对原字节/schema/AST/历史输出，两版各10原格及四组H16小BERT训练等旧验证保留，本轮未重跑。新增四段正文例子另外核对标量手算、4块H2分段、8元素Dropout重放和1706参数小BERT两条路径各一次SGD；详见复现说明。未复现原BERT-large/GPU峰值与速度。

来源：[课程固定 notebook](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/gradient_checkpointing.ipynb)。阅读版保留原CUDA加载与训练配置，CPU执行改动见复现说明。

</details>
