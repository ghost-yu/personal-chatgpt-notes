# 第 11 讲：梯度累积 gradient accumulation

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：梯度求和、有效 batch、loss 缩放、末批与 Trainer 更新流程。
2. [纯文本 Python 阅读版](代码/tutorials/gradient_accumulation_代码注释.py)：原 Python 代码与中文注释，Markdown 循环以注释保留。
3. [注释 notebook](代码/tutorials/代码注释.ipynb)：全部 28 格说明，15 代码格及 Markdown 中两段循环完整注释。

原循环不足 4 批的尾窗口仍除以 4，梯度偏小。正确归一化的可执行示例见 [窗口归一化教学补充](代码/窗口归一化_教学补充.py)，按实际样本数处理短窗口与不等批次。

<details>
<summary>原版对照、单元说明与验证资料</summary>

- [上游原 notebook](代码/tutorials/gradient_accumulation.ipynb)与[只插说明的对照版](代码/tutorials/原代码_带说明.ipynb)。
- [全部 28 格编号说明](代码单元解释.md)。
- [实测范围、环境与重做命令](复现/README.md)。

从 `代码/tutorials` 启动内核以显示原图。原代理、BERT-large/CUDA 语句仍保留，不要把历史输出当作新运行结果。

原/注释各 11 格不改源码执行一致；两段原 Markdown 循环真实执行，小 BERT 的受控 baseline/累积训练各两次更新。未下载 BERT-large 或复现双卡显存/速度实验。

</details>
