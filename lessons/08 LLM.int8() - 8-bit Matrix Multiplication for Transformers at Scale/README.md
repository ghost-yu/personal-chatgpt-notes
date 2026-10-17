# 第08讲：LLM.int8量化与bitsandbytes

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从bit/byte和三个数的量化讲起，再沿原矩阵计算；论文Figure2、Table1直接截图内嵌，中文解释就在图旁。
2. [纯文本 Python 阅读版](代码/tutorials/LLM.int8_代码注释.py)：直接看代码和中文注释，不用看JSON。
3. [需要输出时看注释 notebook](代码/tutorials/代码注释.ipynb)：主notebook全部56格说明，37个代码格（含空格55）保留原语句并详注。

<details>
<summary>辅助配置、原版对照与验证资料</summary>

- [bitsandbytes辅助配置注释](代码/tutorials/bitsandbytes_代码注释.ipynb) · [Python版](代码/tutorials/bitsandbytes_代码注释.py)：另外3格，包含4/8bit互斥说明。
- 原版：[LLM.int8.ipynb](代码/tutorials/LLM.int8.ipynb) · [bitsandbytes.ipynb](代码/tutorials/bitsandbytes.ipynb)。两份原文件按下载字节保留，三张原图放在`代码/imgs`。
- 原代码带说明版：[主notebook](代码/tutorials/原代码_带说明.ipynb) · [辅助配置](代码/tutorials/bitsandbytes_原代码_带说明.ipynb)。
- [全部单元的编号说明](代码单元解释.md) · [实际验证和重做命令](复现/README.md)。

原矩阵示例只做浮点缩放，没有真正int8舍入/转换。注释明确标出此处，补充验证独立执行int8存储、int32累加及异常FP16支路。未下载OPT6.7B，未运行bitsandbytes CUDA量化与论文指标；随机小OPT只用于核对生成长度和参数遍历。

</details>
