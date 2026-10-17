# 第 17 讲：RMSNorm 与 SwiGLU

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从向量、均方根讲到归一化与门控分支，配四张论文原截图、手算和代码。
2. [Python 文本阅读版](代码/tutorials/llama2_src_02_RMSNorm_SwiGLU_代码注释.py)：直接阅读原代码与中文注释。
3. [注释 notebook](代码/tutorials/代码注释.ipynb)：全部28格说明、20代码格及Markdown中的类。

课程只实现RMSNorm和SiLU；完整SwiGLU见[官方源码阅读补充](代码/SwiGLU_源码阅读.py)，正文第6–7节连接两者。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/llama2_src_02_RMSNorm_SwiGLU.ipynb)：固定Git blob，原字节及历史输出保留。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原单元不改，插入逐格说明。
- [按编号查单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。

原版与注释版各20格在真实CPU Torch中执行一致；完整未修改Meta模块的RMSNorm与FairScale SwiGLU前向、自动求导及独立公式验证通过。没有加载Llama权重或复现论文训练/速度。

</details>
