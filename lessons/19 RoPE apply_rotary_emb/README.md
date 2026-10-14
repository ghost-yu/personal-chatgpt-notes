# 第19讲：apply_rotary_emb 与相对位置

[返回总目录](../../README.md)

1. [中文讲解](解释.md)：相邻通道旋转、shape、广播及相对点积。
2. [原代码详细注释 notebook](代码/tutorials/代码注释.ipynb)：27格说明、15代码格，含注释格与空格。
3. [Python 文本阅读版](代码/tutorials/llama2_src_03_RoPE_apply_rotary_emb_代码注释.py)：直接阅读原语句及中文注释。

原notebook只定义核心函数，没有调用它，也缺helper。原语句保留，官方补足与真实运行结果单独说明。

<details>
<summary>官方函数补充、原版对照与验证资料</summary>

- [三个官方RoPE函数的中文源码阅读](代码/RoPE_官方函数阅读.py)：含原课程缺少的reshape_for_broadcast。
- [上游原 notebook](代码/tutorials/llama2_src_03_RoPE_apply_rotary_emb.ipynb)与[原代码带说明](代码/tutorials/原代码_带说明.ipynb)。
- [全部27格单元说明](代码单元解释.md)：保留早期15代码格详解，补齐公式、模型表与Markdown说明。
- [实际验证、依赖和重做命令](复现/README.md)。

原tokenizer.model也已按字节保留在代码/tutorials，内核从该目录启动即可执行原分词格。它是分词器文件，不是Llama神经网络权重。两个派生版各55格，原单元及历史输出保留。没有整模型训练、推理性能或论文实验复现声明。

</details>
