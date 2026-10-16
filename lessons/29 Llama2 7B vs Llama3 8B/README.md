# 第 29 讲：Llama2 7B vs Llama3 8B

[返回课程总目录](../../README.md)

1. [先看中文讲解](解释.md)：词表、GQA、SwiGLU、精确参数账与内存。
2. [原 notebook 的 Python 注释版](代码/tutorials/llama/llama2_7b_llama3_8b_代码注释.py)：全部原代码及 Markdown 摘录的中文解释。
3. [完整 HF 核心实现注释版](代码/tutorials/llama/Llama核心实现_代码注释.py)：Attention、SDPA、MLP、RMSNorm 和三个辅助函数。
4. [注释 notebook](代码/tutorials/llama/代码注释.ipynb)：用 GitHub 或 Jupyter 逐格查看；历史输出保留。

原加载格会请求真正的7B/8B权重。日常直接阅读代码；本次验证使用完整尺寸 meta 结构和明确的随机小模型，本讲未运行原权重推理。

<details>
<summary>原版、来源与验证资料</summary>

- [原 notebook](代码/tutorials/llama/llama2_7b_llama3_8b.ipynb)、[原代码带说明版](代码/tutorials/llama/原代码_带说明.ipynb)。
- [全部24格说明](代码单元解释.md)、[实际复现记录与命令](复现/README.md)。
- [来源清单](来源清单.json)、[核心实现 Apache-2.0 许可证](代码/tutorials/llama/LICENSE.transformers)。
- [第25讲分词与模型整体接口](../25%20Llama%203%20introduction/README.md)、[第21讲GQA](../21%20GQA%20与分组查询注意力/README.md)。

课程原件固定到 chunhuizhang/personal_chatgpt 的 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。核心HF源码固定到已安装并核对RECORD的 Transformers 4.44.2；讲师精确版本未知。原Markdown部分乱码保留，新增中文说明解释语义。字幕只保留本地。

</details>
