# 第25讲：Llama 3 整体介绍

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从四词查表与选词，读到真实词表、参数量、GQA、SwiGLU、聊天模板与生成；论文原图和代码结果就在对应段落。
2. [主notebook Python注释代码](代码/tutorials/llama/llama3_0419_代码注释.py)。
3. [补充配置Python注释代码](代码/tutorials/llama/llama3_supp_data_config_代码注释.py)，再看 [完整FeedForward / ModelArgs注释](代码/tutorials/llama/Llama3核心源码_代码注释.py) 与 [完整Tokenizer / ChatFormat注释](代码/tutorials/llama/tokenizer_代码注释.py)。
4. 需要逐格输出时看 [主注释notebook](代码/tutorials/llama/代码注释.ipynb) 和 [补充注释notebook](代码/tutorials/llama/supp_代码注释.ipynb)。

Built with Meta Llama 3

<details>
<summary>原版、来源与实际验证</summary>

- 上游原件：[主notebook](代码/tutorials/llama/llama3_0419.ipynb)、[补充notebook](代码/tutorials/llama/llama3_supp_data_config.ipynb)。
- 原代码不改的说明版：[主](代码/tutorials/llama/原代码_带说明.ipynb)、[补充](代码/tutorials/llama/supp_原代码_带说明.ipynb)。
- [全部49格说明](代码单元解释.md)、[来源清单](来源清单.json)、[复现记录](复现/README.md)。
- [第三方许可证](代码/tutorials/llama/LICENSE)、[版权Notice](代码/tutorials/llama/Notice)。

原教程固定7fbc29824171fd07dc2c30fe922e61ebc018ad1d，Meta参考固定cf3ea595be1edfb019d21b14d8b215a39161c860。原字节及历史输出保留；原输出不是新运行证据。无原7B/8B权重与CUDA；完整配置在meta核对，随机小模型/分词/单rank FairScale在CPU验证，不评价语言质量。

原相对路径以本讲代码/tutorials/llama为工作目录。

</details>
