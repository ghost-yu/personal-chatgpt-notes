# 第 21 讲：GQA 与分组查询注意力

[返回总目录](../../README.md)

1. [中文讲解](解释.md)：从四个查询头、两组K/V讲起，配论文原图、手算例子、repeat_kv与完整Attention。
2. [核心 Python 代码阅读版](代码/tutorials/GQA核心源码_代码注释.py)：固定官方 `repeat_kv` 与完整 `Attention` 类，带详细中文注释。
3. [原 notebook 的详细中文注释](代码/tutorials/代码注释.ipynb)：全部5格，含Markdown源码的逐语句说明。

原notebook以配图和shape摘录介绍GQA。核心Python阅读版提供repeat_kv与完整Attention源码；中文讲解补充小张量与真实模块的运行例子。

<details>
<summary>原版、其他阅读版本与验证结果</summary>

- [原 notebook](代码/tutorials/llama2_src_grouped_query_attention.ipynb)、[原代码带说明版](代码/tutorials/原代码_带说明.ipynb)、[全部单元说明](代码单元解释.md)。
- [原课程 Python 阅读版](代码/tutorials/llama2_src_grouped_query_attention_代码注释.py)：仅对应原来的两格展示代码。
- [Markdown 六条语句注释版](代码/tutorials/Markdown源码_代码注释.py)。
- [实际运行范围与命令](复现/README.md)、[验证报告](复现/outputs/完整交付验证.json)、[来源清单](来源清单.json)。

原文和历史输出保留；共享方向、8个KV heads的适用型号与质量/速度表述在新增说明中校正。原图与notebook来自chunhuizhang/personal_chatgpt固定提交，完整官方实现来自Meta固定提交，见解释中的来源。

从本讲 `代码/tutorials` 启动图片展示notebook；本次真实模型验证使用随机小型Attention与单rankCPU/FairScale，不代表70B权重、论文uptraining或GPU性能复现。

</details>
