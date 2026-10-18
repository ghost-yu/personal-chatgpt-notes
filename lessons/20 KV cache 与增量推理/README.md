# 第 20 讲：KV cache 与增量推理

[返回总目录](../../README.md)

1. [中文讲解](解释.md)：从三个token讲QKV、缓存槽位、prefill/decode与矩形mask；论文原图和公式在对应段落内。
2. [Python 阅读版](代码/tutorials/llama2_src_cache_kv_代码注释.py)：直接阅读原课程代码与中文注释。
3. [详细中文代码注释](代码/tutorials/代码注释.ipynb)：全部 25 个原单元的说明，包含 Markdown 中的 Attention 摘录。

<details>
<summary>原版对照、完整 Attention 类与验证结果</summary>

- [原 notebook](代码/tutorials/llama2_src_cache_kv.ipynb)、[原代码带说明版](代码/tutorials/原代码_带说明.ipynb)、[全部单元索引](代码单元解释.md)。
- [课程 Attention 摘录注释](代码/tutorials/Attention摘录_代码注释.py)：不完整类，仅供阅读。
- [固定官方完整 Attention 类注释](代码/tutorials/Attention官方源码_代码注释.py)：补齐摘录略去的初始化与最终运算；需要原模块的依赖，不能单文件运行。
- [复现范围与命令](复现/README.md)、[实际验证报告](复现/outputs/完整交付验证.json)、[来源哈希](来源清单.json)。

原 notebook 与图片来自 chunhuizhang/personal_chatgpt，固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。原单元、元数据及历史输出保留。注释版只增加注释，运行教学验证时的显式变体另行记录。

从本讲 `代码/tutorials` 启动 notebook。原代码含远程 GPT-2 加载、代理环境设置和 1000 新 token ×10 次计时；教学验证使用本地随机小模型，结果不代表预训练模型质量或 GPU 速度。

</details>
