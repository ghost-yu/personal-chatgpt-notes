# 第 22 讲：Llama generate 完整流程

[返回课程总目录](../../README.md)

1. [中文讲解](解释.md)：按一次生成的真实顺序理解位置、缓存、采样与停止。
2. [完整生成源码的中文注释](代码/tutorials/generate核心源码_代码注释.py)：重点读这里，包含完整 `generate`、`text_completion` 和 `sample_top_p`。
3. [注释 notebook](代码/tutorials/代码注释.ipynb)：保留原课程 17 格与历史输出，逐格增加说明。

原 notebook 只有四格分词代码，生成循环主要是 Markdown 摘录。完整官方方法作为明确补充一起注释；只看原四格无法看到完整生成逻辑。

<details>
<summary>原版对照、其他代码与验证证据</summary>

- [原 notebook](代码/tutorials/llama2_src_generate.ipynb)与[原代码带说明版](代码/tutorials/原代码_带说明.ipynb)。
- [原四格分词代码的 Python 阅读版](代码/tutorials/llama2_src_generate_代码注释.py)。
- [Markdown 四段源码注释](代码/tutorials/Markdown片段_代码注释.py)：不同片段需要不同上下文，不能直接拼成生成器。
- [全部单元说明](代码单元解释.md)、[复现范围与命令](复现/README.md)。
- [来源清单](来源清单.json)、[实际验证结果](复现/outputs/runtime_verification.json)。

上游课程固定为 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`；Meta 源码固定为 `7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e`，版本链接在讲解中。完整源码阅读文件保留 Meta 版权与 Llama 2 许可声明；中文注释为本项目补充。

CPU 教学运行使用真实小 Transformer 和 FairScale/Gloo，未加载原 Llama 权重。旧版新增 token 的 `logprobs=-0` 是实际复现的缺陷；教学修正版单独验证，原件未改。

</details>
