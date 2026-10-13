# 第 01 讲：LLaMA introduction

先打开[详细中文注释 notebook](代码/tutorials/代码注释.ipynb)，查看代码、注释和逐格说明。想直接看文本代码，可打开[Python 阅读版](代码/tutorials/01_llama_intro_代码注释.py)，其中没有JSON和历史图片编码。

| 文件 | 用途 |
|---|---|
| [原 notebook](代码/tutorials/01_llama_intro.ipynb) | 从上游固定提交下载，原样保留9个单元和历史输出 |
| [代码注释 notebook](代码/tutorials/代码注释.ipynb) | 两格Python和六条启动命令增加中文注释，覆盖全部原单元 |
| [原代码带说明 notebook](代码/tutorials/原代码_带说明.ipynb) | 原9格内容不变，在每格前插入说明 |
| [Python 阅读版](代码/tutorials/01_llama_intro_代码注释.py) | `# %%` 分格，Markdown/命令以注释呈现，不启动模型 |
| [代码单元解释](代码单元解释.md) | API、参数、返回值、路径、数据流及掌握等级 |
| [论文与推理代码解释](解释.md) | LLaMA公式、shape、模型家族和外部官方推理调用链 |
| [复现结果](复现/README.md) | 阅读结果表格，查看运行命令和未复现范围 |

来源为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/01_llama_intro.ipynb)，提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。新增中文说明、注释与验证脚本由本仓库编写。

本讲的原 notebook 只包含两行 Python 展示代码，没有实现 Attention 或训练算法。Markdown中的六条命令也完整注释，但由于缺少配套checkpoint未运行。运行两个展示单元时，请从本讲 `代码/tutorials` 目录启动内核；配套图片在 `代码/imgs`。
