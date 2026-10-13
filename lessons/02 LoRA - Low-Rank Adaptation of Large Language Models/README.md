# 第 02 讲：LoRA 原理与矩阵代码

先看[详细中文注释 notebook](代码/tutorials/代码注释.ipynb)。想直接阅读代码而不看JSON，打开[Python 阅读版](代码/tutorials/02_lora_baics_代码注释.py)，其中完整保留中文说明，Markdown伪代码以注释呈现。

| 文件 | 内容 |
|---|---|
| [原 notebook](代码/tutorials/02_lora_baics.ipynb) | 固定提交下载原文件，保留16格和作者历史输出 |
| [详细代码注释 notebook](代码/tutorials/代码注释.ipynb) | 7个Python格、Markdown13的15条伪代码语句均有详细中文注释 |
| [原代码带说明 notebook](代码/tutorials/原代码_带说明.ipynb) | 原单元不改，在每格前加入说明，共33格 |
| [Python 阅读版](代码/tutorials/02_lora_baics_代码注释.py) | 无JSON或图片编码，`# %%` 分格，伪代码保留为注释 |
| [代码单元解释](代码单元解释.md) | 逐格用途、API、变量shape、图解、论文对应与掌握等级 |
| [主解释](解释.md) | LoRA公式、参数量、初始化、缩放、梯度和权重合并 |
| [可运行的CPU教学示例](复现/scripts/verify_lora.py) | 自编补充，验证前向/合并/梯度/冻结与21次小型SGD更新 |
| [实际验证结果](复现/README.md) | 结果表格、重做命令、环境和未复现边界 |

来源为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/02_lora_baics.ipynb)，固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。`baics` 是原文件名，不改写成 `basics`。新增中文说明、注释与验证脚本由本仓库编写。

运行图片展示时，从本讲 `代码/tutorials` 启动内核，图片位于 `代码/imgs`。原代码的7格实际只是导入Image和展示图片；LoRA训练不是这些单元的运行结果。Markdown13有未导入torch/nn/math、W占位、alpha未定义和return缩进错误，本注释版保留并标明；可运行教学例另列，未运行BLOOM大模型或论文评测。
