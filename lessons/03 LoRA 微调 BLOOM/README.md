# 第 03 讲：LoRA 微调 BLOOM

先打开[详细中文注释 notebook](代码/tutorials/代码注释.ipynb)。要直接阅读代码，打开[Python 阅读版](代码/tutorials/03_lora_practice_代码注释.py)，无需看JSON；模型加载、冻结、LoRA注入、数据拼接、分词、训练和生成均有注释。

| 文件 | 用途 |
|---|---|
| [原 notebook](代码/tutorials/03_lora_practice.ipynb) | 固定提交下载，原样保留47格、历史输出和原安装命令 |
| [详细代码注释 notebook](代码/tutorials/代码注释.ipynb) | 37个代码格详注，原语句不改，逐格补充说明，共95格 |
| [原代码带说明 notebook](代码/tutorials/原代码_带说明.ipynb) | 原47格逐字保留，在每格前插入说明，共95格 |
| [Python 阅读版](代码/tutorials/03_lora_practice_代码注释.py) | `# %%` 分格，无JSON/历史输出；五条IPython指令保留为注释 |
| [全部单元解释](代码单元解释.md) | API、shape、状态变化、字段转换与掌握等级 |
| [主解释](解释.md) | LoRA公式、调用链、CLM loss与训练/推理边界 |
| [CPU教学验证脚本](复现/scripts/verify_pipeline.py) | 真正调用PEFT、BLOOM、Datasets、collator和Trainer，模型/数据明确另设 |
| [验证结果](复现/README.md) | 表格、环境、重做命令和未复现范围 |

来源为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/03_lora_practice.ipynb)，固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。新增中文解释、注释与验证脚本由本仓库编写。原notebook未引用配套图片文件；本讲不需要下载额外PNG。

Python文件用于阅读，原notebook仍保留安装与magic语法，建议在notebook逐格核对环境后执行。当前验证用CPU随机小型BLOOM、新编5条数据、自建词表和2步Trainer训练；没有运行原7B量化加载、FP16 200步训练或两个CUDA生成示例。原历史输出不作为本次运行结果。
