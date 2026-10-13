# 第 04 讲：PEFT 怎样把 LoRA 注入原模型

先打开[详细中文注释 notebook](代码/tutorials/代码注释.ipynb)。要直接阅读代码，打开[Python 阅读版](代码/tutorials/04_PEFT_lora_代码注释.py)，其中没有JSON和历史输出，六段Markdown源码也完整保留为注释。

| 文件 | 用途 |
|---|---|
| [原 notebook](代码/tutorials/04_PEFT_lora.ipynb) | 固定提交下载，保留18格、历史输出及原空代码格 |
| [详细中文注释 notebook](代码/tutorials/代码注释.ipynb) | 8个Python格、6个Markdown源码块详注，逐格插入说明，共37格 |
| [原代码带说明 notebook](代码/tutorials/原代码_带说明.ipynb) | 原单元不改，仅插入逐格说明，共37格 |
| [Python 阅读版](代码/tutorials/04_PEFT_lora_代码注释.py) | `# %%` 分格，无JSON；Markdown节选保留为注释，原缺陷明确标注 |
| [全部单元解释](代码单元解释.md) | 三类type、配置、默认目标、匹配、引用、替换和shape |
| [主解释](解释.md) | 官方调用链、源码对应、低秩前向、冻结与合并 |
| [CPU验证脚本](复现/scripts/verify_injection.py) | 直接导入固定官方PEFT v0.3.0，比较原版与注释版的实际行为 |
| [实际结果](复现/README.md) | 表格、环境、命令、来源和未复现范围 |

来源为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/04_PEFT_lora.ipynb)，固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。新增中文解释、注释与验证脚本由本仓库编写。原notebook未引用配套图片文件，本讲不需要额外PNG。

实际验证直接使用[官方PEFT v0.3.0固定源码](https://github.com/huggingface/peft/tree/7120c9a2636f93c8db3fc4ae466e02f338ecead8)及CPU随机小型BLOOM；老师的0.4.0.dev0没有SHA，仍未精确重建。原版和注释版的模块匹配、真实注入、冻结、前向/梯度、disable及merge/unmerge结果一致。未运行原7B量化加载，也没有训练模型。Markdown7缺默认字段的dataclass节选仍按预期报错，注释版只标明、不静默修复。
