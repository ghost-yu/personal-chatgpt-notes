# 第 24 讲：PEFT LoRA merge、inject 与 SVD

[返回总目录](../../README.md)

1. [中文讲解](解释.md)：从注入、缩放和矩阵形状，读到多 adapter 合并与生成。
2. [主教程 Python 注释代码](代码/tutorials/peft/lora_merge_代码注释.py)：59 个原单元中的全部代码，含手写 SVD。
3. [主教程注释 notebook](代码/tutorials/peft/代码注释.ipynb)：逐格说明、代码内注释与原历史输出。

本讲还用到两个短 notebook，建议按“注入 → SVD 基础 → 主教程”的顺序读：

| 内容 | Python 注释代码 | 注释 notebook |
|---|---|---|
| 给 DummyModel 注入 LoRA | [inject](代码/tutorials/peft/lora_inject_代码注释.py) | [逐格看](代码/tutorials/peft/inject_代码注释.ipynb) |
| 低秩与满秩矩阵的 SVD | [svd basics](代码/tutorials/peft/lora_svd_basics_代码注释.py) | [逐格看](代码/tutorials/peft/svd_代码注释.ipynb) |

需要往库内部读时，看 [PEFT 核心源码及中文注释](代码/tutorials/peft/PEFT合并核心源码_代码注释.py)。该文件完整摘录相关方法，仍依赖原模块上下文，供阅读对照。

<details>
<summary>原版、全部单元索引与验证资料</summary>

- 原版：[inject](代码/tutorials/peft/lora_inject.ipynb)、[merge](代码/tutorials/peft/lora_merge.ipynb)、[SVD basics](代码/tutorials/peft/lora_svd_basics.ipynb)。
- 原代码不改的说明版：[inject](代码/tutorials/peft/inject_原代码_带说明.ipynb)、[merge](代码/tutorials/peft/原代码_带说明.ipynb)、[SVD](代码/tutorials/peft/svd_原代码_带说明.ipynb)。
- [98 个单元逐项解释](代码单元解释.md)、[实际运行与重做命令](复现/README.md)、[来源清单](来源清单.json)。
- 上游固定于 [7fbc298](https://github.com/chunhuizhang/personal_chatgpt/tree/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/peft)。

原大模型、4bit 加载及原任务质量没有复现。实际运行包括原 DummyModel、原 100×100 SVD 示例，以及明确替换加载/设备/续写预算的随机 CPU 小 Llama。PEFT 0.12.0 SVD 的重复缩放行为、未自动激活 merge 和手工代码对应关系均另外验证，不能把“与旧库一致”当成“符合预期的加权公式”。

</details>
