# 第26讲：GPT-4o tokenizer 与 o200k_base

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：字节BPE、压缩率、词表洞位和观察边界。
2. [主notebook详细Python注释](代码/tutorials/tokenizer/tokenizer_gpt4o_代码注释.py)。
3. [配套数字/空格接口注释](代码/tutorials/tokenizer/tokenizer_supply_代码注释.py) 与 [tiktoken完整核心对象注释](代码/tutorials/tokenizer/tiktoken核心源码_代码注释.py)。
4. 需要逐格输出时看 [主注释notebook](代码/tutorials/tokenizer/代码注释.ipynb) 和 [补充注释notebook](代码/tutorials/tokenizer/supp_代码注释.ipynb)。

<details>
<summary>原版、数据与验证证据</summary>

- 原件：[主](代码/tutorials/tokenizer/tokenizer_gpt4o.ipynb)、[补充](代码/tutorials/tokenizer/tokenizer_supply.ipynb)。
- 原代码不改、插入说明：[主](代码/tutorials/tokenizer/原代码_带说明.ipynb)、[补充](代码/tutorials/tokenizer/supp_原代码_带说明.ipynb)。
- [全部33格说明](代码单元解释.md)、[来源清单](来源清单.json)、[复现记录](复现/README.md)。
- [独立安全诊断](复现/scripts/tokenizer_diagnostics.py)：精确处理洞位、UTF8字节与特殊ID。
- 原《西游记》文本和两套完整校验词表随代码发布；[tiktoken MIT许可](代码/tutorials/tokenizer/LICENSE-tiktoken)。

固定上游7fbc29824171fd07dc2c30fe922e61ebc018ad1d。原课扫描会触发并捕获无效ID的Rust panic，历史输出不改；请先读复现说明。本次不调用GPT-4o、不训练模型、不验证原训练因果推测。原相对路径以代码/tutorials/tokenizer为工作目录。

</details>
