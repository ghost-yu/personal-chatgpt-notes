# 第05讲：FP16与BF16的位表示和代码

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：理解位字段、正常/次正规数、范围与精度。
2. [再看详细代码注释](代码/tutorials/代码注释.ipynb)：8个代码格详注，全部27格有中文说明。
3. [纯文本 Python 阅读版](代码/tutorials/05_float16_bf16_代码注释.py)：直接阅读代码，不看JSON。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/05_float16_bf16.ipynb)：保留全部27格和作者历史输出。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原代码不改，仅插入说明，共55格。
- [按编号查全部单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。

来源为 [chunhuizhang/personal_chatgpt 固定提交](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/05_float16_bf16.ipynb)。原notebook和两张PNG直接下载、校验哈希、原样保存；新增中文解释、注释与验证脚本由本仓库编写。

从本讲`代码/tutorials`启动notebook内核，配图相对路径才正确。双问号是IPython语法，Python阅读版中保留为注释。原CUDA设备格在当前CPU包中按预期失败；原版与注释版各8格实际尝试，7格成功、1格记录相同失败。CPU补充验证了真实tensor的位模式、极值、舍入、溢出、下溢和存储字节；未验证GPU/TPU算术、训练或性能。

</details>
