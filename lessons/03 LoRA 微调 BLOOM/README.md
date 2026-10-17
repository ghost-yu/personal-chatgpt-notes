# 第 03 讲：LoRA 微调 BLOOM

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：沿着一条样本理解编号、目标、误差和LoRA更新；论文原图直接内嵌在讲解旁。
2. [纯文本 Python 阅读版](代码/tutorials/03_lora_practice_代码注释.py)：直接阅读代码与注释。
3. [需要输出时看注释 notebook](代码/tutorials/代码注释.ipynb)：原课程代码旁有中文注释，每格前有说明。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/03_lora_practice.ipynb)：保留原单元及历史输出。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原代码不改，只插入说明。
- [按编号查单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。

来源为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/03_lora_practice.ipynb)，固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`。新增中文解释、注释与验证脚本由本仓库编写。原notebook未引用配套图片文件；本轮从LoRA v2原论文直接截图Figure1，放在解释正文的训练分支讲解旁。

Python文件用于阅读，原notebook仍保留安装与magic语法，建议在notebook逐格核对环境后执行。当前验证用CPU随机小型BLOOM、新编5条数据、自建词表和2步Trainer训练；没有运行原7B量化加载、FP16 200步训练或两个CUDA生成示例。原历史输出不作为本次运行结果。

</details>
