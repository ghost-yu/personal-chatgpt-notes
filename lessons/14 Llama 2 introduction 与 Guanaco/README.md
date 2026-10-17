# 第 14 讲：Llama 2 introduction 与 Guanaco

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从模型/文字编号讲起，直接内嵌Llama 2训练阶段图、配置表和LoRA原图；沿模板→生成→对话监督→训练/保存解释，四块正文代码实际执行，旧完整讲解折叠保留。
2. [Python 文本阅读版](代码/tutorials/llama2_introduction_代码注释.py)：不用阅读JSON。
3. [详细代码注释](代码/tutorials/代码注释.ipynb)：全部23格说明、16代码格及Markdown训练命令注释。

原代理、远程7B和FP16配置保留。CPU验证有明确改动，随机小模型结果不作为原7B聊天能力或4-bit训练结果。

<details>
<summary>原版对照、单元索引与验证记录</summary>

- [固定上游原 notebook](代码/tutorials/llama2_introduction.ipynb)：原字节与历史输出保留。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原单元不改，只插说明。
- [全部单元说明](代码单元解释.md)。
- [验证范围、环境与重做命令](复现/README.md)。
- [分词与监督补充](代码/分词与监督_教学补充.py)。
- `来源清单.json` 用于追溯来源，日常阅读无需打开。

</details>
