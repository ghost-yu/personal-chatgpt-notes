# 第28讲：从 RoPE 到 CoPE

[返回课程总目录](../../README.md)

1. [中文讲解](解释.md)：从三位置注意力讲到句子计数、小数插值与完整CoPE类，五张论文原图/原式/原表就地解释。
2. [原notebook Python注释版](代码/tutorials/position_encoding/rope_cope_代码注释.py)：按原40格顺序阅读。
3. [完整CoPE与SelfAttn注释代码](代码/tutorials/position_encoding/CoPE核心源码_代码注释.py)：原实现位于Markdown39，这里加必要imports，可直接导入。
4. [注释notebook](代码/tutorials/position_encoding/代码注释.ipynb)：用GitHub/Jupyter逐格查看。

正文六段CPU算例可顺序阅读，完整CoPE位置表用5次Adam展示更新；具体输出与环境见复现说明。

<details>
<summary>原版、逐格对照与验证资料</summary>

- [上游原notebook](代码/tutorials/position_encoding/rope_cope.ipynb)。
- [原代码带说明版](代码/tutorials/position_encoding/原代码_带说明.ipynb)。
- [全部40格原文与说明](代码单元解释.md)。
- [来源清单](来源清单.json)。
- [实际复现说明](复现/README.md)、[正文算例报告](复现/outputs/基础讲解验证.json)与[历史验证报告](复现/outputs/delivery_verification.json)。

源为[课程固定Git版本](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/position_encoding/rope_cope.ipynb)，CoPE两个类对照[论文v2附录B](https://arxiv.org/pdf/2405.18719v2)。保留原作者归属，新增中文讲解、注释和独立诊断。本讲实现依据为论文附录与课程原notebook，独立官方训练工程仍未锁定。

六张本地图位于代码/imgs，保持原../../imgs相对路径；远程BERT图只有原URL，内容未验证。字幕与完整PDF留在本地；2026-10-18已下载固定v2原PDF，相关截图直接放入正文。

</details>
