# 第28讲：从 RoPE 到 CoPE

[返回课程总目录](../../README.md)

1. [中文讲解](解释.md)：位置分类、RoPE布局、CoPE公式/shape及梯度。
2. [原notebook Python注释版](代码/tutorials/position_encoding/rope_cope_代码注释.py)：按原40格顺序阅读。
3. [完整CoPE与SelfAttn注释代码](代码/tutorials/position_encoding/CoPE核心源码_代码注释.py)：原实现位于Markdown39，这里加必要imports，可直接导入。
4. [注释notebook](代码/tutorials/position_encoding/代码注释.ipynb)：用GitHub/Jupyter逐格查看。

两版各25代码格原样执行，完整CoPE在真实PyTorch下完成前向/梯度/5步小参数更新验证；没有复现论文完整任务或语言模型质量。

<details>
<summary>原版、逐格对照与验证资料</summary>

- [上游原notebook](代码/tutorials/position_encoding/rope_cope.ipynb)。
- [原代码带说明版](代码/tutorials/position_encoding/原代码_带说明.ipynb)。
- [全部40格原文与说明](代码单元解释.md)。
- [来源清单](来源清单.json)。
- [实际复现说明](复现/README.md)与[验证报告](复现/outputs/delivery_verification.json)。

源为[课程固定Git版本](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/position_encoding/rope_cope.ipynb)，CoPE两个类对照[论文v2附录B](https://arxiv.org/pdf/2405.18719v2)。保留原作者归属，新增中文讲解、注释和独立诊断。未找到并锁定独立官方训练工程，不冒称该文件是某个官方Git仓库下载源码。

六张本地图位于代码/imgs，保持原../../imgs相对路径；远程BERT图只有原URL，内容未验证。字幕只留本地，论文PDF在线读取成功但本机curl下载TLS失败，未虚构本地PDF。

</details>
