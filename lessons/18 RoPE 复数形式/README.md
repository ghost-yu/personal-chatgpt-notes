# 第 18 讲：RoPE 复数形式

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：复数旋转、频率、shape与相对点积。
2. [主讲详细代码注释](代码/tutorials/代码注释.ipynb)：全部29格说明、16代码格。
3. [主讲 Python 文本阅读版](代码/tutorials/llama2_src_03_RoPE_01_complex_position_encoding_代码注释.py)：直接阅读代码与注释。

本讲另有RoPE总览。主讲第二个位置表类依赖全局d_model，原语句保留，影响与修正见讲解。

<details>
<summary>RoPE总览补充、原版对照与验证资料</summary>

- [总览注释 notebook](代码/tutorials/代码注释_RoPE总览.ipynb)及[总览 Python 阅读版](代码/tutorials/llama2_src_03_RoPE_代码注释.py)：另外29格/19代码格，含原图与频率错误例。
- [原配图代码转录与注释](代码/配图_RoPE源码阅读.py)：半区配对的rotate_half与apply_rotary_pos_emb，截图对应HF版本未找到。
- [主讲原 notebook](代码/tutorials/llama2_src_03_RoPE_01_complex_position_encoding.ipynb)与[总览原 notebook](代码/tutorials/llama2_src_03_RoPE.ipynb)。
- [主讲原代码带说明](代码/tutorials/原代码_带说明.ipynb)与[总览原代码带说明](代码/tutorials/原代码_带说明_RoPE总览.ipynb)：原单元不改。
- [全部58格单元说明](代码单元解释.md)。
- [实际验证、环境及命令](复现/README.md)。

固定上游提交7fbc29824171fd07dc2c30fe922e61ebc018ad1d；两原版按字节保留，四派生版各59格。原/注释各35格真实CPU执行一致；未运行完整Llama或RoFormer训练实验。

</details>
