# 第09讲：BPE、GPT-2 tokenizer与词表训练

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从三个短词手算BPE，再讲UTF-8、ID和新词表；论文Figure1、Algorithm1原截图及课程流程图就地内嵌。
2. 纯文本阅读：[基础 Python版](代码/tutorials/tokenizer_basics_代码注释.py) → [训练 Python版](代码/tutorials/gpt2_train_tokenizer_代码注释.py)，不用看JSON。
3. 代码注释按这个顺序读：[基础与GPT-2分词](代码/tutorials/代码注释.ipynb) → [训练新tokenizer](代码/tutorials/gpt2_train_tokenizer_代码注释.ipynb)。两份主文件共66格，全部有说明和代码内注释。

<details>
<summary>原版对照、辅助文件与验证资料</summary>

- 三份原notebook：[tokenizer_basics](代码/tutorials/tokenizer_basics.ipynb) · [gpt2_train_tokenizer](代码/tutorials/gpt2_train_tokenizer.ipynb) · [gpt2辅助](代码/tutorials/gpt2.ipynb)。原下载字节、元数据和历史输出保留。
- 辅助6格：[gpt2注释版](代码/tutorials/gpt2_代码注释.ipynb) · [Python版](代码/tutorials/gpt2_代码注释.py)。
- 只插说明的对照版：[基础](代码/tutorials/原代码_带说明.ipynb) · [训练](代码/tutorials/gpt2_train_tokenizer_原代码_带说明.ipynb) · [辅助](代码/tutorials/gpt2_原代码_带说明.ipynb)。
- [全部72格编号说明](代码单元解释.md) · [实测范围、环境和重做命令](复现/README.md)。

原basics22有SyntaxError，注释notebook保留错误；Python阅读版把该格作为原文注释，修正版独立记录。两条Linux shell也标出适用环境。实际训练使用固定GPT-2 tokenizer和小型JSONL语料，未复现课程10/20万代码数据，未上传讲师的Hub组织或训练语言模型。

</details>
