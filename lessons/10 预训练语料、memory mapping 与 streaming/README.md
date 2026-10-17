# 第 10 讲：预训练语料、memory mapping 与 streaming

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从磁盘/内存、6行小文件讲起，区分三种map；C4/Pile原论文表直接截图，中文含义就在旁边。
2. [纯文本 Python 阅读版](代码/tutorials/pretrain_corpus_大模型预训练语料_代码注释.py)：阅读原代码及中文注释，不必看 JSON。
3. [原 notebook 的详细代码注释](代码/tutorials/代码注释.ipynb)：全部 13 格说明，6 个代码格中文注释，四张原图保留。

正文已直接放入本轮6行小算例。需要第二遍看更完整的历史补充时，打开[数据读取教学补充.py](代码/数据读取_教学补充.py)。原 notebook 仅展示语料图，没有保存字幕中的加载/测速实现；补充单独演示 JSONL、Arrow/mmap、streaming map、GPT-2 分词/padding 和 PyTorch DataLoader，包括 2 worker。

<details>
<summary>原版对照、单元编号与验证资料</summary>

- [上游原 notebook](代码/tutorials/pretrain_corpus_大模型预训练语料.ipynb)：原下载字节、元数据与历史输出保留。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原代码不改，只插说明。
- [全部 13 格编号说明](代码单元解释.md)。
- [实际运行范围、环境与重做命令](复现/README.md)。

原图路径相对 `代码/tutorials`，从该目录启动 Jupyter 内核。Python 版以阅读为主；普通终端运行不会自动渲染 Image。

本轮正文三个6行小例实际执行通过；原/注释各6格及5行补充属于保留的历史验证，不代表课程约 20 GB 数据或完整 C4/Pile/T5 实验复现。

</details>
