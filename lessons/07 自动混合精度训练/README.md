# 第07讲：CNN与自动混合精度训练

[返回总目录](../../README.md)

1. [先看中文讲解](解释.md)：从图片、误差与一次更新讲起，再区分autocast和梯度缩放；论文原图已直接截图内嵌。
2. [纯文本 Python 阅读版](代码/tutorials/07_fp16_mixed_precision_training_代码注释.py)：直接阅读代码，不看JSON。
3. [需要输出时看注释 notebook](代码/tutorials/代码注释.ipynb)：12个代码格及Markdown1示例详注，全部19格有说明。

<details>
<summary>原版对照、单元索引与验证资料</summary>

- [上游原 notebook](代码/tutorials/07_fp16_mixed_precision_training.ipynb)：原大CNN、原配置、历史输出保留。
- [原代码带说明版](代码/tutorials/原代码_带说明.ipynb)：原单元不改，仅插入说明，共39格。
- [按编号查全部单元说明](代码单元解释.md)。
- [实际验证、环境与重做命令](复现/README.md)。

来源为 [chunhuizhang/personal_chatgpt 固定提交](https://github.com/chunhuizhang/personal_chatgpt/blob/7fbc29824171fd07dc2c30fe922e61ebc018ad1d/tutorials/07_fp16_mixed_precision_training.ipynb)。原notebook直接下载、核对哈希，原notebook未引用额外PNG；本轮补充Mixed Precision Training v3第3页Figure1原PDF截图，在正文解释前向/反向、FP16计算和FP32更新。中文解释、注释及验证脚本由本仓库编写。

原通道5120/10240十分大，完整运行会下载MNIST并分配大量内存。本次只在meta设备核对原大CNN；另用明确改小的8/16通道模型和8个合成样本验证原训练/评估函数及真实CPU FP16 AMP。未复现RTX3090的128→256显存实验，不把合成准确率当成MNIST结果。

</details>
