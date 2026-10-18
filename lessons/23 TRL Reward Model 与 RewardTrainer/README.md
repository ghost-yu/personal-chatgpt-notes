# 第 23 讲：TRL Reward Model 与 RewardTrainer

[返回课程总目录](../../README.md)

1. [中文讲解](解释.md)：从同一问题的两个回答讲起，手算分数差、概率、损失和梯度，再运行分词、padding、OPT/LoRA训练与保存；论文流程图和公式就地内嵌。
2. [原课程代码的中文注释](代码/tutorials/reward_model/trl_reward_trainer_代码注释.py)：全部 29 个代码格，直接看文本。
3. [注释 notebook](代码/tutorials/reward_model/代码注释.ipynb)：37 个原单元、逐格说明和原历史输出。

读到 `RewardTrainer` 时配合看 [完整训练器与 collator 的中文源码注释](代码/tutorials/reward_model/RewardTrainer核心源码_代码注释.py)。它解释原 notebook 没展开的成对 loss 与 padding。

<details>
<summary>原版对照、版本与实际验证</summary>

- [上游原 notebook](代码/tutorials/reward_model/trl_reward_trainer.ipynb)、[原代码带说明版](代码/tutorials/reward_model/原代码_带说明.ipynb)。
- [全部单元说明](代码单元解释.md)、[复现记录与重做命令](复现/README.md)。
- [来源清单](来源清单.json)、[实际验证输出](复现/outputs/runtime_verification.json)。

课程固定提交 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`，源码对照使用TRL0.8.6 / `e90e8d9…`。当前正文从InstructGPT原PDF截取Figure 2、奖励模型阶段放大图和公式1；原notebook中的两个外部图片URL继续保留。

当前四个正文例子使用CPU小OPT、两条自编偏好对、72个可训练参数，完成两次更新和保存重载。此前原格教学运行和4bit/完整数据的运行范围，见[复现记录](复现/README.md)。

</details>
