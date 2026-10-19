# 第 30 讲：InstructGPT reward modeling

[返回课程总目录](../../README.md)

1. [中文讲解](解释.md)：从两个回答开始，讲概率、损失、梯度与分组训练；五张论文原图就地解释。
2. [主 notebook Python 详细注释](代码/tutorials/reward_model/reward_model_Bradley_Terry_model_代码注释.py)：原公式与全部绘图/数值代码。
3. [补充 BFGS 估计代码注释](代码/tutorials/reward_model/bradley_terry_modeling_代码注释.py)：四队、五品牌的真实最大似然估计。
4. [偏好损失与真实小模型训练补充](代码/tutorials/reward_model/偏好损失_教学补充.py)、[完整 Llama 奖励打分类注释](代码/tutorials/reward_model/Llama奖励打分源码_代码注释.py)。
5. 需要逐格输出时看 [主注释 notebook](代码/tutorials/reward_model/代码注释.ipynb) 和 [补充注释 notebook](代码/tutorials/reward_model/bradley_terry_modeling_代码注释.ipynb)。

原版代码与历史输出保留；“log10与自然对数混用”“分差3以后梯度为0”等表述在新增解释中校正。正文给出六段可运行算例，详细结果见复现记录。

<details>
<summary>原版、来源与验证记录</summary>

- 原 [主 notebook](代码/tutorials/reward_model/reward_model_Bradley_Terry_model.ipynb)、[补充 notebook](代码/tutorials/reward_model/bradley_terry_modeling.ipynb)。
- 原代码带说明：[主](代码/tutorials/reward_model/原代码_带说明.ipynb)、[补充](代码/tutorials/reward_model/bradley_terry_modeling_原代码_带说明.ipynb)。
- [全部35格说明](代码单元解释.md)、[来源清单](来源清单.json)、[复现记录与命令](复现/README.md)。
- [原图](代码/tutorials/reward_model/imgs/quality-labels.png)、[HF源码许可证](代码/tutorials/reward_model/LICENSE.transformers)。

固定上游提交7fbc29824171fd07dc2c30fe922e61ebc018ad1d。论文为InstructGPT v1；实际补充模型用Transformers4.44.2，补充估计用SciPy1.13.1，不冒称原OpenAI训练工程。字幕仅保留本地。

</details>
