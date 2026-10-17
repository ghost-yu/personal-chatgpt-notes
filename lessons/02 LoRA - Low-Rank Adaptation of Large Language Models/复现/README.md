# 第02讲代码注释与实际验证

复现状态：**部分成功**。全16格说明、7/7可执行单元中文注释，以及Markdown13的15条有意义伪代码语句注释已完成。真实展示成功；原伪代码按预期报函数外return，未运行BLOOM或论文评估。教学LoRA验证成功是更小的验证范围。

## 来源与文件

[来源清单](../来源清单.json)固定教程提交及原文件/图片哈希。`download_original.py`直接下载原notebook与六张PNG，并逐文件核对已登记的SHA-256。原版、两种33格说明/注释notebook、[Python阅读版](../代码/tutorials/02_lora_baics_代码注释.py)和配套图片均直接发布在本仓库 `代码/tutorials` 与 `代码/imgs`。`build_annotated.py`保留原16格顺序、元数据和历史输出，注释版只额外增加中文注释。

`verify_annotations.py`对原版及两份生成版核对schema、原单元/输出、注释新增行、AST，并实际执行所有7格；还核对Python阅读版AST、插入说明中的相对链接。每格执行环境一致，六张PNG字节哈希一致；单元4/7/10宽600，8宽500，14/15没有显式宽度。伪代码能解析AST，但compile失败，两项不是同一验证。

`verify_lora.py`是新编的最小CPU教学实现，未冒充修复后的原notebook。`forward(x, weight, factor_a, factor_b, scale)`是独立函数、无默认参数；输入分别为最后一维k的浮点Tensor、`[k,d]`、`[k,r]`、`[r,d]`及float缩放，返回最后一维d的Tensor。不原地修改输入，梯度由后面的loss.backward计算。

## 环境和命令

展示验证使用现有Python3.7.6、IPython7.12.0、nbformat5.0.4；教学Tensor验证使用独立Python3.12.14、torch2.5.1+cpu、float64。两份requirements对应两个环境，不能把旧IPython环境当成CPU torch环境。无需GPU、checkpoint、训练数据或API key；本次没有测量RAM/磁盘，也没有据此启动大模型。

以下PowerShell命令从公开仓库根目录执行；其他机器需替换Python路径。下载脚本只下载本讲固定文件，不下载模型；生成、展示验证及CPU教学验证无需联网。

```powershell
& 'D:/anaconda/python.exe' './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/scripts/download_original.py'
& 'D:/anaconda/python.exe' './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/scripts/build_annotated.py' --notebook './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/代码/tutorials/02_lora_baics.ipynb' --output-dir './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/代码/tutorials' --record './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/outputs/code_unit_coverage.json'
& 'D:/anaconda/python.exe' './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/scripts/verify_annotations.py' --directory './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/代码/tutorials' --record './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/outputs/annotation_verification.json'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/scripts/verify_lora.py' --record './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/outputs/lora_torch_verification.json'
```

## 实际结果与范围

| 检查 | 结果 |
|---|---|
| 原文件下载 | 1个notebook、6张PNG与来源清单哈希全部相同，原文件字节不变 |
| 注释覆盖 | 16格均说明；7个Python格及Markdown13的15条语句详细注释 |
| 原单元/历史输出、schema、AST | 两种生成版全部通过；Python阅读版AST与原7格拼接一致 |
| 图片展示 | 三份notebook的7格均实际执行，图片字节/哈希/展示宽度一致 |
| 原伪代码 | compile仍报 `'return' outside function`，没有把原文缺陷静默修复 |
| CPU教学LoRA | 前向、合并、转置方向、解析梯度、冻结和小型SGD验证通过，详见下面记录 |
| 未复现范围 | 原伪代码训练、BLOOM加载/微调和论文任务指标 |

下面的JSON链接只是补充记录；直接读表格即可了解结果。

- [覆盖记录](outputs/code_unit_coverage.json)：16格说明、7格代码注释、Markdown13的15条语句注释，两份33格notebook。
- [展示/注释核对](outputs/annotation_verification.json)：原单元、历史输出、注释前后AST、schema及7格真实执行通过；伪代码编译失败为`'return' outside function`，保留原缺陷。
- [真实CPU PyTorch记录](outputs/lora_torch_verification.json)：手算例输出[16,22]；多维输出shape为[2,3,5]；显式增量/连续小乘法/权重合并/Linear转置方向一致，合并误差3.55e-15；autograd与手推梯度误差3.55e-15。
- 零B初始化的A梯度为0，B梯度范数4.1773；更新B后A梯度范数1.1769；基权重始终无梯度且未改变。21次小型SGD更新使单个教学样本MSE从4降到约1.11e-14；两因子均为0的反例出现两边零梯度。

单样本拟合仅证明这个脚本的数据流和梯度正常，不能证明泛化、LoRA最优rank、性能提升、BLOOM显存或论文实验成立。原伪代码还缺torch/nn/math导入、真实W、alpha与冻结/训练路径；仅修return缩进仍不足以运行。完整文件入口见[本讲目录](../README.md)，运行图片展示时把内核工作目录设为本讲 `代码/tutorials`。

## 2026-10-17：基础讲解与原论文截图

本轮按“先手算、再读图与代码”的顺序改写笔记。实际取得arXiv:2106.09685v2的26页PDF，截取第1页Figure 1和第4页公式(3)，图片直接放在相应段落。使用pypdfium2按4倍分辨率渲染后裁剪，没有重绘；已逐张核对箭头、A/B标签、公式及编号完整。教程原有六张图片也已检查并改为本地内嵌。

正文第8节的Python代码块实际执行，依次得到base `[[1,2]]`、middle `[[5]]`、update `[[15,20]]`、h `[[16,22]]`和merged_h `[[16,22]]`；显式增量矩阵rank=1。参数量50000对3000、比例6%的算术也通过。这是前向小算例，没有在本轮重跑历史训练。详细结果见[基础改写验证](outputs/基础改写验证.json)。原notebook与已有注释代码未改。

截图维护命令从仓库根目录执行：

```powershell
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' './lessons/02 LoRA - Low-Rank Adaptation of Large Language Models/复现/scripts/extract_paper_images.py' --pdf 'G:/zzz/Personal ChatGPT 前七讲/02 LoRA - Low-Rank Adaptation of Large Language Models/资料/LoRA_2106.09685v2.pdf'
```

解释器需要pypdfium2与Pillow。PDF保存在本地资料目录；公开笔记用本地PNG，不依赖读者重新下载论文。机器迁移时替换路径。[截图来源记录](outputs/论文截图来源.json)保存PDF、图片、脚本哈希以及页码和裁剪坐标。原BLOOM/GPT-3权重微调、论文任务质量和GPU性能均未复现。
