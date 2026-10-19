# 第30讲复现记录

## 2026-10-18 正文基础算例

直接提取主讲解折叠前的六块 Python，使用 Python 3.12.14、Torch 2.5.1+cpu、Transformers 4.44.2、SciPy 1.13.1，CPU 单线程执行。

- 分差1：概率0.731059、loss0.313262，chosen/rejected 梯度约-0.268941/+0.268941。自然对数和log10换底、极端分差的稳定loss、四种交叉熵形式、一/二阶导数、gradcheck、平移/缩放和空比较检查通过。
- 原主第23格与补充第2格完整代码 AST 相同。原绘图重新生成的PNG与已发布图片字节相同；曲线数组用 logaddexp 独立核对。
- D>C>A=B 得到5个严格偏好对；第二问题3对。按问题均值与全pair均值的权重差异核对。
- 真实随机一层 Llama 奖励模型2880参数，输入 `[7,6]`，隐藏 `[7,6,16]`，head `[1,16]`。实际5次 AdamW 更新，损失0.691456→0.420224；12个状态张量发生变化，含score和骨干。
- 手算最后有效位置的线性head、追加右padding、完整摘录类输出、保存/重载状态与分数核对通过。7条候选分数复用与16条重复输入+BCE独立实现的loss差2.98e-8，全部参数梯度最大差4.18e-7。
- 补充原5个代码格实际执行，四队22场、五品牌1000次比较用真实SciPy估计。独立Torch稳定似然分别13.42845008、657.80181606；自由梯度和正定Hessian核对通过。
- 两份原notebook/四份派生notebook共6份schema，原35格字节/历史输出/AST token与完整HF类/包RECORD通过。本轮同时精简单元说明和阅读版中的重复、自证注释，原计算语句保持不变。

详细结果见 [基础讲解验证.json](outputs/基础讲解验证.json)。原绘图字符串带有Python 3.12的反斜杠转义提示，原语句保留，图与数值检查通过。

~~~powershell
$py = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
& $py ./复现/scripts/verify_foundation_examples.py
~~~

论文固定为 arXiv:2203.02155v1，共68页，本地文件为 `G:/zzz/Personal ChatGPT 系列/30 InstructGPT reward modeling/资料/InstructGPT_v1.pdf`。Figure2、公式1、分组比较段落、C.2及平局/批量续段共五张截图，均逐张视觉检查；PDF与PNG的SHA256和裁剪坐标见 [论文截图来源.json](outputs/论文截图来源.json)。原论文相关段落已在正文解释。

本轮运行教学算例与自建ID小模型；原GPT-3/6B RM权重、人工训练数据、系统质量评测和PPO未运行。下面保留此前全格与八步训练记录，维持原日期和结果。

## 此前交付记录

# 第 30 讲复现记录

状态：**部分成功**。两个原 notebook 的全部 17 个代码单元，均在原版与注释版中原样执行，输出一致；概率性质、SciPy 估计及真实随机小 Llama 的训练路径验证通过。没有复现原 InstructGPT 6B 的训练或人类偏好指标。

## 环境与命令

实际环境为 Windows 11、Python 3.12.14、Torch 2.5.1+cpu、NumPy 1.26.4、SciPy 1.13.1、Matplotlib 3.9.4、Transformers 4.44.2、nbformat 5.10.4、IPython 8.37.0、psutil 7.2.2；CPU 单线程、seed 30，无 CUDA。物理内存约 16.85 GB，执行时磁盘可用约 23.22 GB；二者不是峰值内存测量。

在本讲目录运行：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' '复现/scripts/verify_delivery.py'
```

解释器执行验证脚本；脚本定位本讲目录，逐格执行两版代码，再验证解析损失、SciPy 拟合和真实小模型。输出写入 [delivery_verification.json](outputs/delivery_verification.json)，绘图保存在本讲 `图片/`。不下载大模型或调用付费 API。

SciPy 仅安装到本讲本地资料目录的 `runtime-libs`，未改变共享 NumPy/Torch。该目录默认是 `G:/zzz/Personal ChatGPT 系列/30 InstructGPT reward modeling/资料`，可用环境变量 `LESSON30_DATA` 改写。换机器时可以在独立 Python 3.12 虚拟环境按 [requirements.txt](requirements.txt) 安装；若 SciPy 已在该环境，无需额外目标目录。静态来源核验另依赖固定上游 Git checkout，脚本当前使用 `G:/zzz/Llama 入门教程/代码/personal_chatgpt`；迁移时需同步修改脚本中的该路径，再重新生成验证记录，不可沿用旧脚本哈希。

`build_delivery.py` 是维护用构建器，按固定 Git blob 生成原版、只加注释版、说明版和 Python 阅读版，并导出安装版 HF 完整类。日常阅读不需要运行；修改构建器或注释代码后应重跑构建及验证。

## 实际核验

| 范围 | 已验证的结果 |
|---|---|
| 原材料保留 | 主 27 格/12 代码、补充 8 格/5 代码；两原件与固定 Git blob 字节相同，保留元数据和历史输出 |
| 派生文件 | 四份派生 notebook 分别 55/55/17/17 格；连同原版共 6 个 schema，通过 AST、非注释 token 和阅读版 AST 核对 |
| 原代码执行 | 原版与注释版各 17 格，无变体或跳过；图像 URL 对象、stdout 和末表达式一致。两张函数曲线数组及新生成 PNG 一致 |
| 损失与导数 | 四种等价损失、解析一/二阶导、double gradcheck、极值稳定性、平移与缩放性质、ties 和两种 mean 层级 |
| SciPy | 真实 BFGS；四队 22 观测的 NLL 13.42845008，五品牌 1000 观测的 NLL 657.80181606；独立 Torch 似然及近最优梯度核对 |
| 完整 HF 类 | LlamaForSequenceClassification 与安装 RECORD/源码 AST 相符，导出完整类可导入，前向相同；许可证保留 |
| 真实教学训练 | 2880 参数随机单层 Llama；7 条自建 ID 序列、5+3 个严格比较对、8 次 AdamW 更新，loss 0.691456 → 0.348109；骨干和 score 都更新，保存重载精确一致 |
| 前向复用 | 7 条唯一序列与 16 条重复序列评分 loss 相同，全权重梯度最大差约 2.84e-8；末非 pad 取分和追加右 padding 不变性通过 |

训练序列不含自然语言语义。loss 下降证明程序确实更新，不能证明语言质量或真实偏好泛化。四队输出的“2 iterations”是外层调用次数，最后一次 BFGS 内部 `nit=0`；品牌内部 `nit=10`，不能混用这些计数。

## 警告、失败与未运行范围

- Python 3.12 对原绘图字符串中的 `\l`、`\s` 报 SyntaxWarning；原字符串保留。Agg 后端的 `show()` 报非交互警告，仍从真实 Figure 保存曲线，没有伪造输出。
- 原朴素 `exp` 似然在极端输入出现非有限数，稳定目标有限；这是实际数值反例。原自然对数和 log10 混用、分差 3 就近似“无梯度”的说法在讲解中纠正。
- 远端三张图片只构造展示对象，没有核验远程图片内容。论文网页 PDF 文本已读，本地 PDF 下载与截图请求失败；没有宣称本地论文文件或译本存在。补充 notebook 中的私有会话未读取。
- 未运行原 GPT-3/InstructGPT 6B 权重、原训练数据、原 RM 全训练、PPO、人类标注、泛化评测或 GPU 性能测试。新图仅核对本次两版一致，不宣称与讲师历史 Matplotlib 图像逐像素一致。

报告保存脚本及教学模块 SHA256；修改代码后旧报告不再代表当前代码。原历史输出与本次结果分开保留。
