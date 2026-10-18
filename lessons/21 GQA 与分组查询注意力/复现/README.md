# 第21讲基础讲解运行记录（2026-10-18）

本轮执行当前解释.md第一个折叠区之前的五个Python代码块，并用固定Meta完整模块比较分组Attention的前向、梯度和缓存。Python3.12.14、Torch2.5.1+cpu、NumPy1.26.4，已有FairScale0.4.13本地目标目录，Gloo单rank。原notebook、注释版及旧运行报告保持原样。

## 重做当前正文例子

从仓库根目录执行：

```powershell
$lessonPython = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
& $lessonPython 'lessons/21 GQA 与分组查询注意力/复现/scripts/verify_foundation_examples.py'
```

脚本复用第20讲verify_foundation_examples.py中的CPU模块加载器。默认读取第15讲资料下的固定llama-model.py和fairscale-local；换位置时传入`--meta-source`、`--fairscale`。独立执行正文代码时，可用环境变量`LESSON20_META_SOURCE`、`LESSON20_FAIRSCALE`指定这两个路径，然后按正文顺序执行五块。

加载器先核对固定源码SHA，将原两处cache.cuda分配改到CPU；Attention.forward和repeat_kv函数体保持原样。完整类仍使用真实FairScale投影层。原文件不改写，当前脚本、笔记、加载器和官方源码哈希一起写入报告。

## 本轮结果

- 两个Q头读同一组K/V，得到两种概率与输出：约[.6698,.3302]/[.3302,.6698]，输出[1.3395,1.3210]/[.6605,2.6790]。它们属于同一个位置的两个查询头。
- 正文repeat_kv与官方函数AST一致，真实官方调用同值；[1,2,2,2]输入重复为[1,2,4,2]，按连续组排列。expand共享存储，当前G2/R2合并后的reshape另分配；R1直接返回原对象。四副本加权的共享梯度为[3,7]。权重平均例得到两组首项8/40。
- 正文实际FP32 Attention：D8、H4、G4/2/1、batch1、容量8。三个模块参数为256/192/160，缓存512/256/128bytes，输出均[1,3,8]；所有权重显式seed21正态初始化。
- 独立逐头参考用矩阵乘法直接读取四投影权重，以实数分量旋转Q/K、逐头选择连续组并计算softmax。真实FairScale原类与它的最大前向差4.4925245e-9，输入及全部四投影梯度最大差2.2068796e-8。这里模型参数为double，但原RoPE和softmax内部仍按固定源码使用float32；没有把整Attention称为纯double gradcheck。
- 组映射误写为循环交错时，当前输出差.31522184。真实GQA先3token再1/1/1，缓存保持G2，各次有效K形状[1,3–6,2,2]，旧K/V前缀不变；对比全前缀计算最大差2.0816682e-17。
- 将MHA组内K/V权重绑定为相同值时，与GQA前向精确同值；输入/投影梯度比较包含组内KV梯度求和，最大差3.3717441e-9。将该绑定权重平均可恢复原GQA权重；一般随机MHA直接平均会改变计算，当前输出差.06576518。

数值及逐块stdout见[基础讲解验证.json](outputs/基础讲解验证.json)。未加载原预训练权重，也未重跑旧全套交付验证、T5 uptraining、论文评估或GPU计时。

## 原论文截图

使用本地已有[GQA v3 PDF](https://arxiv.org/pdf/2305.13245v3)，第2页Figure2、第1页Figure1和第3页Table1三张原图按3倍像素渲染裁剪，已逐张视觉检查。原PDF保存在外部课程资料中；坐标、页码与PDF/PNG哈希见[截图来源](outputs/论文截图来源.json)。正文翻译并解释了分组、按组平均、继续预训练和表中时间/任务分数的含义。

---

# 第21讲复现记录

状态：**部分成功**。原文件与详细注释完整交付，原两代码格实际执行，固定官方模块的GQA前向/梯度/缓存与分组边界验证通过。没有运行原Llama2权重或GQA论文T5训练/评估。

## 环境与来源

Windows11，Python3.12.14、Torch2.5.1+cpu、FairScale0.4.13、nbformat5.10.4、IPython8.37.0。系统RAM16849256448字节；磁盘空闲实测见报告。CPU不可用CUDA，原cache.cuda构造失败已真实捕获。模型只有D16、H4、G4/2/1的小型Attention，最多1024个参数；合成隐藏状态，无tokenizer、数据集、checkpoint或API key需求。

源notebook与PNG固定上游提交见[来源清单](../来源清单.json)；完整Meta模块SHA256由builder和verifier分别锁定。完整官方快照及FairScale目标目录保留在本地来源材料，公开提供两个核心对象的中文注释阅读版。GQA v3论文PDF保留本地 `G:/zzz/Personal ChatGPT 系列/21 GQA 与分组查询注意力/原论文.pdf`；论文与模型质量实验分开处理。

## 命令

从正式仓库根目录运行：

```powershell
$lessonPython = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
$lessonScripts = 'lessons/21 GQA 与分组查询注意力/复现/scripts'
$lessonMeta = 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/llama-model.py'
$lessonFairscale = 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local'
& $lessonPython "$lessonScripts/build_delivery.py" --upstream 'G:/zzz/Llama 入门教程/代码/personal_chatgpt' --meta-source $lessonMeta
& $lessonPython "$lessonScripts/verify_delivery.py" --meta-source $lessonMeta --fairscale $lessonFairscale
& $lessonPython '工具/check_documents.py'
```

builder用git show读取原blob，不重新序列化原notebook；`--upstream`指定已有上游clone，`--meta-source`指定固定官方源码。verifier的`--fairscale`将已有固定版本目标目录加入sys.path；建立真实单进程Gloo组，最后清理。新电脑需先准备匹配依赖和来源路径，不能把本地安装状态推广到其他机器。

## 实际执行与比较

1. 原notebook5格全部说明、两11格派生版、原输出/元数据、3schema、2代码与三阅读材料AST核对。原两格与注释两格实际运行、图片字节/hash/宽度一致。
2. Markdown4六条原赋值不改语句，明确提供CPU shape上下文执行。变量来自合成Tensor和SimpleNamespace；没有伪称运行70B模型或多rank。
3. repeat_kv的G/R四配置与repeat_interleave逐元素一致；随机非均匀cotangent梯度按重复轴求和精确一致；double gradcheck、非连续切片输入通过，正负/零重复数和rank错误另记。
4. 完整Meta模块仅两次cache分配.cuda改.to(cpu)。参数double，显式正态std0.1初始化；原init_method=lambda x:x不初始化权重。官方RoPE和softmax内部仍用float32，整Attention没有被宣称为纯double gradcheck。
5. 三种H4/G4、G2、G1的真实FairScale Attention，对照独立逐head线性投影/分组attention/输出投影，前向与x、四投影权重梯度一致（容差atol1e-6、rtol1e-5）；另对照不显式重复KV的分组einsum前向。
6. 同组绑定权重的真实MHA与GQA前向相等，WK/WV梯度按组求和吻合；相同heads池化恢复，一般随机MHA池化使输出改变，未执行uptraining或优化器更新。
7. prefill3后逐token1/1/1，与完整前缀重算Attention输出、有效KV一致；cache仍为G2。非法H6/G4实际head维报错；None默认MHA和D18/H4截断宽度的形状行为分别记录。

报告：[完整交付验证.json](outputs/完整交付验证.json)，包含环境、异常、逐case形状/梯度差/存储、初始源码及执行脚本/阅读文件哈希。

初次自编校验脚本在求和后的KV权重梯度形状比较中漏合并G/d轴，出现8与4维不匹配；修正为sum后reshape回`[G*d,D]`再比较，完整重跑通过。这个错误属于验证脚本，不是原Meta模型缺陷。旧notebook缺cell id的nbformat兼容警告保留，未为消除警告改写原始metadata。

未运行：原Llama2/GQA论文权重、T5 uptraining与评估集、GPU计时、多rank通信、论文质量指标。原2格执行成功不能代替这些实验，随机小模块输出也不代表语言能力。

## 讲解改版前的运行说明

## 12. 小模型中的分组注意力

下面在CPU小模型中比较原代码、分组计算和官方Attention的输出。数值见[运行报告](outputs/完整交付验证.json)，配置与命令见[复现说明](README.md)。

| 检查 | 实际证据 |
|---|---|
| 原文件/派生版 | 原notebook与PNG哈希，3schema、原元数据/历史输出、2代码AST/token、六Markdown语句与官方核心阅读AST/token |
| notebook执行 | 两版各2原格真实执行，图片100287字节、hash及width完全一致 |
| Markdown上下文 | 六条原赋值真实执行，Q=[1,2,64,128]，K/V=[1,2,8,128]，R=8 |
| repeat_kv | 四种G/R、正确连续重复、R1对象相同、存储共享与分配、非连续输入、上游梯度求和、double gradcheck |
| 官方Attention | 完整固定模块只将两次cache.cuda改CPU；真实FairScale/Gloo、world_size=1，显式随机初始化 |
| MHA/GQA/MQA | H4/G4、H4/G2、H4/G1分别与逐head参考前向及x/Wq/Wk/Wv/Wo梯度吻合；分组einsum前向一致 |
| 错误重复顺序 | G2的循环配对输出差约0.575，真实反例不是示意猜测 |
| 绑定与池化 | 特殊绑定MHA前向同值、KV梯度求和吻合；相同heads平均恢复，一般池化改变输出 |
| 增量缓存 | 先3 token再1/1/1，输出与完整前缀重算最大差约6.94e-17；持久cache始终G2，非repeat后的H4 |
| 配置边界 | H6/G4真实matmul失败；默认None退化MHA；D18/H4被//截断但仍能返回18通道，不能把整除当成已有assert |

最后一个D18案例只验证shape及截断事实，mask=None的多token调用不作为因果语义验证。R=0也会返回空head轴而非主动报错，不能作为有效GQA配置；负R与rank3输入的实际异常保存在报告。
