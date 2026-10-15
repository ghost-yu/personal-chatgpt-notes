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
