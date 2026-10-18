# 第28讲复现说明

## 2026-10-18 正文基础算例

状态：**部分成功**。正文六个Python块已在真实CPU PyTorch执行，完整原CoPE/SelfAttn前向与梯度、真实Adam更新通过。固定v2原PDF已成功下载到本地，补充5张原图/原公式/原表裁剪。

| 内容 | 本轮结果 |
|---|---|
| 普通注意力 | Q/K/V为[1,3,2]，三角mask；逐行纯NumPy标量参考与Torch权重/输出吻合 |
| 原RoPE函数 | 两个完整函数AST等于原11/23格；相位[3,2] complex64，两点积约1.806731，与独立double半区旋转一致（原矩阵FP32精度） |
| CoPE计数 | 论文12token句号门得到2/2/2/2/1/1/1/1/0/0/0/0；软门后缀和1.5/1.5/1.25/.75；全1门计数含当前项 |
| 插值 | p1.25得到向量[3,1]、bias5、位置导数12、表两槽梯度[.75,1.5]/[.25,.5]；真实CoPE输出5，输入logit梯度3/2.25 |
| 完整类 | 新正文CoPE与SelfAttn整个类AST等于原Markdown39；原两派生版及Python阅读版的源代码/输出/元数据核对通过 |
| Adam训练 | 原位置表8参数、固定三组Q/K/V、全0目标，5次真实Adam；optimizer状态step=5；MSE1.6707432132→1.4448043604 |
| 独立向量插值参考 | 优化实现/密集向量参考前向误差0；Q/K/V/表四种梯度最大误差2.22e-16 |
| 数值导数 | 通过torch.autograd.gradcheck核对Q/K/V/位置表四个输入，避开整数和饱和结点 |
| 梯度路径 | 零表位置支路gate梯度0、表梯度非0；更新后gate路径非0；整数结点和clamp饱和的门梯度0 |
| 因果性 | 完整输出与三个独立前缀误差0；修改未来Q/K/V不影响过去；全屏蔽行实际返回NaN |
| 原PDF | 17页，首面明确2405.18719v2；第12–13页Listing1与原两个类人工逐行核对；Table2及任务条件重新读取 |
| 原图 | Figure1、式3–4、式5–6、式7–9、Table2共5截图，全部视觉检查；页码/坐标/PDF与PNG SHA记录完整 |

本轮没有重跑原25格全部历史数值实验，保留此前原样执行报告；当前实际执行范围为正文六块、完整原函数与类及独立参考。论文Flip-Flop/Selective Copy/Counting/语言建模训练、全Transformer、GPU、KV cache与性能指标尚待验证。

### 环境与命令

Windows，Python3.12.14，Torch2.5.1+cpu；NumPy版本见[本轮报告](outputs/基础讲解验证.json)。沿用共享CPU环境，不增加依赖或下载预训练模型。PDF处理使用现有内置Python的pypdf/pypdfium2。

从课程仓库根目录执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/28 从 RoPE 到 CoPE/复现/scripts/verify_foundation_examples.py'
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' '工具/check_documents.py'
```

`LESSON28_PDF`可覆盖本地原PDF位置，默认`G:/zzz/Personal ChatGPT 系列/28 从 RoPE 到 CoPE/原论文.pdf`。原件来源审计沿用旧验证器的UPSTREAM路径，默认`G:/zzz/Llama 入门教程/代码/personal_chatgpt`，要求固定Git提交仍可读取。正文六块本身只依赖Torch；来源及图片校验另需这些本地资料。

### 图片与来源

2026-10-18从https://arxiv.org/pdf/2405.18719v2下载本地原论文，固定17页v2；此前TLS失败属于下方历史记录。截图按3倍渲染，裁剪坐标为PDF point、左上原点。第2页Figure1、第4页式3–6、第5页式7–9、第6页Table2均放入主笔记。原课程rope_1与rel_position_matrix两图原字节保留并在正文说明。

- [本轮基础讲解验证](outputs/基础讲解验证.json)：全部正文输出、环境、当前正文/脚本/PDF hash及数值/梯度证据。
- [原论文截图来源](outputs/论文截图来源.json)：固定PDF URL、页码、裁剪坐标及图像hash。
- [此前原交付报告](outputs/delivery_verification.json)：保留其原运行日期、原实验和脚本hash。

原讲解逐字放入主笔记的历史折叠，原notebook/派生注释文件/历史运行报告保持原内容。来源清单更新了PDF本地下载状态与SHA。旧build_delivery面向原交付，勿用来覆盖当前基础改写正文。

## 此前交付记录

# 第28讲验证记录

状态：**部分成功**。原代码与完整附录类、前向/反向、微训练通过；论文任务与完整语言模型未复现。

## 命令与环境

2026-10-16，Windows，Python3.12.14、torch2.5.1+cpu；NumPy和nbformat实际版本见outputs/delivery_verification.json。复用共享CPU环境，不安装新框架，不访问模型Hub/数据集或API。主机约16GB物理内存，本次只用小矩阵和24个可训练位置参数，无需GPU。

从正式仓库根目录运行：

~~~powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/28 从 RoPE 到 CoPE/复现/scripts/build_delivery.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/28 从 RoPE 到 CoPE/复现/scripts/verify_delivery.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' '工具/check_documents.py'
~~~

build_delivery只读取仓库内原notebook/图，重建两81格派生版、两Python阅读版、单元说明与来源清单。verify_delivery核对来源Git对象、schema、AST/token，再实际执行两版及诊断。来源审计默认读取G:/zzz/Llama 入门教程/代码/personal_chatgpt中的固定Git提交；其他机器需在脚本UPSTREAM设置相应clone路径，不能用空文件替代来源校验。其余训练/梯度验证不依赖外部权重或专有资产。

## 静态与执行范围

原40格中25格代码，两版均25格原样执行，无变体、无跳过。保留所有原metadata、execution_count及历史输出。原代码说明版单元不变，注释版只加注释；Markdown5/9/39代码块同样AST/token核对。原Python阅读版与25格拼接AST相同。CoPE核心文件完整保留两个类并加真实torch.nn导入，独立导入检查通过。

第4格远程图片只实际构造Image(url=...)，不等于获取或渲染远程内容；未验证该图内容。六张本地图均按固定Git blob核对字节并实际查看。原单元31中的result_1被注释，没有假报原格已经算过第一种dot；独立验证另行比较三种表达式。

Markdown39的完整类在真实torch.nn命名空间单独执行，不使用假的Module。第5格BERT/GPT摘录、9格不完整LLaMA片段只做注释/静态审计，不当完整模型运行。

## 实际结果

| 检查 | 观测 |
|---|---|
| 原件、六PNG、3schema、注释AST/token | 通过，40格全部解释 |
| RoPE polar与独立cos/sin | 相位[3,2]，结果相同 |
| 相邻/半区布局 | 置换[0,2,1,3]后矩阵相同 |
| 相对旋转 | 原三种dot吻合；独立double前向/qk梯度与共同位置平移吻合 |
| 无位置attention置换等变 | 独立double数值检查通过，限定无mask情形 |
| CoPE零初始化 | 等于普通causal attention；位置支路gate梯度0，表梯度非0 |
| 非零表与独立密集向量插值 | 前向差1.11e-16，Q/K/V/表梯度误差最大4.45e-16 |
| 数值导数 | 原SelfAttn在double非整数/未饱和区域通过gradcheck |
| 因果性 | 四个前缀与全序列输出误差0，修改未来输入不影响前缀 |
| 真实Adam | 原位置表24参数，lr=.02，5更新，MSE 1.2536735→1.2374721 |
| 原/注释对照 | 展示、所有诊断、微训练loss完全一致 |

微训练的Q/K/V与目标都是固定随机张量，只优化原位置表，不添加原代码没有的Transformer或声称任务质量。使用真实torch.optim.Adam与autograd，不以手写NumPy更新模拟训练。

## 边界与失败记录

- 原get_rope_matrix用float32 CPU zeros，即使传double角也输出float32；奇数维和错误角度长度的行为记录在报告，原函数未修。
- npos_max=1完全饱和，gate梯度0；npos_max=0真实gather越界；head_dim不匹配真实矩阵乘维度错误。
- 全零mask行返回非有限结果；传0/-inf加性mask到mask.log接口，NaN转long后gather产生非法索引RuntimeError。
- 插值整数结点ceil=floor时原代码选择0门梯度。gradcheck没有强行在不可微结点要求通过。
- 首次验证器未去除Markdown5展示缩进，AST解析报IndentationError；修正展示层共同缩进。随后错误预期加性mask只返回NaN，实际更早索引报错，已记录真实行为并修正断言。两版完整重跑成功，未将验证器错误归因于论文实现。

## 来源与未运行部分

CoPE依据固定论文v2 Appendix B Listing1与原notebook第39格逐行文本核对，没有找到并锁定独立官方训练仓库。官方候选地址请求失败也不作为“仓库不存在”的证据。web读取固定PDF文本成功；系统curl多次TLS失败，因此没有本地原论文.pdf。字幕匹配后仅留本地；六张原图随仓库提供，远程图不伪装成本地图。

未复现原论文selective copy/counting/Flip-Flop、Wikipedia/代码建模训练、perplexity、8GPU资源配置、长文本性能、完整LM、多head/多层、KV cache、CUDA或峰值显存。教学5步更新不能替代这些实验。

## 讲解改版前的运行说明

## 10. 小张量中的位置与梯度

[复现说明](README.md)记录命令与边界。两版各25个原代码格**无替换、无跳过**执行，展示/打印一致；其中Image对象只证明构造成功。另在真实torch.nn上下文执行两版Markdown39完整类，并核对独立Python文件可导入。

CoPE优化实现与独立“先插值完整向量再点积”的double参考前向最大差约1.11e-16；Q/K/V/位置表四种梯度最大差不超过4.45e-16。原SelfAttn通过真实torch.autograd.gradcheck。因果完整/前缀输出误差为0；两版得到相同结果。

用原类的24个位置参数、固定随机Q/K/V、随机MSE target做5次真实Adam更新，lr=.02，loss从1.2536735降至1.2374721；没有语言训练数据或完整Transformer。原CoPE/SelfAttn体未改，教学数据、double精度和优化器都在独立验证中声明。

补充导入验证最初未把动态加载模块注册到sys.modules，inspect误认为类是内置对象；按标准模块加载方式注册后重新核对。这也是验证工具问题，不是CoPE类无法导入。
