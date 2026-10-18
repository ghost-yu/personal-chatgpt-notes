# 第25讲实际验证记录

## 2026-10-18：基础讲解与原图

正文沿四词Embedding/输出头→模型尺寸→参数量→GQA→SwiGLU→原词表→聊天协议→真实forward/generate/pipeline展开。原讲解逐字收进末尾折叠栏，原代码与历史报告保持本轮开始时的内容。

运行结果：[基础讲解验证.json](outputs/基础讲解验证.json)。验证器提取正文折叠栏前的五段Python，按阅读顺序执行，随后进行独立数值比较。环境为Python3.12、Torch2.5.1 CPU、Transformers4.44.2、tiktoken0.7.0、SentencePiece0.2.0、FairScale0.4.13；精确版本与文件摘要在JSON。

| 项目 | 当前结果 |
|---|---|
| 四词例子 | 输入[1,1]→向量[1,1,2]→分数[1,1,4]；logits[0,3,1,0]选择编号1“吃” |
| 完整两代结构 | HF Llama2/Llama3全部权重确为meta；参数分别6738415616/8030261248；按四部分独立公式累加相同，输入/输出权重不共享 |
| 宽度与容量 | 原补充第6/8格四个赋值AST相同；中间14336；半精度权重14.957527GiB；KV元素存储2/0.5/1GiB计算核对 |
| 真实英文切分 | Hello, world!：Llama2为[15043,29892,3186,29991]，Llama3为[9906,11,1917,0]；均还原原文 |
| 词表与模板 | 完整固定MetaTokenizer/ChatFormat；中文、数字、组合音符、emoji、空白/空串往返；全部256特殊ID；EOS/EOT文字三种编码分支 |
| 聊天拼接 | 两条简短消息共22token；独立按协议拼ID、Meta类、本地显式Jinja/HF三者相同；BOS一次、EOT两次，字符串再编码相同 |
| 真实CPU小模型 | V128256/D8/FF32/1层/2Q/1KV，共2053080参数；hidden[1,22,8]、logits[1,22,128256]、K/V[1,1,22,4] |
| 三条生成路径 | 完整前缀手写greedy、实际generate和实际pipeline的新增ID均为[62446,9649,50490] |
| 缓存与参数 | 增量/完整前缀logits最大差2.98023e-8；末隐藏状态乘输出权重等于实际logits；重跑推理后所有参数精确相同且grad=None |
| 完整官方FFN | 原补充第7格类AST等固定官方类；单rankGloo，D16/M56/2688参数，显式初始化；与NumPy独立前向及输入、三权重解析梯度核对，最大梯度误差3.05312e-16 |

两张3倍原PDF截图分别为GQA v3第2页Figure2、GLU v1第2页公式6与宽度说明，并逐张检查；原课程模块树直接内嵌。[论文截图来源.json](outputs/论文截图来源.json)记录PDF版本、来源、页码、裁剪坐标与摘要。PDF副本在本地资料/GQA_v3.pdf、GLU_v1.pdf。首发Llama3规格使用2024年4月发布介绍、固定模型卡和原配置，组件论文用于解释GQA/SwiGLU。

初次验证把FairScale原类在meta上下文中构造的大FFN也预期为meta，实际其旧`torch.Tensor`构造仍分配CPU权重，断言失败。已按真实行为记录为CPU形状/参数计数，仅查看其14336宽度与176160768参数，不使用未初始化的权重做前向。正文的HF完整两代配置另行逐项验证确为meta。小FFN全部数值显式初始化，实际前向和解析梯度比较完成。固定Transformers对旧tuple缓存的兼容警告不影响本次结果。

重做本轮：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/25 Llama 3 introduction/复现/scripts/verify_foundation_examples.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/工具/check_documents.py'
```

本轮仍使用下文说明的LESSON25_LOCAL、LESSON25_RUNTIME_LIBS、LESSON25_FAIRSCALE路径配置。无原7B/8B权重、GPU或语言质量基准；随机模型演示推理数据流。

## 此前交付记录

状态：**部分成功**。文件完整性、分词、meta结构和真实CPU小模型验证成功；没有原7B/8B权重、GPU、官方benchmark或训练复现。

## 环境和输入

Windows11、Python3.12.14、PyTorch2.5.1+cpu（CUDA不可用）、Transformers4.44.2、NumPy1.26.4、nbformat5.10.4、sentencepiece0.2.0、FairScale0.4.13。约15.69GiB可见内存，完整8B FP16权重单独就需14.96GiB，不按原大模型加载方式运行。

tiktoken0.7.0、blobfile2.1.1、pycryptodomex3.20.0、lxml5.2.2位于本讲本地资料/runtime-libs，进程通过sys.path加载，未改共享环境。filelock/requests/urllib3由现有环境提供。实际版本、生成结果、dtype子进程原错误和脚本SHA见 [delivery_verification.json](outputs/delivery_verification.json)。

输入为固定上游两notebook、两词表、PNG及tokenizer.py；无训练数据、无API推理调用、无预训练checkpoint。官方快照在仓库外，固定cf3ea595be1edfb019d21b14d8b215a39161c860，含llama/model.py、llama/tokenizer.py、llama/generation.py、MODEL_CARD.md、README.md、LICENSE。本次重新从GitHub读取固定模型卡，与本地文本逐字相等。

## 实际命令

工作目录G:/zzz/personal-chatgpt-notes，固定源码和依赖已经准备：

~~~powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/25 Llama 3 introduction/复现/scripts/build_delivery.py' --upstream 'G:/zzz/Llama 入门教程/代码/personal_chatgpt' --meta-sources 'G:/zzz/Personal ChatGPT 系列/25 Llama 3 introduction/资料/meta-llama3'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/25 Llama 3 introduction/复现/scripts/verify_delivery.py'
~~~

换电脑时在专用环境按requirements准备，再获取两源仓库：

~~~powershell
git clone https://github.com/chunhuizhang/personal_chatgpt.git C:/work/lesson25-upstream
git clone https://github.com/meta-llama/llama3.git C:/work/lesson25-meta
git -C C:/work/lesson25-meta checkout cf3ea595be1edfb019d21b14d8b215a39161c860
~~~

上游只需含固定7fbc29824171fd07dc2c30fe922e61ebc018ad1d对象，builder直接git show读取原字节，不修改上游。将上述六官方文本按原目录复制到本地资料/meta-llama3，保持字节，--meta-sources指向它。验证支持环境变量：

- LESSON25_UPSTREAM：上游Git仓库。
- LESSON25_LOCAL：本地资料目录，其下meta-llama3含固定官方源码。
- LESSON25_RUNTIME_LIBS：独立pip --target目录；若在专用环境统一安装，可指向该环境site-packages。
- LESSON25_FAIRSCALE：FairScale包父目录，也可指向专用环境site-packages。

两个脚本不自动安装或下载权重；验证会进入代码/tutorials/llama以支持原相对路径。不要直接运行Python阅读版期望自动替换原大模型加载。为准备本机独立依赖，实际用pip install --target本地runtime-libs、--only-binary=:all:与--no-deps安装上述四个新包。

## 原件和派生版

主38格/23代码、补充11格/4代码，共49格/27代码；两种派生版分别77格和23格。两原件和四派生版共6份schema验证。缺少cell id沿用原notebook兼容行为，nbformat给MissingIDFieldWarning；原件未因此重写。所有原metadata、execution_count和输出保留；说明版原单元完全相同。注释代码AST/非注释token一致；Markdown构造和完整类也验证；四Python文件与对应代码/完整官方对象AST核对。

每个原文件不仅与清单SHA比较，还与固定Git blob直接比字节。tokenizer.py与固定官方快照字节相同；完整ModelArgs/FeedForward阅读摘录与官方AST相同。

词表文件和原tokenizer.py通过Git属性禁止换行转换。官方LICENSE原文有行尾空白及混合缩进，保持原字节，并仅对该许可证文件关闭空白规范检查；新增Markdown片段的行尾空白已清理，原notebook文本未动。

## 真实执行

两版各22格原语句、4格明确CPU变体，代理格1跳过。22包括确实执行但无计算的注释/空格。时钟格0两次返回真实datetime，只比较类型且分别记录时间；不能强求时钟相同。其他可比输出/trace和最终模型逐参数SHA完全一致。

| 变体格 | 原行为 | 实际行为 |
|---|---|---|
| 主15 | 7B FP16/auto加载 | 相同完整配置的真实HF meta模型，无权重数据 |
| 主19 | 8B FP16/auto加载 | 相同完整配置的真实HF meta模型，无权重数据 |
| 主34 | 基础8B/BF16/auto pipeline | 原词表转换的Fast tokenizer+CPU FP32随机小Llama，新增预算4 |
| 主36 | Instruct8B/BF16/CUDA，新增256 | 相同小模型/CPU，新增256→4；消息、模板调用、EOS列表、采样参数保留 |

小Llama D16/M56/L1/H4/G1/V128256，4,107,568参数。真实text-generation和聊天采样执行；logits[1,3,128256]，K cache[1,1,3,4]。不评价生成质量。Fast tokenizer来自真实TikTokenConverter与原rank文件，按原顺序补256特殊词并显式给本地Jinja模板，不是原Hub配置。

完整meta计数6,738,415,616与8,030,261,248。嵌入/头+788529152，Attention−805306368，MLP+1308622848，norm0。8B两字节权重14.957527GiB；同4096上下文KV存储2GiB→0.5GiB只是shape估计。

真实原SentencePiece32000词与tiktoken128256词；八种普通样例逐ID/往返，三长文本往返；默认普通编码/允许特殊ID/禁止时报错均通过。Meta原/注释ChatFormat与本地HF模板在非空样例逐ID一致、BOS仅一份；Meta空对话分支另核对。

真实FairScale0.4.13/Gloo/world1，完整官方模型模块未经修改；补充类与官方AST相同。seed25/normal std.1显式初始化2688参数double小SwiGLU，[2,3,16]→[2,3,56]→[2,3,16]。独立NumPy前向差8.33e-17，三个输入坐标有限差分最大差6.98e-11，所有输入/权重梯度与独立Torch公式精确相同。完整4096宽类及原Markdown构造只在meta实例化，14336宽/176160768参数。两种真实并行embedding单rank查表/梯度同普通embedding。

JSONC移除整行//后解析，启动器字段匹配，未执行CLI。原CUDA默认类型片段在隔离CPU子进程实际失败，报告保留错误；CPU BF16/FP16均2bytes实测。没有把CUDA的else当成CPU回退。

## 错误、警告与未运行范围

初次tiktoken.__version__属性不存在，改为importlib.metadata.version；本地Jinja明确对content先trim。之后Windows错误子进程解码失败，改PYTHONIOENCODING=utf-8及errors=replace，完整重跑通过。均为本次工具修正，不是源模型bug。缺cell id、Transformers空白清理默认值、旧tuple缓存接口有兼容警告，未因此改原单元。

无原7B/8B权重、原Hub完整配置、NCCL/CUDA、多rank、原海盗答案、全量训练、完整语料、质量benchmark或速度结论。全课程尚未完成，下一讲26。

## 讲解改版前的运行说明

## 10. 小模型的结构与推理

下面用小模型观察词表、网络结构和生成接口。原代码及运行配置可在本讲的代码目录与复现说明中对照。

见 [复现记录](README.md) 和 [机器可读证据](outputs/delivery_verification.json)。两版各22格原语句、4格明确CPU变体，代理格跳过；时钟只比类型并分别记录真实时间。两版可比输出及最终随机小模型状态核对相同。完整7B/8B仅meta结构计数；真实生成用D16/一层/原128256词表随机Llama，新增预算改4。

真实FairScale单rankSwiGLU与独立NumPy公式、输入有限差分、所有输入/权重梯度对照及两种嵌入查表/梯度验证。没有原checkpoint、CUDA/NCCL、多rank、全量训练、原采样答案、质量或速度复现。脚本初次读取tiktoken不存在的__version__属性失败，改为发行元数据；CPU错误子进程输出编码失败后明确UTF8，均属于验证工具修正，非源模型bug。
