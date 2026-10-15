# 第20讲复现记录

状态：**部分成功**。原文件与注释一致性、真实随机小型 Meta Transformer/FairScale/Gloo 及真实 GPT-2 缓存路径通过。没有复现原预训练权重和原性能实验。

## 实际环境与资源

Windows11，Python3.12.14，PyTorch2.5.1+cpu，CUDA不可用。Transformers4.44.2、tokenizers0.19.1、SentencePiece0.2.0、NumPy1.26.4、nbformat5.10.4；FairScale0.4.13从已有本地目标目录导入，不替换当前Torch。系统RAM16849256448字节，验证时空闲内存和磁盘实测值见报告，不能当固定资源需求。

原模型权重与数据集均未下载；已有固定SentencePiece与GPT-2词表用于真实分词。随机Meta模型6864参数，GPT2模型811728参数、float32 CPU；临时checkpoint在系统临时目录创建并自动回收。不需要API key；不运行原7B、CUDA或训练benchmark。

## 运行命令

在正式仓库根目录执行。PowerShell变量 `$lessonPython` 指已有3.12 CPU环境；其他变量指已有来源材料目录。它们不是可随意删除的旧稿依赖。

```powershell
$lessonPython = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
$lessonScripts = 'lessons/20 KV cache 与增量推理/复现/scripts'
$lessonMeta = 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/llama-model.py'
$lessonFairscale = 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料/fairscale-local'
$lessonTokenizer = 'G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/gpt2-tokenizer/tokenizer.json'
& $lessonPython "$lessonScripts/download_original.py" --upstream 'G:/zzz/Llama 入门教程/代码/personal_chatgpt'
& $lessonPython "$lessonScripts/build_annotated.py" --meta-source $lessonMeta
& $lessonPython "$lessonScripts/verify_delivery.py" --meta-source $lessonMeta --fairscale $lessonFairscale --tokenizer-json $lessonTokenizer
& $lessonPython '工具/check_documents.py'
```

`download_original.py` 用git show读取固定提交的8个blob并按字节落盘，不重新序列化原notebook。`build_annotated.py` 在每个原格前加说明、代码中只加注释，输出两份51格派生版与阅读Python。`--meta-source` 必须符合固定官方SHA；`--tokenizer-json` 也核对既有固定词表哈希。`verify_delivery.py` 真实导入FairScale并建立单进程Gloo，不用伪造线性层替代。

新电脑应先准备上述固定来源与匹配依赖，再改命令的本地路径。本仓库不把个人绝对路径当成已在任何电脑安装成功的证明，也不包含模型权重。

## 验证范围

1. 原字节/元数据/历史输出、3份schema、全部25格说明、14代码AST与非注释token、阅读版AST、Markdown18摘录与完整官方Attention类注释核对。
2. 原版与注释版各13格实际执行，其中11格不改原语句；20格代理跳过；21格仅两加载字符串改本地模型；23格仅重复10→2与新增1000→16。真实词表、六图片、prompt8ID和生成结果均核对。计时stdout仅作为实测记录，不作两版相等条件或性能结论。
3. 完整固定Meta模块先实际构造捕获CPU环境的原.cuda失败；独立CPU变体仅改两次cache分配设备。因原FairScale init_method=lambda x:x不初始化矩阵，显式seed20/正态std0.1初始化全部矩阵，RMS增益设1。参考与增量模型共享同一state_dict，关闭dropout并推理。
4. 原Meta没有use_cache=False开关：参考是每次start_pos=0重算全部前缀，仍会写自己的cache；不能把它称作原实现的关闭缓存选项。比较逐层KV、全部本次logits、旧前缀不变、尾部未写，hook记录三投影输入长度。
5. 真实Attention投影后捕获，验证K带RoPE/V不带、独立repeat_interleave注意力路径；真实4步greedy与完整重算一致，未来token不改变先前结果。
6. 原版在P3/S2实时报mask尺寸RuntimeError；单独教学变体只改一条mask赋值（另加上述两CPU设备改动），[3,2,1]块通过。另捕获错误前缀、位置空洞、容量越界，验证新会话有效长度排除旧尾部。
7. 本地真实GPT2Tokenizer导出与AutoTokenizer加载、真实GPT2LMHeadModel保存/加载；两种cache设置各2次greedy新增16，四次IDs一致。模型两个head、每head8通道、两层，tuple past为[1,2,6,8]；完整前缀与增量一步误差在容差内，关闭缓存不返回past。

报告：[完整交付验证.json](outputs/完整交付验证.json)。其中脚本与阅读文件SHA256用于确认实际交付版本；源码版本、环境、错误、投影shape、生成ID、时钟stdout与未运行范围均保留。

## 解释结果的限制

原notebook保留历史 `6.636±0.023` 与 `8.283±0.049` 秒，没有替换成教学计时。当前小模型运行不验证原GPT2预训练质量、Llama2权重、原1000×10benchmark、CUDA同步/显存、多rank通信或论文指标。没有通过缩小模型就宣称原实验完成。

nbformat对旧notebook缺cell id发出兼容警告；验证通过，但没有为消除警告改写原始cell metadata。Transformers分词器默认清理空格的未来行为提示不影响当前锁定版本的比较。
