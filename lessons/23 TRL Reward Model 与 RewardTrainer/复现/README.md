# 第 23 讲复现记录

状态：**部分成功**。原文件/派生件静态核对、两份真实CPU小OPT/LoRA RewardTrainer各两步及保存重载通过；原4bit OPT350M/全HH训练未运行。

## 环境、来源与资源

Windows，Python3.12.14，torch2.5.1+cpu，Transformers4.44.2，TRL0.8.6，PEFT0.12.0，Accelerate0.34.2，Datasets2.19.2，NumPy1.26.4，PyArrow16.1.0；完整版本在输出JSON。CUDA不可用；本机约16GB RAM，本次基础模型807648参数，训练开放1040参数，不需原350M权重或API密钥。

共享解释器：`G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe`。

本地来源目录：`G:/zzz/Personal ChatGPT 系列/23 TRL Reward Model 与 RewardTrainer/资料`。其中runtime-libs为额外安装的TRL0.8.6及tyro/rich等，不覆盖共享环境的Transformers/PEFT。验证器显式把它加入sys.path；实际安装的完整RewardTrainer/utils源码字节必须等于固定官方提交 `e90e8d91d2265e484f229c45a5eb8982f94a2936`。来源SHA在公开清单中，不是只根据包版本字符串认定同源码。

OPT固定提交 `08ab08cc4b72ff5593870b5d527cf4230323703c`，只下载config.json、tokenizer_config.json、special_tokens_map.json、vocab.json、merges.txt到opt-tokenizer。分词器50265项，embedding配置50272；PAD1/BOS2/EOS2。没有下载OPT权重、全HH语料、使用付费API或运行外部代理。

真实小OPT配置：vocab_size50272，hidden_size=word_embed_proj_dim=16，ffn32，1层，2heads，max_position_embeddings64，num_labels1，全部dropout/layerdrop0，do_layer_norm_before=False，seed23。LoRA保留原r16/alpha16/q-v/SEQ_CLS/score，所以此小配置不用于证明低秩压缩收益。

## 重做命令

从正式仓库根目录执行：

```powershell
$python = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
$lesson = 'lessons/23 TRL Reward Model 与 RewardTrainer'
$sources = 'G:/zzz/Personal ChatGPT 系列/23 TRL Reward Model 与 RewardTrainer/资料'
& $python "$lesson/复现/scripts/build_delivery.py" --upstream 'G:/zzz/Llama 入门教程/代码/personal_chatgpt' --sources $sources
& $python "$lesson/复现/scripts/verify_delivery.py" --sources $sources
& $python '工具/check_documents.py'
```

第一条从固定上游Git对象保留原字节并生成注释文件；第二条执行实际验证；第三条查项目文档链接与必要章节。--sources同时提供完整外部源码、固定分词资产和runtime-libs。换电脑先按来源清单相同提交准备这些资产；不把只有精选类的中文源码阅读文件当完整可安装模块。

额外包实际安装方式（不覆盖已有包）：

```powershell
& $python -m pip install --no-deps --target "$sources/runtime-libs" trl==0.8.6 tyro==0.8.14 rich==13.9.4 docstring-parser==0.16 shtab==1.7.1 typeguard==4.4.4 markdown-it-py==3.0.0 mdurl==0.1.2
```

其余版本见requirements；CPU Torch轮子需从匹配来源安装。在线准备资产后，验证器设置HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1，证明本次运行不靠远程权重或语料请求。

## 每格执行范围与变体

原37格，29代码；每版21格原语句实际成功，6格明确变体，2格跳过。原4bit配置20格另尝试一次并捕获实际PackageNotFoundError。

| 原格 | 教学上下文 / 修改 |
|---|---|
| 0 | 不执行进程代理设置 |
| 9 | 不下载远程全数据，提供8条自建普通偏好对；2条含长候选 |
| 13 | 只把OPT模型名改成本地相同固定分词资产路径；之后显式设置教学tokenizer上限1024 |
| 15 | 两处num_proc16→1，函数体与filter阈值512不改；实际剩6条 |
| 20 | 原4bit因无bitsandbytes失败；明确把load_in_4bit改False |
| 21 | 整个pretrained/量化加载调用换成实际OPTForSequenceClassification(tiny_config)，原use_cache=False仍执行 |
| 29 | TrainingArguments改两步/batch2/lr.001/save2/log1/临时输出/report_to空列表；增加use_cpu/seed23/data_seed23/worker0/关闭进度条/safetensors；RewardTrainer调用不换成假对象 |
| 31 | 真实trainer.train保留，仅save_pretrained路径改临时目录 |

其他原语句在上述教学上下文中执行，不能称为原350M完整顺序执行。两图只创建真实Image(url)并核对HTML，未下载或核验远端图片；23格实际pinfo2取真实模型源码，展示hook只捕获文本避免CLI外部分页等待。

## 实际结果与证据

- 原SHA、全部历史/元数据、3schema、IPython转换AST与非注释token一致；两派生版各75格。原29格阅读版及完整RewardTrainer/collator/accuracy源码AST通过。
- 两版全部参数最终精确一致；每次真实2个optimizer step；冻结参数不变，optimizer仅含可训练参数，q/v的LoRA A/B与score副本真实更新。
- 1040可训练=1024 LoRA+16 score；当前PEFT冻结score.original_module。原历史两份512 score均列为可训练、总1573888，只记录历史差异，不强行伪造旧包行为。
- 固定前两对loss .7043679953→.6982839108；两训练批loss .6886/.7605不被写成单调下降；不是持出评价。
- adapter保存键为四个LoRA权重与score.weight；匹配基础状态重载logits最大差0，临时checkpoint测试后回收，公开JSON保留键名/config与结果。
- pairwise loss与softplus、两侧autograd梯度、prediction[B,2]与sigmoid差值一致；左右padding均与单条前向一致，chosen/rejected分别[B,21]/[B,19]。
- 原preprocess在上限1024时8→6，上限512时8→8；collator默认padding=True即使max_length512也保留600-token长度。
- 固定版margin[B]实际广播到[B,B]，独立column输入得到逐pair结果；不修改原类。平分时accuracy包含第一列tie且有真实警告。
- 稳定loss[-1000,0,1000]有限，naive sigmoid再log在-1000为Inf；原末尾手写sigmoid与历史模型分数分别核对。

详见[实际运行结果](outputs/runtime_verification.json)。artifact_sha256绑定两脚本、三个notebook、两个阅读文件；来源清单另外绑定官方源码/分词资产，防止用旧报告代表不同代码。

## 失败与未运行范围

首次验证卡在检查pinfo2捕获文本的断言：高亮ANSI分开了def/forward字符；仅对捕获文本去ANSI后完整重跑通过，IPython语句/模型源码检查没有替换。旧cell缺id产生schema提示，保留原元数据不消除历史提示。

原20格无bitsandbytes的PackageNotFoundError是实际边界，CPU变体明确关闭4bit。padding max_length被忽略、max_steps覆盖epoch、模型输入非标准四列无法估算FLOPs、tie accuracy等提示均真实保留，没有将其当作原实验成功。

未运行：OPT350M预训练权重、CUDA/4bit后端、16进程预处理、HH160800全量、原1000步/09:13计时、多卡、PPO、概率校准、人类持出胜率或InstructGPT完整实验。讲师精确依赖、原历史异常loss字段和两份score可训练的版本原因未确认，本讲独立字幕未找到。训练loss下降与保存契约不能证明语言或偏好质量。

## 讲解改版前的运行说明

## 11. 小模型中的成对训练

环境：Windows / Python3.12.14 / torch2.5.1+cpu / Transformers4.44.2 / TRL0.8.6 / PEFT0.12.0 / Accelerate0.34.2 / Datasets2.19.2。仅把TRL及其额外依赖安装到本讲本地runtime-libs，保留共享环境主包；未安装bitsandbytes。模型807648参数，配置与全部版本见[复现记录](README.md)。

通过项：

- 原件字节、全部元数据/历史输出/单元顺序、3schema；29代码的IPython转换AST及非注释token一致；两个阅读文件的相关AST与原代码/官方完整对象一致。
- 两版各21格原语句成功执行；另6格明确本地CPU变体成功；代理/远程数据2格跳过。原4bit格另实际尝试并记录缺包失败，具体错误见复现记录。
- pinfo2真实检查OPT forward；两图只核对HTML引用，不下载/冒称检查远端图片。
- 原preprocess真实map/filter、截断顺序反例、真实padding/单序列pooling与两侧不同长度。
- 真实pairwise前向/梯度、softplus/logsigmoid一致、预测概率与差值sigmoid一致、共同平移/单score不校准、稳定极值/平分accuracy与margin广播反例。
- 真实两次两步LoRA训练、冻结/optimizer/更新、两版状态精确一致、adapter与score保存/重载差0。

初次验证在pinfo2输出断言失败，因为IPython高亮ANSI把`def forward`字符分开；给**捕获文本**去掉ANSI后检查通过，未替换源码检查或模型实现。nbformat对旧单元缺id给提示，但验证通过；不为消除提示篡改原元数据。原4bit缺包是预期边界，不是已经解决了量化支持。
