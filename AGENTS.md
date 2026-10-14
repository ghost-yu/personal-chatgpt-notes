# 本项目工作约定

用户要求按 paper-code-deep-reading skill 完成 Personal ChatGPT 第01–30讲，中文解释公式、shape、API及全部notebook单元。上游固定提交为 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`，单元从0编号。

每讲公开交付原notebook及必要图片、代码注释.ipynb、原代码_带说明.ipynb、Python阅读版、解释.md、代码单元解释.md及复现记录。原版按下载字节保留；说明版不改原单元；注释版只增加注释，保留历史输出，核对schema、AST和可行的执行。重要Markdown源码、命令、空单元也需说明。不得将教学验证或历史输出当成大模型复现。

用户已授权公开仓库 https://github.com/ghost-yu/personal-chatgpt-notes ，分支main；每完成一讲，更新索引，运行相关验证及 `工具/check_documents.py`，创建独立commit并push。不得仅提交本地。保留已有解释和验证证据，字幕附件、权重及完整外部源码快照保留本地。

当前第01–19讲主解释已存在；第01–13讲已按完整公开文件格式交付，下一讲第14讲。第14–18讲仍需核验全部单元并补齐公开注释文件；第19讲15个代码单元说明已完成，代码内注释待补；第20–30讲待完成。不能以主文存在代替全部代码交付。一次完成一讲并独立提交。

第03讲原版与注释版各26格在CPU小型BLOOM上下文执行并训练2步；第04讲直接使用固定官方PEFT v0.3.0，核对注入、引用、冻结、前向/梯度及adapter状态。均未运行原7B配置。具体证据见各讲复现目录。

第05讲全部27格说明、8格代码详注，两种派生notebook各55格；原版与注释版各尝试8格，7格成功，CUDA设备格18在PyTorch2.5.1+cpu中预期失败。真实CPU位模式、finfo/nextafter、舍入、溢出/下溢、存储核对通过。双问号源码检查使用实际IPython pinfo2，展示hook只捕获文本以避免外部分页器等待。未运行GPU/TPU算术或训练，旧记录保留。

第06讲全部28格说明/18格详注，两种派生notebook各57格，原语句/历史输出、IPython转换AST及阅读版AST通过。原版和注释版各13格在明确CPU教学上下文执行；原inference调用在.cuda()失败，补充仅将该表达式改为.to('cpu')后真实生成。固定PEFT0.3.0、1层H16 LLaMA、真实96词SentencePiece/LlamaTokenizer和随机非零r2/alpha4四目标adapter，本地保存/重载/冻结与4beam生成核对一致。原7B/8bit/远程adapter/多卡/交互与质量评测未运行，勿把随机输出当作原模型能力。

用户要求保持阅读入口清晰。根目录README是唯一总索引；各讲README只突出解释、注释notebook与Python阅读版。来源链接放在解释中，维护资料在资料目录，检查工具在工具目录。不要重新生成只有来源链接的原代码.md；保留原始notebook、说明版及验证证据。

第07讲19格说明/12格代码与Markdown1片段详注，两种派生notebook各39格；AST/schema/历史输出核对通过。原大CNN仅meta检查参数476938250及RGB224尺寸错误；四个通道常量改为8/16的小型CNN、8合成样本/batch4/epoch1，原train/evalute体真实执行。原CUDA AMP在CPU禁用，补充只改autocast表达式并另建CPU scaler，真正FP16 logits/FP32 loss和梯度、两次Adam更新、scale128→256、Inf跳步回退128通过；原/注释版一致。未下载MNIST或复现3090 batch128→256显存实验。

第08讲主56格/37代码格（含空55）+bitsandbytes辅助3格/1代码格，三张原图、四份派生notebook、两份阅读py已交付。全部59格说明/38代码AST、6schema/历史输出核对；原/注释各32主格实际CPU执行，53/54另在独立随机2960参数小OPT/本地词表上下文执行，50代理/51远程tokenizer/52量化加载跳过。原矩阵实际float64缩放、无舍入；独立int8存储/int32累加/FP16异常支路、零尺度/溢出、真实配置互斥和总长/新增token区别通过。未安装bitsandbytes、未下载OPT6.7B或运行CUDA/论文指标。

第09讲basics39+train27+辅助gpt2 6共72格、52代码格；三原notebook/原图、六派生notebook、三py交付。51可解析AST（含shell转换）/9schema/历史输出核对；原basics22悬空逗号SyntaxError保留，阅读版整格注释，多行str同值排版，Markdown2代码详注并实际核对。固定本地GPT2/T5资产仅改加载路径；真实2000行JSONL IterableDataset、原默认batch1000，两轮常量审计补充得到320/63merges和336/79merges，继续消费、旧对象不变、保存重载、Unicode/byte/末批/耗尽通过。basics36长度并列输出可变，保留双方原列表并核对排序契约，其余受控结果一致。原语料/代理/讲师Hub/语言模型训练未运行。（第09讲交付时下一讲为10。）

第10讲原13格/6代码/4图、两份27格派生notebook、Python阅读版与独立数据读取教学补充已交付。原字节/全部元数据与历史输出、3schema/6AST核对；原/注释各6格实际展示一致。补充真实5行JSONL/Arrow映射/显式缓存/延迟与batched map/skip顺序、固定GPT2分词与int64[5,7] padding、float32[100,5] DataLoader末批、Windows2worker range分片无重复。215bytes LF与旧220bytes CRLF区分，145/648bytes非RSS。原字幕数据代码未保存，未冒充原格；无C4/Pile/约20GBbenchmark/模型训练/多GPU。下一讲11。

第11讲原28格/15代码/原图、两份57格派生notebook、Python阅读版与窗口归一化教学补充已交付。原字节/元数据/历史输出、3schema/15AST及Markdown5/7循环AST通过；原/注释各11格整格执行一致，真实float64 Linear/MSE/SGD完整/短/不等窗口与旧图错误验证。真实Transformers4.44.2/Accelerate0.34.2、490786参数随机CPU BERT，8×16同数据/同初始化/dropout0，原23整格提供CPU上下文、26仅三个AST教学替换，四次train均2更新；原/注释状态精确一致，baseline/累积最大参数差约9.3e-10。实际NVML1卡但CPU torch可见0，原helper打印0不得当空闲GPU。未执行代理/远程BERT-large/CUDA/AMP/DDP或原显存benchmark，框架短窗口未验证。下一讲12。

工作区 G:/zzz/README.md 指向正式仓库；另外三个本地材料目录含上游、字幕、缓存和共享环境，仍被复现引用，不能当成无用副本删除。各讲README为阅读入口；未补齐注释的讲次必须明确标明状态。一次性旧整理脚本准备资料.py、prepare_lesson09.py、prepare_lesson10.py及prepare_publication.py已移除，维护以仓库内复现/scripts为准。

用户再次要求整理文件：总目录只列各讲 README，三种日常阅读文件集中说明；原版、说明版和验证证据折叠展示。本地旧学习索引与废弃 prepare_lesson11.py 已删除，旧材料目录 README 指向正式项目；缓存可删除，来源材料及共享运行环境保留原路径。此前清理提交保留第12讲在制文件，未混入整理提交；该讲随后独立完成交付。

第12讲原27格/22代码/两55格派生版/Python阅读版完整交付；原字节、3schema、AST、全部元数据及历史输出通过。Python3.11.0/torch2.0.1+cpu/Transformers4.30.0/PEFT0.4.0/TRL0.4.7真实运行，两版各16原格及24/25明确CPU变体各2步，固定GPT2 len50257、六行自建数据、一层H16 LLaMA，基础1610836参数、q/v r16共1024可训练参数。冻结/FP16→FP32、初始输出/梯度、真实packed与非packed collator、shift loss、AdamW仅adapter、两版状态/输出精确一致通过。两步warmup首lr0/A未变，B真实更新；checkpoint-2保存及匹配基础重载状态精确、logits差1.192e-7容差内，错误50320词表仍加载且输出尺寸不匹配。原代理/Alpaca全量/XgenTokenizer/7B/CUDA4bit/量化checkpointing/整轮与质量未运行。下一讲13。

第13讲原23格/12代码/Markdown21详注、两47格派生版与Python阅读版已交付；原字节/元数据/历史输出、3schema与AST通过。真实torch2.0.1+cpu/Transformers4.30.0，两版各10原格与明确CPU变体、493010参数两层H16/dropout0小BERT四组各两步。固定B/K只切checkpoint参数精确一致，累积差1.8626e-9，两版四组状态/trace精确一致；layer调用2→4或8→16，首forward no_grad/重算enable_grad实际观测。真实sequential4块2段调用[2,2,1,1]、梯度差0，初次forward非参数保存引用640→512bytes不作为峰值；dropout RNG重放/不重放、reentrant输入失败/修复、冻结embedding后encoder32参数无grad与input hook修复、eval_dataset缺失实际错误通过。NVML真实1卡/CPU CUDA不可用，不把整卡读数当CPU训练显存。原BERT-large/GPU/峰值/论文全实验未复现。下一讲14。
