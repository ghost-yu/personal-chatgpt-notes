# 本项目工作约定

用户要求按 paper-code-deep-reading skill 完成 Personal ChatGPT 第01–30讲，中文解释公式、shape、API及全部notebook单元。上游固定提交为 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`，单元从0编号。

每讲公开交付原notebook及必要图片、代码注释.ipynb、原代码_带说明.ipynb、Python阅读版、解释.md、代码单元解释.md及复现记录。原版按下载字节保留；说明版不改原单元；注释版只增加注释，保留历史输出，核对schema、AST和可行的执行。重要Markdown源码、命令、空单元也需说明。不得将教学验证或历史输出当成大模型复现。

用户已授权公开仓库 https://github.com/ghost-yu/personal-chatgpt-notes ，分支main；每完成一讲，更新索引，运行相关验证及 `工具/check_documents.py`，创建独立commit并push。不得仅提交本地。保留已有解释和验证证据，字幕附件、权重及完整外部源码快照保留本地。

当前第01–21讲主解释已存在；第01–21讲已按完整公开文件格式交付，下一讲第22讲。第22–30讲待完成。不能以主文存在代替全部代码交付。一次完成一讲并独立提交。

第03讲原版与注释版各26格在CPU小型BLOOM上下文执行并训练2步；第04讲直接使用固定官方PEFT v0.3.0，核对注入、引用、冻结、前向/梯度及adapter状态。均未运行原7B配置。具体证据见各讲复现目录。

第05讲全部27格说明、8格代码详注，两种派生notebook各55格；原版与注释版各尝试8格，7格成功，CUDA设备格18在PyTorch2.5.1+cpu中预期失败。真实CPU位模式、finfo/nextafter、舍入、溢出/下溢、存储核对通过。双问号源码检查使用实际IPython pinfo2，展示hook只捕获文本以避免外部分页器等待。未运行GPU/TPU算术或训练，旧记录保留。

第06讲全部28格说明/18格详注，两种派生notebook各57格，原语句/历史输出、IPython转换AST及阅读版AST通过。原版和注释版各13格在明确CPU教学上下文执行；原inference调用在.cuda()失败，补充仅将该表达式改为.to('cpu')后真实生成。固定PEFT0.3.0、1层H16 LLaMA、真实96词SentencePiece/LlamaTokenizer和随机非零r2/alpha4四目标adapter，本地保存/重载/冻结与4beam生成核对一致。原7B/8bit/远程adapter/多卡/交互与质量评测未运行，勿把随机输出当作原模型能力。

用户要求保持阅读入口清晰。根目录README是唯一总索引；各讲README只突出解释、注释notebook与Python阅读版。来源链接放在解释中，维护资料在资料目录，检查工具在工具目录。不要重新生成只有来源链接的原代码.md；保留原始notebook、说明版及验证证据。

2026-10-14 再次整理：删除本地第17–19讲三份重复原代码.md及第18讲资料/RoPE总览原代码.md；先核对四个对应原notebook与正式仓库原版的全部JSON内容一致，再将旧文档引用改到正式原notebook。保留有检查脚本或历史文档依赖的其他来源文件，不移动仍被复现引用的本地环境目录。

最新整理删除了本地 Personal ChatGPT 系列第09–16讲的8份旧原代码.md文本副本：未被现有文档或脚本引用，正式项目已有对应原notebook及历史输出。不要重新生成这些副本。日常阅读只进入正式仓库；本地材料目录及共享环境保留原路径。

第07讲19格说明/12格代码与Markdown1片段详注，两种派生notebook各39格；AST/schema/历史输出核对通过。原大CNN仅meta检查参数476938250及RGB224尺寸错误；四个通道常量改为8/16的小型CNN、8合成样本/batch4/epoch1，原train/evalute体真实执行。原CUDA AMP在CPU禁用，补充只改autocast表达式并另建CPU scaler，真正FP16 logits/FP32 loss和梯度、两次Adam更新、scale128→256、Inf跳步回退128通过；原/注释版一致。未下载MNIST或复现3090 batch128→256显存实验。

第08讲主56格/37代码格（含空55）+bitsandbytes辅助3格/1代码格，三张原图、四份派生notebook、两份阅读py已交付。全部59格说明/38代码AST、6schema/历史输出核对；原/注释各32主格实际CPU执行，53/54另在独立随机2960参数小OPT/本地词表上下文执行，50代理/51远程tokenizer/52量化加载跳过。原矩阵实际float64缩放、无舍入；独立int8存储/int32累加/FP16异常支路、零尺度/溢出、真实配置互斥和总长/新增token区别通过。未安装bitsandbytes、未下载OPT6.7B或运行CUDA/论文指标。

第09讲basics39+train27+辅助gpt2 6共72格、52代码格；三原notebook/原图、六派生notebook、三py交付。51可解析AST（含shell转换）/9schema/历史输出核对；原basics22悬空逗号SyntaxError保留，阅读版整格注释，多行str同值排版，Markdown2代码详注并实际核对。固定本地GPT2/T5资产仅改加载路径；真实2000行JSONL IterableDataset、原默认batch1000，两轮常量审计补充得到320/63merges和336/79merges，继续消费、旧对象不变、保存重载、Unicode/byte/末批/耗尽通过。basics36长度并列输出可变，保留双方原列表并核对排序契约，其余受控结果一致。原语料/代理/讲师Hub/语言模型训练未运行。（第09讲交付时下一讲为10。）

第10讲原13格/6代码/4图、两份27格派生notebook、Python阅读版与独立数据读取教学补充已交付。原字节/全部元数据与历史输出、3schema/6AST核对；原/注释各6格实际展示一致。补充真实5行JSONL/Arrow映射/显式缓存/延迟与batched map/skip顺序、固定GPT2分词与int64[5,7] padding、float32[100,5] DataLoader末批、Windows2worker range分片无重复。215bytes LF与旧220bytes CRLF区分，145/648bytes非RSS。原字幕数据代码未保存，未冒充原格；无C4/Pile/约20GBbenchmark/模型训练/多GPU。下一讲11。

第11讲原28格/15代码/原图、两份57格派生notebook、Python阅读版与窗口归一化教学补充已交付。原字节/元数据/历史输出、3schema/15AST及Markdown5/7循环AST通过；原/注释各11格整格执行一致，真实float64 Linear/MSE/SGD完整/短/不等窗口与旧图错误验证。真实Transformers4.44.2/Accelerate0.34.2、490786参数随机CPU BERT，8×16同数据/同初始化/dropout0，原23整格提供CPU上下文、26仅三个AST教学替换，四次train均2更新；原/注释状态精确一致，baseline/累积最大参数差约9.3e-10。实际NVML1卡但CPU torch可见0，原helper打印0不得当空闲GPU。未执行代理/远程BERT-large/CUDA/AMP/DDP或原显存benchmark，框架短窗口未验证。下一讲12。

工作区 G:/zzz/README.md 指向正式仓库；另外三个本地材料目录含上游、字幕、缓存和共享环境，仍被复现引用，不能当成无用副本删除。各讲README为阅读入口；未补齐注释的讲次必须明确标明状态。一次性旧整理脚本准备资料.py、prepare_lesson09.py、prepare_lesson10.py及prepare_publication.py已移除，维护以仓库内复现/scripts为准。

用户再次要求整理文件：总目录按讲次直接链接中文解释、注释 notebook 和 Python 阅读版，课程名链接各讲 README；原版、说明版和验证证据折叠展示。本地旧学习索引与废弃 prepare_lesson11.py 已删除，旧材料目录 README 指向正式项目；缓存可删除，来源材料及共享运行环境保留原路径。此前清理提交保留第12讲在制文件，未混入整理提交；该讲随后独立完成交付。

第12讲原27格/22代码/两55格派生版/Python阅读版完整交付；原字节、3schema、AST、全部元数据及历史输出通过。Python3.11.0/torch2.0.1+cpu/Transformers4.30.0/PEFT0.4.0/TRL0.4.7真实运行，两版各16原格及24/25明确CPU变体各2步，固定GPT2 len50257、六行自建数据、一层H16 LLaMA，基础1610836参数、q/v r16共1024可训练参数。冻结/FP16→FP32、初始输出/梯度、真实packed与非packed collator、shift loss、AdamW仅adapter、两版状态/输出精确一致通过。两步warmup首lr0/A未变，B真实更新；checkpoint-2保存及匹配基础重载状态精确、logits差1.192e-7容差内，错误50320词表仍加载且输出尺寸不匹配。原代理/Alpaca全量/XgenTokenizer/7B/CUDA4bit/量化checkpointing/整轮与质量未运行。下一讲13。

第13讲原23格/12代码/Markdown21详注、两47格派生版与Python阅读版已交付；原字节/元数据/历史输出、3schema与AST通过。真实torch2.0.1+cpu/Transformers4.30.0，两版各10原格与明确CPU变体、493010参数两层H16/dropout0小BERT四组各两步。固定B/K只切checkpoint参数精确一致，累积差1.8626e-9，两版四组状态/trace精确一致；layer调用2→4或8→16，首forward no_grad/重算enable_grad实际观测。真实sequential4块2段调用[2,2,1,1]、梯度差0，初次forward非参数保存引用640→512bytes不作为峰值；dropout RNG重放/不重放、reentrant输入失败/修复、冻结embedding后encoder32参数无grad与input hook修复、eval_dataset缺失实际错误通过。NVML真实1卡/CPU CUDA不可用，不把整卡读数当CPU训练显存。原BERT-large/GPU/峰值/论文全实验未复现。下一讲14。

最新文件整理：根 README 的课程表直接提供三种阅读链接；第08/09讲的多文件阅读顺序仍由各讲README说明。G:/zzz/README.md提供第01讲三个直接入口。清理提交只包含入口与维护说明，第14讲在制文件单独保留，不能混入或标记已完成。

第14讲原23格/16代码/Markdown15三命令、两47格派生版及Python阅读版完整交付；原字节/历史输出/3schema/AST通过。Python3.11/Torch2.0.1+cpu/Transformers4.31/TRL0.5/PEFT0.4，两版各13原格+6/18明确本地CPU变体，32000词表/1026608参数真实随机小LLaMA pipeline生成精确一致。完整固定官方参考CLI真实跑两步，默认seq512/r64/alpha16、q/v共4096可训练，基础冻结/B更新/optimizer仅adapter、五行截断/左pad/全ignore NaN负例/shift一致、四adapter状态与重载logits差0、4/8bit互斥及meta7B参数6738415616通过。原7B权重/4bit CUDA/full9846/924步/五小时/质量未运行，早期SP报告保留。下一讲15。

第15讲主12格/3代码及接口补充16格/9代码，两原件/两PNG、四派生版(25/33格)、两Python阅读版完整交付；原字节/历史/schema6/AST及Markdown片段/4命令语义通过。两版主3格Image和补充9格真实SP执行一致。Torch2.0.1+cpu/FairScale0.4.13单进程Gloo，完整固定官方tokenizer/model/generation源码，模型仅两cache.cuda改CPU、generate三个device改CPU；随机1027376参数/1层H16/V32000/heads2小Transformer真实text/chat/采样/echo/6dialog/batch4拒绝和8接受/角色错误/unsafe先生成/受控真实权重EOS通过。实际旧源码新增logprobs=-0，真实greedy log概率约-10.06，因填采样token前CE目标仍pad被忽略，保留并说明反例。未build/NCCL/原checkpoint/GPU/MP2+/质量性能；早期fake generate报告保留。下一讲16。

第16讲原42格/28代码含空41、Markdown10三命令/31缩略概览，两85格派生版/五原图/Python阅读版完整交付；原字节/历史/3schema/AST通过。原根/嵌套notebook同blob，不重复发布；图放tutorials/imgs使原../imgs可读。Python3.11/Torch2.0.1+cpu/Transformers4.30/TRL0.4.7/Accelerate0.20.3，固定GPT2tokenizer.json导出真实slow词表，随机809489参数H16/1层/value17。两版全部28格执行（仅18的三加载字符串本地变体），真实原query7/完整response27/输入34，mask[1,33]有效27含第二份query，四Adam更新与全active状态/生成/stats精确一致。ref状态不变但18requires_grad属性仍True/gradNone；optimizer仅active。独立纯response修正版同20续写、输入27/mask[1,26]有效20，四更新。原生成参数保留兼容；Path→str旧wrapper类型问题记录。没有原GPT2权重/learned RM/Llama2 SFT/GPU或偏好质量，早期数学报告保留。下一讲17。

第17讲原28格/20代码及Markdown3类、两57格派生版/原版/Python阅读版和完整官方FeedForward中文源码阅读已交付；原字节/所有元数据历史/3schema/comment-only token/AST与两阅读版类AST通过。Python3.12.14/Torch2.5.1+cpu/FairScale0.4.13，两版20原代码格同外置seed17/Agg实际执行精确一致。完整未修改固定Meta模块，官方RMSNorm与课程类输出相同，FP32 autograd Jacobian误差8.35e-8、weight8.34e-8，六dtype条件/零eps NaN/近零差/增益/train-eval通过；内部.float不冒称double gradcheck。真实Gloo/FairScale world1，显式初始化144参数SwiGLU，[2,3,4]→[2,3,12]→[2,3,4]，独立NumPy前向差2.43e-17、所有输入/权重FD差最大1.89e-11；4096宽度仅meta得到11008/135266304。旧NumPyfixture记录保留，没有原Llama权重/全模型/optimizer训练/GPU/多rank/论文质量速度。初次Python3.7 ast.get_source_segment失败，改3.12恢复。下一讲18。

第18讲主/总览各29格共58格35代码含两空28，四59格派生版/两Python阅读版/原PNG与配图rotate_half/apply_rotary_pos_emb转录详注交付。原字节/历史/6schema/comment-only token/AST/Markdown21函数与两py通过，两版各35原格真实CPU执行trace精确一致。完整未修改Meta模块的precompute与两课程函数精确相同；默认ModelArgs频率表[4096,64]2MiB，不是模型运行。主26原global d_model缺陷d4差0.31582/改global12差1.27394/删global NameError，一处forward Name→self Attribute独立修正d2/4/6/8通过，奇数反例保留。outer/broadcast跨路径差2.98e-8、complex恢复5.96e-8，不能跨路径强求bit等价。截图是HF半区配对代码，精确版本未找到；转录小函数实际布局排列后与相邻复数旋转相同，混用差1.59950。start_pos切片/存储view共享flip分配/相对点积及qk梯度/polar complex gradcheck通过。旧报告保留，没有原Llama权重/整模型Attention/Meta apply/GPU/多rank/论文质量速度。当前E:/30个txt仍未找到本讲独立字幕。下一讲19。

第19讲全部27格/15代码含注释17与空26、两55格派生版/原tokenizer499723bytes/Python阅读版及三官方函数中文源码阅读交付；原字节/历史元数据/3schema/comment-only token/AST、reading AST、官方函数执行AST一致。旧15代码长说明保留并补齐Markdown。两版各15原格真实执行trace精确一致、token8ID同历史；随后两版缺helper调用均真实NameError，仅在原格全部执行后显式补完整固定Meta helper，原/注释/完整未修改官方模块输出及梯度精确相同，独立sin/cos和逆旋转gradient差0。dtype四种finite backward/FP64内部降FP32/stride2错误和contiguous修复/合法非连续/静默错axis2.90499/短K1广播成4位置/rank3返回rank4反例通过；相对score7.47e-7/共同移位4.77e-7/重置位置7.08538。curve cumsum误差5.33e-15、113次上升，全局d64不传s时63而显式d64为15.5。初次dtype叶tensor复用错误，改detach-clone各case恢复。旧verify/build_annotated/43格覆盖/教学报告保留；新完整生成用build_delivery。没有原Llama权重/整模型Attention/KV/repeat_kv/GPU/通信/论文质量速度。下一讲20。

第20讲全部25格/14代码含空24、Markdown18不完整Attention逐句与完整固定官方Attention类详注、两51格派生版/原版/六图/SP模型/Python阅读版已交付。原字节/历史/3schema/comment-only token/AST通过，两版各13格实际执行（11原格、21两加载字符串和23两整数教学变体，20代理跳过）；真实8ID和生成契约一致，计时仅记录。Torch2.5.1+cpu/FairScale0.4.13单rankGloo，完整Meta仅两cache.cuda改CPU，显式初始化6864参数2层D16/Q4/KV2；prefill3+三次单token与完整前缀重算最大logit差6.08e-7，逐层KV/旧前缀/未来槽与投影长度通过，K带RoPE/V不带、独立attention/4步greedy/因果不变及脏尾部通过。固定旧Meta P3/S2多token增量mask真实尺寸报错；一条mask赋值的独立修正[3,2,1]通过。错误前缀/空洞/越界实际反例，cache普通属性不进state_dict/buffers。真实随机811728参数GPT2与固定50257词表，开关各2次新增16，四次ID相同；past两层[1,2,6,8]，增量logits差4.48e-8，无缓存不返回past。未用原GPT2/Llama权重、原1000×10计时、CUDA/多rank/论文或质量指标。下一讲21。

第21讲原5格/2图片展示代码、两11格派生版/原PNG/三Python阅读版及完整固定官方repeat_kv/Attention中文注释交付。共享方向、型号8KV与质量/速度表述在新增说明校正，原文保留；根索引Python链接核心类，原展示版折叠。原字节/历史/3schema/AST/token通过，两版各2原格实际相同，Markdown4六原赋值在明确CPU形状上下文执行。Torch2.5.1+cpu/FairScale0.4.13真实Gloo单rank，完整Meta仅两cache.cuda改CPU；repeat四G/R情形、非连续、R1同对象、expand共享/reshape多G分配与MQA stride0共享、梯度按R求和与double gradcheck通过。真实H4/G4/2/1 Attention（1024/768/640参数）与逐head参考前向、x和四投影梯度及分组einsum前向吻合，错循环配对差0.575。绑定MHA前向同值，KV梯度求和最大差2.12e-8；同头池化恢复，一般MHA池化改变0.351，不冒称无损。prefill3后1/1/1与前缀重算输出差6.94e-17/cache仍G2，H6/G4报错、None默认MHA/D18被整除截断但仍能运行、R0空轴/负R/rank3边界记录。初次自编梯度聚合少reshape错误已修正并重跑，不是源模型bug。GQA论文v3 PDF本地实存，较新架构字幕不当作本讲独立字幕。未用预训练权重、T5 uptraining/指标、GPU测速或多rank通信。下一讲22。
