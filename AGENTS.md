# 本项目工作约定

用户要求按 paper-code-deep-reading skill 完成 Personal ChatGPT 第01–30讲，中文解释公式、shape、API及全部notebook单元。上游固定提交为 `7fbc29824171fd07dc2c30fe922e61ebc018ad1d`，单元从0编号。

每讲公开交付原notebook及必要图片、代码注释.ipynb、原代码_带说明.ipynb、Python阅读版、解释.md、代码单元解释.md及复现记录。原版按下载字节保留；说明版不改原单元；注释版只增加注释，保留历史输出，核对schema、AST和可行的执行。重要Markdown源码、命令、空单元也需说明。不得将教学验证或历史输出当成大模型复现。

用户已授权公开仓库 https://github.com/ghost-yu/personal-chatgpt-notes ，分支main；每完成一讲，更新索引，运行相关验证及 `工具/check_documents.py`，创建独立commit并push。不得仅提交本地。保留已有解释和验证证据，字幕附件、权重及完整外部源码快照保留本地。

当前第01–19讲主解释已存在；第01–09讲已按完整公开文件格式交付，下一讲第10讲。第10–18讲仍需核验全部单元并补齐公开注释文件；第19讲15个代码单元说明已完成，代码内注释待补；第20–30讲待完成。不能以主文存在代替全部代码交付。一次完成一讲并独立提交。

第03讲原版与注释版各26格在CPU小型BLOOM上下文执行并训练2步；第04讲直接使用固定官方PEFT v0.3.0，核对注入、引用、冻结、前向/梯度及adapter状态。均未运行原7B配置。具体证据见各讲复现目录。

第05讲全部27格说明、8格代码详注，两种派生notebook各55格；原版与注释版各尝试8格，7格成功，CUDA设备格18在PyTorch2.5.1+cpu中预期失败。真实CPU位模式、finfo/nextafter、舍入、溢出/下溢、存储核对通过。双问号源码检查使用实际IPython pinfo2，展示hook只捕获文本以避免外部分页器等待。未运行GPU/TPU算术或训练，旧记录保留。

第06讲全部28格说明/18格详注，两种派生notebook各57格，原语句/历史输出、IPython转换AST及阅读版AST通过。原版和注释版各13格在明确CPU教学上下文执行；原inference调用在.cuda()失败，补充仅将该表达式改为.to('cpu')后真实生成。固定PEFT0.3.0、1层H16 LLaMA、真实96词SentencePiece/LlamaTokenizer和随机非零r2/alpha4四目标adapter，本地保存/重载/冻结与4beam生成核对一致。原7B/8bit/远程adapter/多卡/交互与质量评测未运行，勿把随机输出当作原模型能力。

用户要求保持阅读入口清晰。根目录README是唯一总索引；各讲README只突出解释、注释notebook与Python阅读版。来源链接放在解释中，维护资料在资料目录，检查工具在工具目录。不要重新生成只有来源链接的原代码.md；保留原始notebook、说明版及验证证据。

第07讲19格说明/12格代码与Markdown1片段详注，两种派生notebook各39格；AST/schema/历史输出核对通过。原大CNN仅meta检查参数476938250及RGB224尺寸错误；四个通道常量改为8/16的小型CNN、8合成样本/batch4/epoch1，原train/evalute体真实执行。原CUDA AMP在CPU禁用，补充只改autocast表达式并另建CPU scaler，真正FP16 logits/FP32 loss和梯度、两次Adam更新、scale128→256、Inf跳步回退128通过；原/注释版一致。未下载MNIST或复现3090 batch128→256显存实验。

第08讲主56格/37代码格（含空55）+bitsandbytes辅助3格/1代码格，三张原图、四份派生notebook、两份阅读py已交付。全部59格说明/38代码AST、6schema/历史输出核对；原/注释各32主格实际CPU执行，53/54另在独立随机2960参数小OPT/本地词表上下文执行，50代理/51远程tokenizer/52量化加载跳过。原矩阵实际float64缩放、无舍入；独立int8存储/int32累加/FP16异常支路、零尺度/溢出、真实配置互斥和总长/新增token区别通过。未安装bitsandbytes、未下载OPT6.7B或运行CUDA/论文指标。

第09讲basics39+train27+辅助gpt2 6共72格、52代码格；三原notebook/原图、六派生notebook、三py交付。51可解析AST（含shell转换）/9schema/历史输出核对；原basics22悬空逗号SyntaxError保留，阅读版整格注释，多行str同值排版，Markdown2代码详注并实际核对。固定本地GPT2/T5资产仅改加载路径；真实2000行JSONL IterableDataset、原默认batch1000，两轮常量审计补充得到320/63merges和336/79merges，继续消费、旧对象不变、保存重载、Unicode/byte/末批/耗尽通过。basics36长度并列输出可变，保留双方原列表并核对排序契约，其余受控结果一致。原语料/代理/讲师Hub/语言模型训练未运行。下一讲10。
