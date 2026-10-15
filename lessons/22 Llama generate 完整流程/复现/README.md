# 第 22 讲复现记录

状态：**部分成功**。原四格分词代码、全部静态完整性及真实小型固定 Meta Transformer 控制流验证通过；没有原权重质量或论文实验复现。

## 环境与资源

Windows / Python3.12.14 / torch2.5.1+cpu / sentencepiece0.2.0 / nbformat5.10.4 / IPython8.37.0；本地 FairScale0.4.13，单rank Gloo。CUDA不可用，未使用GPU训练或推理。共享环境为 `G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu`。本机约16GB RAM，本次只运行百万参数模型；未下载原7B权重，无语料下载、API调用或付费任务。实际环境版本保存于输出JSON。

模型真实配置：D16、2层、Q4、KV2、V32000、multiple_of4、max_seq_len32、max_batch_size2；1,029,840参数。seed22 normal(0,.02)，norm=1，eval。实际 CPU cache 容量小，非原7B配置。EOS另用真实受控权重：全零注意力/MLP，指定少数embedding/output项，让两行分别在第1/2步生成EOS；没有替换forward为假结果。

## 重做命令

从正式仓库根目录执行（PowerShell）。`--upstream` 是包含固定上游Git提交的本地checkout；`--sources` 含三个完整官方文件及 `fairscale-local`。

```powershell
$python = 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe'
$lesson = 'lessons/22 Llama generate 完整流程'
$sources = 'G:/zzz/Personal ChatGPT 系列/15 Llama 2 text 与 chat completion/资料'
& $python "$lesson/复现/scripts/build_delivery.py" --upstream 'G:/zzz/Llama 入门教程/代码/personal_chatgpt' --sources $sources
& $python "$lesson/复现/scripts/verify_delivery.py" --sources $sources
& $python '工具/check_documents.py'
```

第一条生成原件与派生件，按来源 SHA 检查完整官方文件；第二条才执行验证并记录结果；第三条检查整个项目文档链接与必要讲解章节。公开的核心源码阅读文件只选了相关类方法，并非可独立导入的完整模块。完整外部快照保留本地，SHA在来源清单；换电脑需先获取相同官方提交三个文件，命名 `llama-model.py`、`llama-generation.py`、`llama-tokenizer.py`，并把对应FairScale安装到sources/fairscale-local或按验证器目录布局准备。CPU Torch轮子需使用匹配来源安装，requirements不是原讲师环境锁文件。

## 实际结果

- 原始notebook 17格，其中4代码；两派生版各35格；原字节/metadata/历史输出/3schema/代码AST/非注释token通过。完整generate/text_completion/sample_top_p源码与官方AST一致，原Markdown4/10/13/16逐项一致。
- 原/注释各四格真实运行，stdout和8个ID精确相同。
- 完整官方模型只在内存把两个cache.cuda改CPU；完整generation只把三个device常量改CPU，Tokenizer不改。随机模型所有参数显式初始化，不使用官方空初始化的未定义数值。
- 不等长2/4提示各新增3项，真实调用S为2/1/1/1/1，start为0/2/3/4/5；每次检查无PAD输入。逐步缓存与另一个同权重模型完整前缀重算logits最大差8.940696716308594e-08，容差2e-6；单样本greedy与batch一致。
- 原版新增logprobs全-0。独立AST教学修正仅将整个CE块移到写token之后，选词不变，概率逐步符合真实log_softmax（如首项-10.07020）。两个版本分别保存，不覆盖原件。
- 真实受控EOS样本先输出[2,6]再[6,2]，flags从[True,False]到[True,True]；返回[[],[6]]。已EOS行继续参与是实际观测，不是推测。
- echo、提示含EOS、等长/不等长G0、C32截断、空batch/空row/错维度/超容量/负预算错误、text_completion、四Markdown片段按显式上下文真实执行。
- top_p=.49/.5/0/1的真实multinomial输入、原ID映射/输入未改、负p错误、T=.5/1/2概率熵、负T走greedy、seed220采样重放通过。

详见[实际运行结果](outputs/runtime_verification.json)。其中的 artifact_sha256 绑定当前脚本、三个Python阅读版、三个notebook及分词资产，避免误拿旧报告冒充新版验证。

## 失败与限制

首次验证：SentencePiece0.2.0在本机中文绝对路径字符串下返回文件未找到；资产实际存在。改用切换到代码目录后传 `tokenizer.model`，随后恢复工作目录，未修改Tokenizer算法；完整重跑通过。旧notebook缺cell id会出现nbformat提示，验证副本通过，不更改原元数据消除提示。

原源码的负预算、空batch、空row等实际异常记录在JSON，是边界审计，不当作本脚本失败。第20讲已发现旧Transformer非零起点多token mask尺寸问题，本次正常首次prefill/后续单token路径未触发，未暗改其mask。

未运行：`Llama.build`、原checkpoint、NCCL、GPU、多rank通信、chat模板全套测试、7B/13B/70B、吞吐/显存benchmark、语言质量评估、训练与核采样论文实验。单rank框架调用不能证明多rank通信正确，随机/受控ID不能当原模型文本能力。
