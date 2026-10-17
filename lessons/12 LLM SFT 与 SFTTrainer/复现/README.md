# 第 12 讲复现记录

## 2026-10-17 基础讲解与原论文截图改写

本轮把两张原PDF图放到对应概念旁：InstructGPT arXiv:2203.02155v1第3页Figure2解释SFT在三阶段中的位置；LoRA arXiv:2106.09685v2第1页Figure1解释冻结与两分支。均直接渲染裁切、逐张视觉检查，未重绘；前者是概念补充，不将本课程视为InstructGPT论文实验。[截图页码/坐标/哈希](outputs/论文截图来源.json)。完整PDF仅保留本地，不在笔记仓库重复存放。

[本轮脚本](scripts/verify_foundation_examples.py)从当前解释.md历史折叠区之前提取四段Python原样执行，结果写入[基础改写验证](outputs/基础改写验证.json)，没有覆盖旧实验输出。真实使用Python3.11.0、Torch2.0.1+cpu、Transformers4.30.0、PEFT0.4.0、TRL0.4.7和本地固定GPT-2分词资产，未联网下载模型或全量数据。

- 两条自建instruction/input/output/text分别24、15token；词表50257，EOS50256，逐条decode还原。
- finite/shuffle=False/块长16：41个含EOS token保留32个成2块，丢9个尾token；第一个样本跨块，第二个回答位于该有限buffer尾部。packed保留1个EOS label，普通LM collator遮掉2个真实EOS与9个pad。
- 第一条含EOS长25，全序列shift后24个目标，仅回答为“3”与EOS两个目标。独立全零float64分数loss均约10.82490511970208；回答mask处梯度0，全序列同位置非零，末位置logits梯度0。
- 随机1层hidden8/2heads/50257词表的真实CPU LLaMA，q/v r2/alpha4/dropout0，共64个可训练参数；SFTTrainer默认DataCollatorWithPadding，真实input/labels[2,16]、logits FP32[2,16,50257]，2次AdamW更新，q/v四个A/B均变化，基础参数逐项不变且梯度None，optimizer参数集合精确匹配trainable。没有保存重载或质量测试。

从正式仓库根目录重做（需要上文指定的旧版环境与固定本地分词资产；换机器先按下方历史准备方法取得分词文件，并修改正文第一块路径）：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy/Scripts/python.exe' 'lessons/12 LLM SFT 与 SFTTrainer/复现/scripts/verify_foundation_examples.py'
```

重新截取图可用`scripts/extract_paper_images.py --instructgpt-pdf <原v1.pdf> --lora-pdf <原v2.pdf>`，需要pypdfium2/Pillow。本次用桌面内置PDF Python实际运行。首次生成后检查两张PNG，确认后才可使用`--confirmed-visual-review`记录已检查；脚本不会自动判定图像质量。InstructGPT首次urllib与系统curl请求遇TLS失败，随后Git附带curl开启正常证书验证成功下载固定v1；正文教学首次计算已通过，打印GPT-2字节展示字符时遇Windows GBK编码错误，改验证器stdout为UTF-8后完整重跑成功。这些是工具问题，未修改模型算法。

本轮没有重跑下方历史六行/H16/1024参数的原版及注释版对比、旧adapter保存重载或负例。未执行原代理、Xgen远程tokenizer、7B权重、CUDA4bit/bitsandbytes、全量Alpaca、量化checkpointing、混合精度、双卡、完整一轮、RM/PPO、显存速度或语言质量实验。

## 历史完整交付记录（本轮保留，未重跑）

状态：**部分成功**。原文件、注释和说明版已完成审计，真实旧版 TRL/PEFT/Torch 在 CPU 小模型上完成训练及保存重载验证。未运行 Xgen-7B、原远程 tokenizer、CUDA 4bit 或 Alpaca 全量训练。原 notebook 的历史输出与本次记录分开。

## 文件检查与真实运行

- [文件审计结果](outputs/annotation_verification.json)：27 格/22 代码格，3 schema、AST、Python 阅读版、原字节、元数据及历史输出通过；两种派生 notebook 各 55 格。
- [真实运行结果](outputs/runtime_verification.json)：实际命令、版本、源码/脚本/分词资产 SHA256、原版与注释版执行范围、标签、训练 trace、保存重载与负例。
- [来源取得记录](outputs/source_download.json)：固定提交原 Git blob，54612 bytes、SHA256 `bcbb4c7c4e7e0310a07e8dccf0437eb3e0597031288e2a6332b3c677384dfa51`。本次 raw URL 请求出现 TLS EOF，随后从已下载的上游 Git 仓库读取原对象；没有用 CRLF 工作副本冒充原字节。

实际使用 Windows、Python 3.11.0、Torch 2.0.1+cpu、Transformers 4.30.0、PEFT 0.4.0、TRL 0.4.7、Accelerate 0.20.3、Datasets 2.13.1。独立环境位于 `G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy`；导入及 `pip check` 通过。这是明确选择的旧版参考环境，作者完整依赖版本未锁，不能声称恢复其原环境。报告另记录本次实际安装包源码哈希。

## 重做方法

以下命令从本讲目录执行，Windows 需可用的 Python 3.11。先安装 CPU Torch，再安装其余已固定依赖：

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r 复现/requirements.txt
.venv/Scripts/python.exe -m pip check
```

用于教学的是真实 GPT-2 tokenizer，不是 XgenTokenizer；只需配置和分词文件，不下载 GPT-2 或 Xgen 模型权重。准备固定 GPT-2 revision 的三份文件：

```powershell
.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e', allow_patterns=['config.json','tokenizer_config.json','tokenizer.json'], local_dir='.cache/gpt2', local_dir_use_symlinks=False)"
.venv/Scripts/python.exe 复现/scripts/build_annotated.py
.venv/Scripts/python.exe 复现/scripts/verify_delivery.py --tokenizer .cache/gpt2
```

`build_annotated.py` 由固定原 notebook 重新生成 27 格说明、两种派生 notebook 和纯文本阅读版。`--tokenizer` 指定本地固定资产目录，真实运行只按该路径加载，训练模型随机初始化。以上是重做命令；本次已执行的验证命令在报告中，准备文件的这条 Hub 下载命令没有另行重跑。

如需重新取得原 notebook，可运行：

```powershell
.venv/Scripts/python.exe 复现/scripts/download_original.py
```

它按固定 URL 取得原字节并核对 SHA256。已有上游 Git clone 时，可用 `--upstream-clone` 指向仓库，读取固定提交的 `git show` 原 blob；本次实际使用该方式。原文件已经公开，无需为了阅读再次下载。

本次验证从正式仓库根目录执行：

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy/Scripts/python.exe' 'lessons/12 LLM SFT 与 SFTTrainer/复现/scripts/verify_delivery.py' --tokenizer 'G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/验证tokenizer缓存/gpt2'
```

训练和 adapter 临时文件由 TemporaryDirectory 管理，不写入阅读目录，不上传随机权重。报告保存实际数值与文件名；临时 checkpoint 随验证结束清理。

## 执行范围和观测结果

原版与注释版各整格执行 0、1、5、6、7、8、9、12、14、15、18、19、20、22、23、26，共 16 格。提供六行自建 Dataset、真实 GPT2TokenizerFast、随机一层 LLaMA 上下文；跳过原 3 代理、4 Alpaca 下载、11 XgenTokenizer、13 7B 四比特加载。原 24 的 fp16=True 实际在 CPU 抛 ValueError，失败尝试只重定向输出目录。成功变体另改 fp16=False/临时目录，添加 max_steps=2/report_to=none；25 仅改块长1024→32。原公开文件均未改这些设置。

小模型 hidden16、intermediate32、1layer/2heads；输入 embedding 与独立 lm_head 都从 `[50320,16]` 缩到 `[50257,16]`，基础参数 1610836。原22对普通FP16模型真实冻结并升FP32，没有伪造 is_loaded_in_4bit，未开启量化 checkpointing 分支。原23的r16/alpha32/dropout.05配置真实注入默认q/v，共1024可训练参数；B初始0，初始输出与关闭adapter时一致，B梯度非零、A梯度0，基础梯度None。

旧版 packing 在独立 shuffle=False/infinite=False 测试中将六行拼成240token，保留7个32token块/224token，丢16token；len(dataset)却报告6原行。packed默认DataCollatorWithPadding保留5个EOS标签。两条明确追加EOS的不同长度文本，经DataCollatorForLanguageModeling后2个真实EOS和7个pad都置-100；真实EOS的attention_mask仍为1。另建实际非packed SFTTrainer，核对默认LM collator和 `[4,32]` batch，未再运行第三次训练。completion-only collator与packing=True真实构造失败，错误保留。

模型内一次shift的loss与手工CE同为10.84434986114502。两版Trainer均两次AdamW步骤、forward `[4,32]`、logits FP32 `[4,32,50257]`、loss FP32标量，优化器只含adapter，基础参数逐项不变。两步计划warmup取整1步，forward时lr分别0/0.0002；A未发生非零更新，B最大变化约0.00020027。平均train_loss10.8355712890625，不是语言质量指标。两版adapter/同输入logits/标签/训练trace精确一致，已包装PeftModel在SFTTrainer中保持同对象，未再次注入。

真实epoch保存产生checkpoint-2/adapter_model.bin；独立adapter保存含q/v四个A/B权重，不含基础embedding/head。匹配基础状态重载后所有模型tensor精确一致、tokenizer IDs一致；logits最大差1.1920928955078125e-7，通过rtol1e-5/atol1e-6。最初零误差断言失败，已据状态逐项核对记录实际误差，原因未隔离。故意用50320行基础模型也成功加载同adapter，但输出 `[1,32,50320]`，证明只看加载成功不足以证明恢复了训练词表结构。

## 保留的早期标准库验证

[verify_sft.py](scripts/verify_sft.py) 与 [教学验证.json](outputs/教学验证.json) 保留 Python3.7.6 的旧算术/边界记录。旧环境的 torch namespace 不完整；旧脚本提取 ConstantLengthDataset 类，注入整数 tokenizer 和 `LongTensor=list`，它不是实际 PyTorch tensor 测试。现在新增真实运行报告，不回填或覆盖旧证据。

```powershell
python 复现/scripts/verify_sft.py --sources 'G:/zzz/Personal ChatGPT 系列/12 LLM SFT 与 SFTTrainer/资料' --output 复现/outputs/教学验证.json
```

此命令仅重做旧验证；本次没有再执行或覆盖该旧输出。原训练仍需独立锁定 Xgen/Alpaca revision、CUDA/bitsandbytes 和实际设备，验证原1024块/整轮训练、量化checkpointing、显存速度与回答质量；这些尚未运行。
