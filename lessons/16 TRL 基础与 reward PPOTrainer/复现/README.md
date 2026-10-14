# 第16讲复现记录

## 新增完整交付与真实PPO运行

状态：**部分成功**。原42格/28代码格（含空41）、Markdown10三命令及31概览片段完整说明和注释；五张配图、两85格派生版及Python阅读版交付，原字节/历史/元数据/schema/AST通过。新增真实PPOTrainer与value forward，原/注释两版及独立修正版各四次Adam更新。原GPT2预训练权重、learned RM及完整RLHF未运行。

环境：Windows/Python3.11.0、torch2.0.1+cpu、Transformers4.30.0、TRL0.4.7、Accelerate0.20.3；[直接依赖](requirements.txt)、[完整包列表](outputs/environment_freeze.txt)。使用已有sft-legacy隔离环境，不修改默认Python。pip check通过，CPU CUDA不可用。

原notebook由已下载clone的固定Git blob取得，SHA256为1d0ec8e4f196914257158d18a48746b12561fa68100aeb778105720bd8b8d474；旧清单CRLF hash保留并注明。根tutorials/trl_basics.ipynb与trl_hf版字节相同，不公开重复副本。五图源自上游imgs，字节保留，放到代码/tutorials/imgs，使原../imgs调用有效。详见[原件记录](outputs/source_download.json)。

真实slow GPT2Tokenizer的vocab/merges从第09讲固定tokenizer.json导出；输入SHA256为8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6，revision607a30d783dfa663caf39e06633721c8d4cfcd7e。导出没有训练新词表，也不证明作者Hub全部配置一致。随机基础GPT2为H16/1层/2heads/V50257/128位置，value包装809489参数、head17；临时保存后通过原类方法加载，结束后释放随机checkpoint。

从仓库根目录重做：

```powershell
py -3.11 -m venv "$env:TEMP/lesson16-env"
$lessonPython = "$env:TEMP/lesson16-env/Scripts/python.exe"
& $lessonPython -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.0.1+cpu"
& $lessonPython -m pip install -r "lessons/16 TRL 基础与 reward PPOTrainer/复现/requirements.txt"
& $lessonPython -m pip check
$lessonTokenizer = "$env:TEMP/lesson16-tokenizer.json"
Invoke-WebRequest -Uri 'https://huggingface.co/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json' -OutFile $lessonTokenizer
& $lessonPython "lessons/16 TRL 基础与 reward PPOTrainer/复现/scripts/download_original.py"
& $lessonPython "lessons/16 TRL 基础与 reward PPOTrainer/复现/scripts/build_annotated.py"
& $lessonPython "lessons/16 TRL 基础与 reward PPOTrainer/复现/scripts/verify_delivery.py" --tokenizer-json $lessonTokenizer
python "工具/check_documents.py"
```

下载命令是重做方式，本次使用已有hash匹配tokenizer。原件取得时download_original.py使用 `--upstream-clone`；没有将Git blob读取当作新HTTP下载。实际完整PPO验证命令：

```powershell
& "G:/zzz/Personal ChatGPT 系列/共享运行环境/sft-legacy/Scripts/python.exe" "lessons/16 TRL 基础与 reward PPOTrainer/复现/scripts/verify_delivery.py" --tokenizer-json "G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/gpt2-tokenizer/tokenizer.json"
```

验证先检查tokenizer与实际已安装PPOTrainer源码hash（8142e1296171adc0a9f664ed0a241d4e7313ad0fac6987b2a093b0982489a407），要求TRL0.4.7/Transformers4.30.0，不模拟模型/优化器API。只有原格18的三处gpt2字符串变成本地目录，其他27格原语句执行，包括历史生成参数、固定score1.0和原错误step。两版rollout、mask、数值stats、active状态精确一致；时间字段不作等价指标。

[注释审计](outputs/annotation_verification.json)证明42格/28AST/3schema、85格派生版、阅读版AST与Markdown仅加注释/命令语义；[实际运行](outputs/runtime_verification.json)记录所有执行格、图像bytes、实际PPO与修正版：

- 原query7，return_prompt=True返回27，step拼34；真实shift后的logprob/value/mask为[1,33]，有效目标27，第二遍query被优化。
- 原默认batch1/mini1/ppo_epochs4，实际四次Adam step；active和value head变化，reference状态精确不变，optimizer仅active。显式ref仍有18个requires_grad Parameter，但grad均None。
- 原1D基础forward实测hidden[34,16]、logits[34,50257]，无梯度value[34]；原历史H768输出保留不改。
- 独立return_prompt=False重新初始化相同基础/seed，20续写ID与原返回去前缀精确相同，拼27、mask[1,26]有效20；同样四次更新，reference不变。两种目标长度不同，loss不能作为优劣对照。
- 首轮在两版运行后，修正版加载传pathlib.Path被旧TRL拒绝；改str后完整检查退出0。这是旧包装参数类型限制，不是原算法失败。

没有训练RM、读取偏好数据或运行原GPT2权重。固定score1.0和随机续写不能证明偏好对齐；Llama2/Guanaco的Markdown SFT shell、GPU/多卡、质量/完整论文实验未运行。Markdown31是省略上下文与缩进层级的概览，保留原片段，不当完整可执行程序。

## 早期数学/列表验证（保留原范围）

状态：**部分成功**。真实GPT-2 tokenizer编码/往返成功，重复prompt与response mask边界、sampled KL和完整KL的差别、末token奖励、GAE及PPO clipping数学断言通过。静态核对TRL v0.4.7源码与配置。没有执行PPOTrainer、value head tensor forward、GPT-2权重或真实optimizer更新。

工作目录为本目录，标准库部分可以执行：

```powershell
python scripts/verify_ppo.py --output outputs/教学验证.json
```

本次实际使用已安装tokenizers0.13.3的Python3.7.6：

```powershell
python scripts/verify_ppo.py --tokenizer-json "G:/zzz/Personal ChatGPT 系列/09 BPE 与 GPT-2 tokenizer/资料/gpt2-tokenizer/tokenizer.json" --sources "G:/zzz/Personal ChatGPT 系列/16 TRL 基础与 reward PPOTrainer/资料" --output outputs/教学验证.json
```

`--tokenizer-json`加载已有固定GPT-2词表；未指定时报告明确写历史ID fixture，不能声称实际编码。`--sources`对已保存官方Python源码做AST检查。数学例子的20个response ID是自造整数，不是本次生成；每token reward/advantage也是教学数值，未冒充模型统计。

本机默认torch是不完整namespace，缺少Tensor/tensor API。RTX4060 Laptop8GiB；没有安装大型GPU框架或下载模型权重。重复prompt由原保存输出、源码拼接/mask逻辑和当前列表检查共同确认；仍未通过真实Trainer运行验证修正版全部依赖。

真实修复方向：`return_prompt=False`返回纯response，或按每条query的token长度切去前缀；传入step的query/response是两个一维tensor列表，score是每样本一个标量；明确配置batch/mini-batch/seed，固定reference，并记录RM及生成参数。反复训练和前后评估才可证明PPO效果，当前没有做这些实验。
