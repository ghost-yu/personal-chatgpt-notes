# 第27讲验证记录

状态：**部分成功**。静态保真、实际CPU小模型PPO与独立数学检查通过；未复现原GPT2能力或OpenRLHF分布式训练。

## 本次实际执行

2026-10-16，Windows，Python3.12.14，torch2.5.1+cpu、Transformers4.44.2、TRL0.8.6、Accelerate0.34.2、datasets2.19.2。没有安装Ray/DeepSpeed，也没有下载预训练GPT2模型或调用收费API。复用既有共享CPU环境及第23讲隔离安装的旧TRL依赖，没有升级全局环境。主机物理内存约15.69 GiB，G盘检查时余量约22.08 GiB；本次使用临时小模型，不要求原GPT2训练显存。

本次命令从正式仓库根目录执行：

~~~powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/27 TRL RLHF PPOTrainer 与 OpenRLHF/复现/scripts/build_delivery.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'lessons/27 TRL RLHF PPOTrainer 与 OpenRLHF/复现/scripts/verify_delivery.py'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' '工具/check_documents.py'
~~~

build_delivery只重建派生文档；原notebook须预先从固定Git对象提取，不能用会自动CRLF转换的工作树复制冒充原字节。verify_delivery会重新执行两版训练和诊断，报告写到outputs/delivery_verification.json。临时checkpoint自动清理，公开目录不保存模型权重。

## 原样执行与明确变体

原文件33格，其中25格代码。两版各22格按原语句执行，含4个仅注释格和末尾空格；7/12格只将三个gpt2加载字符串换为同一个临时本地路径（两模型、一tokenizer）；0格代理/NCCL环境设置跳过，其Image导入在外部提供。其余配置、generate、score与step原语句保留。Image(url=...)原代码实际只构造展示对象，另以真实图片文件/hash和人工查看证明三张图内容；不把构造url当作浏览器渲染成功。

本地checkpoint是随机GPT2：V=50257、H=16、1层、2 heads、context64、全部dropout=0，另加17参数value head，共808465参数。checkpoint初始化seed270，每版运行外置seed27；没有修改notebook加入seed。真实GPT2词表来自第09讲已保存的tokenizer.json，经tokenizers原生BPE模型导出vocab.json/merges.txt，供真正GPT2Tokenizer读取。

一次step使用原默认ppo_epochs=4，真实底层Adam step hook计数4。optimizer只包含active可训练参数；所有ref状态前后精确不变，active骨干与value head都有变化。两版全部最终active/ref状态、展示输出和非墙钟统计精确一致。没有强行要求计时相同。

## 已核对内容

| 检查 | 实际证据 |
|---|---|
| 原notebook及三PNG | 与固定7fbc298 Git对象逐字节相同 |
| 两个派生notebook | 每个67格，原元数据/历史输出不变；原代码说明版原单元相同 |
| 注释版与Python版 | 原代码AST及非注释token相同；三份notebook schema通过 |
| TRL核心源码 | 10个完整类/函数对象，含整个PPOTrainer类；与安装文件、dist-info/RECORD hash相同 |
| OpenRLHF源码 | 固定fbc1eab历史提交的8个完整对象AST相同；非当前main |
| query/response | Q=7、T=20、拼接27、logits[1,27,50257]；shift[1,26]，mask有效[6,26) |
| forward与mask | 手工log_softmax/gather精确；纯response返回，无重复prompt |
| old/reference | 更新后ref logprob差0.004940，下一轮old等于当前策略时ratio=1 |
| 奖励与KL | 负sample log比、真正full词表KL、末有效token才加score |
| GAE | 三步raw A=[.72725,.555,.9]，returns=[.92725,.955,1]；优势白化和detach |
| clip与梯度 | 四种符号/越界分支，policy loss .05、梯度[0,-.175,.325,0]；value clip反例 |
| beta控制 | 高KL后.2004，再低KL后.1999992 |
| checkpoint | 实际保存与重载，基础及value head所有状态/logits/value精确相同 |
| OpenRLHF数学对应 | 原AST纯函数/类在真实torch中运行；reward/GAE与单序列loss符合对应条件 |
| 框架差异 | score20被OpenRLHF截为10；长度2/1时全token均值4.3333与逐序列均值5.5 |

OpenRLHF执行范围只含compute_approx_kl、compute_reward、log_probs_from_logits、masked_mean、PolicyLoss、ValueLoss和get_advantages_and_returns；完整make_experience仅静态检查，内含cuda/模型调用未运行。没有stub、假的Ray/模型或替代框架冒充真库启动。

## 观察到的边界和修复

左、右padding都得到有效动作数[5,3]；右padding batch与单条logprob一致，原左padding短样本差约0.058259。独立加入基于attention_mask的position_ids后恢复一致；该补充不改原文件。全零response mask真实IndexError；scores中Python float真实ValueError；仅1个有效token的旧masked_var真实NaN。B=1时分数及序列长度无偏标准差统计也为NaN，不是本次训练loss失败；报告用明确字符串保留这种事实，保持标准JSON格式。

前三次审计/验证发现并修复：工作树CRLF与Git原字节不同、抽取方法时dedent会改变docstring内部空格、验证器误判基础模型state_dict前缀，以及beta断言dtype不同。随后完整重新生成/审计/训练通过。这些不是源模型错误。实际加载提示基础GPT2暂时不用v_head键，TRL包装器随后读取，已经状态比较验证，没有忽略警告就宣称成功。

## 在其他机器复做

requirements.txt列本讲核心依赖，使用独立虚拟环境安装；CPU torch可按PyTorch官方CPU wheel方式安装。脚本不会自动拉取模型/语料；需要准备：课程Git仓库及固定commit、真实GPT2 tokenizer.json、TRL0.8.6完整安装和RECORD、OpenRLHF固定三份源码。脚本可传--source-dir、--upstream、--libs、--tokenizer-json覆盖本机默认路径；--libs是实际安装目录，须能看到trl-0.8.6.dist-info及trl文件，通常是虚拟环境Lib/site-packages或pip --target输出目录。

source-dir结构：trl-0.8.6/trl下对应trainer/ppo_trainer.py、trainer/utils.py、models/modeling_value_head.py、core.py；此外openrlhf-models-utils.py、openrlhf-loss.py、openrlhf-experience_maker.py需对应来源清单中的固定commit和hash。构建时另需trl-LICENSE、openrlhf-LICENSE。原件与图已在公开仓库，无需从历史输出反向重建。验证环境与本地资产是显式前置，不承诺陌生电脑只一条命令就具备全部条件。

## 未运行

原GPT2预训练权重与原语言输出、真实Reward Model/偏好数据训练、人类或自动质量评测、PPO原论文MuJoCo/Atari、OpenRLHF Ray/DeepSpeed/vLLM/CUDA/多机、性能表、长期稳定性。成功的是教学规模真实模型优化和明确范围的源码/数学验证，不是完整论文或生产RLHF复现。

来源补充：TRL v0.8.6远端标签源码下载实际遇到TLS握手失败，本次只将安装RECORD与源码hash作为精确验证证据，不声称已与远端tag逐字节比对。OpenRLHF历史源码/README/license下载成功。

### 整理工具的修正记录

初次整理复制了上游工作树CRLF notebook，字节审计发现它与固定Git blob不同；改为直接读取Git对象原字节后通过。源码方法抽取时也修复了会改变docstring内部空格的通用dedent；随后用AST核对完整对象。验证工具最初误认为基础state_dict键都带pretrained_model前缀，以及beta断言混用float32/float64，这些是验证器错误，已更正并重新完整执行，不是模型或论文错误。没有删掉失败边界，也没有把原训练效果夸大为完整复现。

## 讲解改版前的运行说明

## 10. 一次小模型 PPO 更新

结果见 [复现说明](README.md)。本次执行使用Python3.12.14、torch2.5.1+cpu、Transformers4.44.2、TRL0.8.6和真实GPT2词表50257。一层、H=16、2 heads、上下文64的随机GPT2加17参数value head，共808465参数；外置seed27，checkpoint初始化seed270。不是预训练GPT2权重。

每版25代码格中，22格按原语句执行（含空/仅注释格），7/12两格仅三个gpt2加载字符串替换为同一个临时本地资产路径；0格代理/NCCL设置跳过，其Image导入在外部提供。**没有声称25格原样全部执行**。两版输出、训练统计（排除墙钟耗时）及最终全部active/ref状态相同；两版各4次真实Adam更新，骨干与value head变化、ref不变、optimizer参数只含active可训练参数。checkpoint实际保存重载后，基础权重/value head以及logits/value精确相同，临时权重不公开。

额外诊断覆盖shift/gather、mask边界、old/ref区别、sample/full KL、末token score、GAE/白化、policy/value裁剪和梯度、beta更新、不同长度padding、OpenRLHF归约/score裁剪差异。注意以下真实边界：

- B=1时ppo/std_scores、queries_len_std和responses_len_std使用无偏样本标准差，得到NaN；这些是统计量，不是证明本次训练loss为NaN。记录保留含义，JSON用明确字符串表示未定义统计。
- 有效mask全零时compute_rewards找不到最后位置，实际IndexError；scores写Python float而非tensor时安全检查报ValueError。
- masked_var只有1个有效token时N/(N-1)导致NaN；不能因为batch有一条样本，就假定必失败，关键是有效token数。
- 左padding实际差异与独立position_ids修正分开记录；原文件没有悄悄修正。
- 保存含value head的checkpoint重新加载时，基础GPT2加载阶段会提示v_head键不用；随后TRL包装器恢复head。已实际比较全部状态，不能只据这条提示判断head丢失。
