# 第04讲全部单元注释与固定官方源码验证

状态：**部分成功**。18/18格说明、8/8代码格中文注释（含空格）、6/6 Markdown源码块中文注释；原单元、历史输出、AST和schema核对通过。原7B量化加载未执行，官方普通Linear路径在小型CPU BLOOM中通过真实验证。

原notebook、两种说明/注释版及[Python阅读版](../代码/tutorials/04_PEFT_lora_代码注释.py)均已直接放入公开仓库；入口见[本讲目录](../README.md)。阅读版无JSON或历史输出，以 `# %%` 分格，六段Markdown源码作为注释保留。

| 检查 | 实际结果 |
|---|---|
| 原文件下载 | 12679字节，SHA-256为 `712d33af261bf54ebac3be88bad46e1a180c8d79464da985e0d01b69cbc79146`，与固定来源一致 |
| 注释覆盖 | 18格均解释，8个Python格及6段Markdown源码详注，两种生成版各37格 |
| 原单元/历史输出、schema、AST | 全部通过；只增加注释，原dataclass缺陷保持 |
| Python阅读版 | AST与原8格拼接一致；不携带JSON或历史输出 |
| 实际执行 | 原版与注释版各7格（含空格）在CPU教学模型上下文执行；原12的7B加载未执行 |
| 六段Markdown源码 | 真实类映射、独立属性上下文、目标扫描均核对；dataclass按预期报TypeError |
| 固定官方源码 | 直接导入PEFT v0.3.0 checkout，核对版本、路径、Git提交和源码哈希 |
| 注释版与原版比较 | 调用链、匹配、模块/参数身份、冻结、shape、adapter状态一致，数值观测差小于1e-7 |
| 未复现范围 | 原7B量化加载、8bit/CUDA kernel、老师dev环境、训练与论文指标 |

表格已列出结论，下面的JSON链接仅作补充记录。

## 来源及两个环境

[来源清单](../来源清单.json)固定教程与PEFT提交、原文件及源码哈希。老师PEFT0.4.0.dev0没有Git SHA；参照源码选官方v0.3.0、7120c9a2636f93c8db3fc4ae466e02f338ecead8。本次直接导入本地checkout的src/peft，并检查版本、路径、Git HEAD和无未提交源码修改；不是执行环境里已安装PEFT0.12.0的路径。

注释生成/审计使用Windows、Python3.7.6、nbformat5.0.4。真实注入使用独立Python3.12.14、torch2.5.1+cpu、Transformers4.44.2、Accelerate0.34.2；它们比源码发布时期新，所以仍不声称老师环境精确重建。requirements.txt记录CPU依赖，PEFT必须通过`--peft-source`指向上述固定checkout。

没有安装或运行bitsandbytes/CUDA；没有7B checkpoint、真实词表、数据集或API key。教学模型是随机BLOOM：H16、2层、2头、V32、CPU FP32、非量化。RAM/磁盘总量未测量，没有据此启动大模型。

## 文件与命令

build_annotated.py为18格插说明，生成两份37格notebook及Python阅读版；代码注释版进一步在8格Python和6段Markdown源码只加中文注释。原17空格保留，原7 dataclass缺陷不修复。verify_annotations.py比较原单元/元数据/历史输出、逐字新增注释、8个代码AST和6个源码块AST，并核对schema、阅读版AST和插入说明的相对链接；不执行原7B加载。

verify_injection.py直接调用固定官方库；原版与注释版分别执行9/11/13/14/15/16/17，上下文明确为教学模型，12未执行。Markdown2类型字典、8架构字典与真实官方类/常量比较；4属性片段使用独立属性上下文，最终包装身份由真实构造器验证；8扫描结果用trace收集，没有重写匹配表达式，循环行号由AST定位以兼容新增注释；7提供真实父类与导入后仍出现预期TypeError。两版的完整CPU验证结果逐项比较。

下列PowerShell命令从公开仓库根目录执行；其他机器需替换Python和官方checkout路径。下载脚本只获取固定原notebook；生成/审计脚本不联网，注入验证以HF离线方式执行，不下载7B权重。

```powershell
& 'D:/anaconda/python.exe' './lessons/04 PEFT LoRA 源码/复现/scripts/download_original.py'
& 'D:/anaconda/python.exe' './lessons/04 PEFT LoRA 源码/复现/scripts/build_annotated.py' --notebook './lessons/04 PEFT LoRA 源码/代码/tutorials/04_PEFT_lora.ipynb' --output-dir './lessons/04 PEFT LoRA 源码/代码/tutorials' --record './lessons/04 PEFT LoRA 源码/复现/outputs/code_unit_coverage.json'
& 'D:/anaconda/python.exe' './lessons/04 PEFT LoRA 源码/复现/scripts/verify_annotations.py' --directory './lessons/04 PEFT LoRA 源码/代码/tutorials' --record './lessons/04 PEFT LoRA 源码/复现/outputs/annotation_verification.json'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './lessons/04 PEFT LoRA 源码/复现/scripts/verify_injection.py' --notebook './lessons/04 PEFT LoRA 源码/代码/tutorials/04_PEFT_lora.ipynb' --annotated-notebook './lessons/04 PEFT LoRA 源码/代码/tutorials/代码注释.ipynb' --peft-source 'G:/zzz/Personal ChatGPT 前七讲/共享资料/peft-v0.3.0' --record './lessons/04 PEFT LoRA 源码/复现/outputs/injection_verification.json'
```

固定源码入口为[官方提交](https://github.com/huggingface/peft/tree/7120c9a2636f93c8db3fc4ae466e02f338ecead8)。本地checkout已存在，本次没有改变其中源码或Git状态。

[下载记录](outputs/source_download.json)固定原notebook来源和哈希。原notebook与两种说明/注释版均在本仓库 `代码/tutorials`，保留作者历史输出；本次实际验证另记，不覆盖历史执行计数。

## 实际证据

- [覆盖记录](outputs/code_unit_coverage.json)与[注释审计](outputs/annotation_verification.json)：18格、8代码格、6源码块，原单元及输出保持，注释-only、AST及schema通过。
- [真实固定源码记录](outputs/injection_verification.json)：关键调用get_peft_model→add_adapter→_prepare_lora_config→_find_and_replace→两次_replace_module→mark_only_lora_as_trainable确实发生；两层从torch.nn.modules.linear.Linear变为peft.tuners.lora.Linear，旧weight/bias是相同Parameter对象，外层复用原BLOOM。
- A[16,16]、B[48,16]、scale2，总2048个可训练参数，只有lora_参数可训练。零B时全模型logits[1,3,32]与注入前逐元素相等。第一次反向A梯度为0，B梯度范数94.1398，基weight/bias梯度为None。
- 为测试非零增量，**人工设置B为非零随机值，没有进行训练**。目标层输出[2,3,48]与手写双分支误差0；disable还原基输出；merge与未合并前向误差1.49e-7，unmerge在容差内恢复权重和输出。不能将浮点merge/unmerge称为逐bit无损。
- 列表后缀与'.*query_key_value'完整正则均匹配2层；裸'query_key_value'和错误['q_proj']匹配0层，并在真实get_peft_model中产生ValueError。
- 原Markdown7的数据类在补齐真实导入/父类后产生`TypeError: non-default argument 'target_modules' follows default argument`；官方完整定义有field(default=None,...)，不是这个缺字段的节选。注释版保留缺陷。

普通CPU Linear、引用身份与梯度链验证不能证明8bit的state搬运、CUDA量化kernel、原7B显存/性能、任意卷积适配或老师dev环境。原notebook没有训练loop或评估指标，本次也没有训练或评价任务质量。


## 2026-10-17：基础说明、原图与正文例子

正文新增对象引用例子，以及直接调用固定官方PEFT v0.3.0的随机1层H4、2头、词表8的CPU BLOOM例子。验证脚本从当前解释.md原样提取两块代码，核对官方checkout提交、源码无修改、实际导入路径；不下载权重与数据。

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './lessons/04 PEFT LoRA 源码/复现/scripts/verify_foundation_examples.py' --peft-source 'G:/zzz/Personal ChatGPT 前七讲/共享资料/peft-v0.3.0'
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' './lessons/04 PEFT LoRA 源码/复现/scripts/extract_paper_images.py' --pdf 'G:/zzz/Personal ChatGPT 前七讲/02 LoRA - Low-Rank Adaptation of Large Language Models/资料/LoRA_2106.09685v2.pdf'
```

实际三个身份判断均True：原基模型保留、目标子层替换、旧weight对象复用。零B时全模型logits与注入前逐元素相等。r1/alpha1/dropout0下，人工将A置1、B置0.5，输入[1,2,3,4]，12个输出的修正量各为5；共16个可训练数，仅两个A/B张量。一次输出求和的backward使A/B梯度非零，所有原参数无梯度，基weight与快照相同。没有优化器更新，这不是训练结果。结果与源码哈希见[本轮记录](outputs/基础改写验证.json)。

运行出现Transformers关于tuple形式past_key_values将弃用的提示，脚本仍以退出码0完成；此处的固定旧PEFT与较新CPU依赖不等于老师dev环境。没有重跑上文H16历史实验，也没有执行原7B量化加载、CUDA或质量评测。

Figure1与公式3均从固定LoRA v2 PDF直接渲染裁剪，已逐张实际查看。原图完整保留x/h、箭头、W/A/B及公式编号；正文解释图中方向与方阵/实际QKV矩形差别，也区分论文正态初始化与官方代码Kaiming均匀初始化。来源/页码/裁剪坐标/哈希见[截图记录](outputs/论文截图来源.json)。其他机器重做时从记录中的官方URL下载同版本PDF，再修改--pdf路径。原notebook及原注释代码保持不变。
