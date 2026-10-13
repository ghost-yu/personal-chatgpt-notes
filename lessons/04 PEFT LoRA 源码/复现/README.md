# 第04讲全部单元注释与固定官方源码验证

状态：**部分成功**。18/18格说明、8/8代码格中文注释（含空格）、6/6 Markdown源码块中文注释；原单元、历史输出、AST和schema核对通过。原7B量化加载未执行，官方普通Linear路径在小型CPU BLOOM中通过真实验证。

## 来源及两个环境

[来源清单](../来源清单.json)固定教程与PEFT提交、原文件及源码哈希。老师PEFT0.4.0.dev0没有Git SHA；参照源码选官方v0.3.0、7120c9a2636f93c8db3fc4ae466e02f338ecead8。本次直接导入本地checkout的src/peft，并检查版本、路径、Git HEAD和无未提交源码修改；不是执行环境里已安装PEFT0.12.0的路径。

注释生成/审计使用Windows、Python3.7.6、nbformat5.0.4。真实注入使用独立Python3.12.14、torch2.5.1+cpu、Transformers4.44.2、Accelerate0.34.2；它们比源码发布时期新，所以仍不声称老师环境精确重建。requirements.txt记录CPU依赖，PEFT必须通过`--peft-source`指向上述固定checkout。

没有安装或运行bitsandbytes/CUDA；没有7B checkpoint、真实词表、数据集或API key。教学模型是随机BLOOM：H16、2层、2头、V32、CPU FP32、非量化。RAM/磁盘总量未测量，没有据此启动大模型。

## 文件与命令

build_annotated.py为18格插说明，生成两份37格notebook；代码注释版进一步在8格Python和6段Markdown源码只加中文注释。原17空格保留；原7 dataclass缺陷不修复。verify_annotations.py比较原单元/元数据/历史输出、逐字注释新增行、8个代码AST和6个源码块AST，并验证schema；不执行原7B加载。

verify_injection.py直接调用固定官方库。原9/11/13/14/15/16/17在明确的教学模型上下文执行；原12未执行。Markdown2两张类型字典、8架构字典使用真实官方类/常量比较；4属性片段使用独立属性上下文，最终包装身份另由真实构造器验证；8扫描片段的结果用trace收集，没有重写匹配表达式；7提供真实父类与导入后仍出现预期TypeError。

下列PowerShell命令的工作目录为本文件所在复现目录；其他机器需替换路径。完整原版、两种生成版和原历史输出保存在本地配套 `代码/tutorials`；公开仓库保留自编说明、脚本及验证记录。

```powershell
& 'D:/anaconda/python.exe' './scripts/build_annotated.py' --notebook 'G:/zzz/Personal ChatGPT 前七讲/04 PEFT LoRA 源码/代码/tutorials/04_PEFT_lora.ipynb' --output-dir 'G:/zzz/Personal ChatGPT 前七讲/04 PEFT LoRA 源码/代码/tutorials' --record './outputs/code_unit_coverage.json'
& 'D:/anaconda/python.exe' './scripts/verify_annotations.py' --directory 'G:/zzz/Personal ChatGPT 前七讲/04 PEFT LoRA 源码/代码/tutorials' --record './outputs/annotation_verification.json'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './scripts/verify_injection.py' --notebook 'G:/zzz/Personal ChatGPT 前七讲/04 PEFT LoRA 源码/代码/tutorials/04_PEFT_lora.ipynb' --peft-source 'G:/zzz/Personal ChatGPT 前七讲/共享资料/peft-v0.3.0' --record './outputs/injection_verification.json'
```

固定源码入口为[官方提交](https://github.com/huggingface/peft/tree/7120c9a2636f93c8db3fc4ae466e02f338ecead8)。本地checkout已存在，本次没有改变其中源码或Git状态。

## 实际证据

- [覆盖记录](outputs/code_unit_coverage.json)与[注释审计](outputs/annotation_verification.json)：18格、8代码格、6源码块，原单元及输出保持，注释-only、AST及schema通过。
- [真实固定源码记录](outputs/injection_verification.json)：关键调用get_peft_model→add_adapter→_prepare_lora_config→_find_and_replace→两次_replace_module→mark_only_lora_as_trainable确实发生；两层从torch.nn.modules.linear.Linear变为peft.tuners.lora.Linear，旧weight/bias是相同Parameter对象，外层复用原BLOOM。
- A[16,16]、B[48,16]、scale2，总2048个可训练参数，只有lora_参数可训练。零B时全模型logits[1,3,32]与注入前逐元素相等。第一次反向A梯度为0，B梯度范数94.1398，基weight/bias梯度为None。
- 为测试非零增量，**人工设置B为非零随机值，没有进行训练**。目标层输出[2,3,48]与手写双分支误差0；disable还原基输出；merge与未合并前向误差1.49e-7，unmerge在容差内恢复权重和输出。不能将浮点merge/unmerge称为逐bit无损。
- 列表后缀与'.*query_key_value'完整正则均匹配2层；裸'query_key_value'和错误['q_proj']匹配0层，并在真实get_peft_model中产生ValueError。
- 原Markdown7的数据类在补齐真实导入/父类后产生`TypeError: non-default argument 'target_modules' follows default argument`；官方完整定义有field(default=None,...)，不是这个缺字段的节选。注释版保留缺陷。

普通CPU Linear、引用身份与梯度链验证不能证明8bit的state搬运、CUDA量化kernel、原7B显存/性能、任意卷积适配或老师dev环境。原notebook没有训练loop或评估指标，本次也没有训练或评价任务质量。
