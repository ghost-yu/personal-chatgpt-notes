# 第03讲全部代码注释与真实调用链验证

状态：**部分成功**。47/47单元说明、37/37代码单元中文注释已完成，说明版与注释版各95格，原notebook另存。完整第三方notebook和生成版本保留在本地配套 `代码/tutorials` 中；公开仓库提供自编说明与生成/验证脚本。

## 三类证据

1. [覆盖记录](outputs/code_unit_coverage.json)：全部原编号与说明匹配，37格都有中文代码注释；安装与magic也保留。两份文件的哈希及单元数已记录。
2. [注释核对](outputs/annotation_verification.json)：原单元/元数据/历史输出保持，代码只增注释；所有37格经IPython TransformerManager转换后AST一致，schema有效。语法转换不执行pip、magic或远程加载，不能当成全notebook运行成功。
3. [真实CPU调用链](outputs/pipeline_verification.json)：在小型随机BLOOM、新编5条数据和自建30词WordLevel词表上，实际执行26个原代码单元；其余使用独立补充实现连接真实Trainer/collator/forward/generate。没有mock PEFT、Dataset、BLOOM或Trainer。

原代码实际执行编号为10、11、12、14、15、16、17、19、20、21、22、26、27、28、29、30、31、32、33、34、35、36、38、39、41、46。这些单元的**执行上下文是教学模型/数据**，不是原7B权重或Abirate数据。

## 环境、输入与参数差别

注释生成/审计用Windows、Python3.7.6、IPython7.12.0、nbformat5.0.4，不运行原安装单元。真实调用链用独立Python3.12.14、torch2.5.1+cpu、Transformers4.44.2、PEFT0.12.0、Accelerate0.34.2、Datasets2.19.2、Tokenizers0.19.1；与老师的2023 dev环境不同，不声称历史精确复现。

教学BLOOM配置：2层、hidden_size=16、2个头、vocab_size=30，随机初始化、CPU FP32、无8bit/device_map。5条名言与标签均为新编教学数据，词表用真实tokenizers从这些文本训练。没有下载预训练权重、真实BLOOM tokenizer或原数据集。无需API key、GPU；RAM/磁盘容量未测量，没有据此启动大模型。

原15–21单元按原代码执行：基参数冻结，输入embedding输出可求梯度、权重仍冻结，gradient checkpointing开启，lm_head转FP32；LoRA r=16、alpha=32、dropout=.05、bias=none、CAUSAL_LM。真实默认目标为两层融合query_key_value，A权重[16,16]、B[48,16]，scale=2，可训练参数2048。这个小模型比例22.38%，不能套到7B；对应原H=4096、30层的推导见主文。

原42的FP16/200步训练不在CPU原样执行。补充Trainer明确设use_cpu=True、fp16=False、微批2、累积2、max_steps=2、warmup0、lr=2e-3、无checkpoint保存，输出使用自动清理的临时目录；原notebook没有被改写。原44/45 CUDA生成未执行；补充生成使用eval、no_grad、CPU输入、最多3个新token。

## 执行命令

以下PowerShell命令的工作目录为本文件所在复现目录。其他机器需替换配套目录和Python路径。注释依赖与真实Torch验证依赖在两个独立环境中，不要合并旧IPython与新Torch环境。

```powershell
& 'D:/anaconda/python.exe' './scripts/build_annotated.py' --notebook 'G:/zzz/Personal ChatGPT 前七讲/03 LoRA 微调 BLOOM/代码/tutorials/03_lora_practice.ipynb' --output-dir 'G:/zzz/Personal ChatGPT 前七讲/03 LoRA 微调 BLOOM/代码/tutorials' --record './outputs/code_unit_coverage.json'
& 'D:/anaconda/python.exe' './scripts/verify_annotations.py' --directory 'G:/zzz/Personal ChatGPT 前七讲/03 LoRA 微调 BLOOM/代码/tutorials' --record './outputs/annotation_verification.json'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './scripts/verify_pipeline.py' --notebook 'G:/zzz/Personal ChatGPT 前七讲/03 LoRA 微调 BLOOM/代码/tutorials/03_lora_practice.ipynb' --record './outputs/pipeline_verification.json'
```

verify_pipeline的notebook实参用于读取并提取上面列出的原单元，代码显式不执行0、8、25、42、44、45等安装/网络/原训练路径，并设置HF离线环境变量。模型、词表、数据和Trainer的教学差异写在脚本中。

## 实际输出、失败修复与边界

动态padding批次input_ids/attention_mask/labels为[2,14]整数Tensor；7个padding标签被忽略，shift后19个有效目标。真实BLOOM返回loss=3.4229278564，与手动交叉熵差为0；整个非padding文本参与目标，并非仅标签段。

初始反向只有两个lora_B具有非零梯度；Trainer训练2步后4个A/B参数改变，所有基权重与训练前逐元素一致。平均训练loss约3.4140；这些是随机小模型的接口观测，无标签准确率意义。补充generate输出[1,7]，含4个输入token与3个新token，不评价生成质量。

首次安装时未固定Arrow，解析到pyarrow26.0.0，与现有NumPy1.26.4冲突，实际报`ImportError: pyarrow requires NumPy 2.0 or newer`。随后固定pyarrow16.1.0与pandas2.2.2，导入与上述完整验证通过。[环境修复记录](outputs/environment_setup.json)保留失败和修复命令；requirements已使用修复后的组合。

没有执行原watermark环境、bitsandbytes/CUDA加载、7B量化训练、原两条生成示例、真实数据任务指标或论文benchmark。完整大模型/数据revision、历史dev库SHA和泛化能力仍未证明，不能把全部注释覆盖或小模型2步训练升级为原论文成功复现。
