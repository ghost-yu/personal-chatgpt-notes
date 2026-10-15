# 第24讲实际验证

状态：**部分成功**。原 DummyModel 与原 SVD basics 全部代码真实执行；原主讲在显式离线 CPU 变体中执行。未下载或运行 TinyLlama1.1B 预训练模型、三种原任务 adapter 和4bit CUDA。

## 环境和来源

Windows，Python3.12.14，Torch2.5.1+cpu，Transformers4.44.2，PEFT0.12.0，SentencePiece0.2.0。完整版本和实际CUDA可用性见 [报告](outputs/delivery_verification.json)。工作站约16GB RAM；本次没有分配原32005²完整Vh，不报告原GPU显存和速度。

课程固定7fbc29824171fd07dc2c30fe922e61ebc018ad1d。三个原件的字节摘要见 [来源清单](../来源清单.json)。本机PEFT发行文件逐项核对安装包RECORD并固定SHA，讲师版本未知。在线官方tag补充请求遇到TLS EOF，未建立远端字节一致证据。

本地来源目录：G:/zzz/Personal ChatGPT 系列/24 PEFT LoRA merge 与 SVD/资料，保存四个未修改的PEFT源文件、来源字幕。公开源码读本保留Apache许可头并明确摘录范围；复现使用完整安装包，不执行摘录文件替代库。

## 执行与审计范围

- 三原件、六派生件共九个notebook schema，全部98原单元/历史输出/元数据/原字节核对。
- 全部代码格经IPython转换后AST和非注释token不变；三Python阅读版AST对应；核心读本方法与完整源方法AST一致。
- 原版/注释版各执行 inject 的全部14个代码格、SVD basics的全部17个代码格；实际pinfo2输出含真实forward源码，三处Image仅验证HTML引用。
- merge格1跳过代理/GPU环境设置；10/12/16/18/51/54/57是明确CPU变体，报告逐项记录字面替换。其余代码包括仅注释和空单元原样执行。
- 主Markdown5的Linear权重/RSLoRA块是上下文片段，静态说明而非独立运行完整类；Markdown8的regex在真实PEFT小模型上下文执行，验证会得到错误类别的名称。

没有更改公开原件中的cuda、4bit、256/64、Hub名称或full_matrices=True。运行用的变体只存在验证器及报告中。

## 实际教学模型

SentencePiece在显式自建普通文本上训练128词BPE，真实LlamaTokenizer加入原文五种特殊字符串后133词；自建ChatML模板不是原Hub资产。基础Llama为1层、hidden16、intermediate32、2头、FP32，三个非零adapter均r8/alpha16/dropout.1。

norobots/adcopy有九个目标，sql仅七个投影，与原历史目标集合一致。所有adapter共享同一随机基础与新增embedding行；路径、词表、seed均在验证器构造。fixture保存在系统临时目录，运行后自动释放，不作为预训练checkpoint公开。

真实generate保留原采样/重复惩罚/EOS逻辑，但新增预算只设4。随机输出不用于语言任务质量评估。

## 关键判据

1. 注入返回同一DummyModel；base_layer和未注入两层引用不变；基础冻结、128000可训练参数；B0时输出不变，额外一次SGD更新B且保留基础状态。
2. 原100×100低秩与满秩两例运行，重建误差符合奇异值尾部范数；独立diag(5,2,1)保留两项误差1，并核对矩形full/reduced形状。
3. 所有有效目标层核对真实旧库SVD乘积、尾部误差与理想增量。PEFT0.12.0的SVD路径重复scaling；原手写代码也重复，因此二者能吻合，但相同源s2时分解的是理想矩阵的两倍。
4. density .2/.9不改变纯SVD结果；cat精确组合理想增量；linear存在交叉项且负系数math.sqrt报错；rank16恢复的是旧库分解矩阵。
5. 原创建merge后仍激活norobots；显式切换影响logits；merge_and_unload移除LoRA层且logits在FP32容差内相同。新比较在clone上执行，原来源因子不变。
6. 两版逐格trace和所有模型状态精确一致。不能仅用单元计数或一个shape断言替代这些语义判据。

## 重做命令

从任意目录执行；路径需按本机环境修改：

~~~powershell
$env:LESSON24_RUNTIME_LIBS = 'G:/zzz/Personal ChatGPT 系列/24 PEFT LoRA merge 与 SVD/资料/runtime-libs'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/24 PEFT LoRA merge 与 SVD/复现/scripts/build_delivery.py' --upstream 'G:/zzz/Llama 入门教程/代码/personal_chatgpt' --peft-root 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Lib/site-packages/peft' --sources 'G:/zzz/Personal ChatGPT 系列/24 PEFT LoRA merge 与 SVD/资料'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/lessons/24 PEFT LoRA merge 与 SVD/复现/scripts/verify_delivery.py' --sources 'G:/zzz/Personal ChatGPT 系列/24 PEFT LoRA merge 与 SVD/资料'
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' 'G:/zzz/personal-chatgpt-notes/工具/check_documents.py'
~~~

builder只生成原件与派生阅读内容；verifier单独实际运行并写新报告。原字幕当前来自E盘指定附件，builder会核对摘要。报告还固定两个脚本SHA，之后若改脚本必须重新运行。

## 失败与限制

初次验证中，第二次IPython查询没有进入本次捕获列表：set_hook累积回调使第一轮的展示hook继续处理。验证器改为替换当前show_in_pager展示回调，查询仍由真实IPython执行，原源码不改。旧notebook缺少新式cell id，nbformat给兼容警告；原元数据保留，未为消除警告改写原件。

AutoTokenizer默认转换为fast tokenizer时首次缺protobuf；复用了已有环境的protobuf3.20.3纯Python发行文件，放在本讲本地runtime-libs，仅通过本进程路径加载，未修改共享环境。若自行安装本讲requirements，可不设置LESSON24_RUNTIME_LIBS。没有把分词器改成假实现或绕开原AutoTokenizer调用。

没有原训练后的基础权重/adapter、完整原答案重放、任务指标、TIES/DARE、Conv2d、GPU测速、多卡或LoRA论文基准。原历史输出不作为本次运行证据；随机小模型成功生成不等于学会原任务。
