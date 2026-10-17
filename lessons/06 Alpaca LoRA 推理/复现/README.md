# 第06讲实际验证记录

[本讲入口](../README.md) · [中文讲解](../解释.md) · [详细注释代码](../代码/tutorials/代码注释.ipynb)

**状态：部分成功。** 全部28单元说明和18代码格注释完成；原版和注释版各13格原语句在明确的CPU教学上下文执行。原inference真正调用时在.cuda()失败；CPU补充只改设备表达式后，真实小型LLaMA、tokenizer、adapter与beam生成链路通过。未运行原远程7B配置，不评价随机模型回答能力。

## 本次验证哪些原代码

| 范围 | 验证方式 | 边界 |
|---|---|---|
| 原notebook下载 | 固定提交字节与SHA256核对 | 不运行其中安装/模型加载 |
| 全部28格 | 18代码、10Markdown；含安装、代理注释、交互和空格 | 两种派生notebook各57格 |
| 三份notebook | schema、全部原元数据/历史输出、仅增注释、IPython转换AST核对 | 原prompt字符串不改 |
| Python阅读版 | 17个非安装格普通Python AST相等；两条安装指令保留为注释 | 阅读版含远程加载/.cuda()/交互，不能当成CPU一键脚本 |
| 单元1、2、7、8、18、24 | 原语句直接执行：模块/版本/类、代理无操作、原模板函数 | 实际软件版本与作者历史分开 |
| 单元11、12、15、20、21 | 原语句在随机小型模型、真实教学tokenizer及本地adapter上执行 | 结构/参数打印属于小模型，不冒充7B/双卡 |
| 单元25、27 | 原配置/函数定义和空格直接执行 | 原函数定义不自动生成回答 |
| 原inference调用 | 实际在inputs['input_ids'].cuda()报CPU包错误 | model.generate未到达，不用假CUDA规避 |
| CPU补充inference | AST仅将.cuda()改成.to('cpu')，反向还原后与原AST相等；其余配置和语句保留 | 明确改动，另记实际生成结果 |
| 单元0、10、14、19、26 | 原远程安装/加载/交互不执行 | 补充用独立本地文件加载；不假称执行了原远程标识 |

执行列表为`1,2,7,8,18,24,11,12,15,20,21,25,27`，按教学依赖组织，并非原notebook整体原环境顺序重跑。[全部单元解释](../代码单元解释.md)说明每个未执行格的作用、API和风险。

证据分别见[下载记录](outputs/source_download.json)、[注释静态核对](outputs/annotation_verification.json)和[实际调用链记录](outputs/pipeline_verification.json)。以前的[模板/ID教学记录](outputs/教学验证.json)仍保留。

## 真实模型与adapter怎样构造

本地教学基模型是随机LlamaForCausalLM：1层、hidden_size16、intermediate_size32、2个attention heads、2个KV heads、最大位置512，FP32 CPU，attention实现为eager。使用真实SentencePiece训练96词的小词表，再通过LlamaTokenizer的save_pretrained/from_pretrained本地保存加载；不是原decapoda词表。

基模型也真实save_pretrained后本地from_pretrained。自编LoRA配置r2、alpha4、dropout0、bias='none'、任务CAUSAL_LM，明确覆盖q/k/v/o四投影；A为[2,16]，B为[16,2]。给A/B写入可复现的随机非零值，**没有训练这些权重**。adapter单独保存，再真正调用固定PEFT的PeftModel.from_pretrained(base,local_adapter_dir)读取它。

| 核对项目 | 本次结果 |
|---|---|
| 加载目标 | q_proj/k_proj/v_proj/o_proj四层 |
| 库默认llama目标 | q_proj/v_proj，与已保存四目标并存 |
| inference_mode | True，所有模型参数requires_grad=False |
| adapter非零前向影响 | 与裸基模型logits最大差约0.0096809771 |
| logits shape | [1,70,96] |
| 保存/重载前向最大误差 | 0.0，断言容差rtol1e−5/atol1e−6 |
| LoRA层merged状态 | 四层均False，未merge_and_unload |

所有权重、词表和语料都放在临时目录中，验证结束删除，不提交权重。设置HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE；本次无需原模型账号或远程数据。加载时显式统一eager，避免保存前与重载后选择不同attention实现来混淆数值核对；不根据一次严格相等检查失败就声称adapter权重坏了。

## 原配置的生成行为

原GenerationConfig实际得到temperature1.5、top_p0.8、num_beams4、do_sample=False、num_return_sequences1。当前Transformers会警告temperature/top_p用于采样，但本配置未采样；警告保留在记录，原参数不静默修改。

原CUDA函数用真实CPU tensor调用，得到`AssertionError: Torch not compiled with CUDA enabled`。随后另建CPU补充函数，只把设备表达式改为`.to('cpu')`；保留max_new_tokens256、return_dict_in_generate=True、output_scores=True、原decode/split/print。对真实PEFT.generate只加透明结果观察器，内部仍调用真实生成；forward hook只记录shape，没有假生成器或假tokenizer。

| 数据位置 | 真实shape/结果 |
|---|---|
| prompt编码 | input_ids int64[1,70]，同shape attention_mask存在，但原函数未传mask |
| 首次beam forward | input_ids[4,70]，扩展为4条候选 |
| 后续cache forward | input_ids[4,1]，逐步传新ID |
| 返回sequences | int64[1,96]，输入前70个ID逐个一致 |
| 被选中序列新增token | 26，含生成终止标记，≤256 |
| 保存scores | 55个搜索步骤，每步[4,96] |
| 打印与函数返回 | 原decode→split→strip→print执行，函数返回None |

**scores长度不必等于最终选中序列的新增token数。** 此次选中序列先结束，beam搜索还处理其他候选，所以26与55不同。4条候选不等于返回4条回答。输出字符串只保存指纹；随机模型输出不作为笑话质量、准确率或真实Alpaca能力证据。

两个prompt分支及空input与此前逐字模板记录相符。还验证了后处理失败反例：缺Response标记时split取[1]抛IndexError；指令本身含同标记时会选错片段。按prompt token长度切新增ID的改进在主解释中标为补充，不伪装成原源码。

原版与注释版分别独立重建上述上下文，13格结果、adapter目标/冻结/数值、CUDA失败、生成shape/ID指纹/打印指纹一致。对比不只数单元数量。

## 环境与重做命令

实际Windows11、Python3.12.14、torch2.5.1+cpu、transformers4.44.2、sentencepiece0.2.0、tokenizers0.19.1、accelerate0.34.2、IPython8.37.0、nbformat5.10.4。PEFT直接导入官方v0.3.0提交`7120c9a2636f93c8db3fc4ae466e02f338ecead8`的源码；脚本检查HEAD/干净状态、导入路径/版本，并记录四个关键源码SHA256。

旧PEFT加载自己刚保存的adapter时有torch.load FutureWarning，当前Transformers有cache旧tuple兼容提示；本次通过，但不证明可以任意混用版本。作者Python元数据为3.10.9，Transformers历史4.30.0.dev0没有SHA，原模型/adapter没有revision，未精确重建老师环境。

从仓库根目录执行，python使用上述环境。PEFT源码放外部`<PEFT源码目录>`，本仓库不提交完整第三方源码：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "lessons/06 Alpaca LoRA 推理/复现/requirements.txt"
git clone https://github.com/huggingface/peft.git "<PEFT源码目录>"
git -C "<PEFT源码目录>" checkout 7120c9a2636f93c8db3fc4ae466e02f338ecead8
python "lessons/06 Alpaca LoRA 推理/复现/scripts/download_original.py"
python "lessons/06 Alpaca LoRA 推理/复现/scripts/build_annotated.py"
python "lessons/06 Alpaca LoRA 推理/复现/scripts/verify_annotations.py"
python "lessons/06 Alpaca LoRA 推理/复现/scripts/verify_pipeline.py" --peft-source "<PEFT源码目录>"
python "工具/check_documents.py"
```

`<PEFT源码目录>`需换成自己的实际Windows路径；已有匹配的干净clone可复用。首次获取源码/原notebook和安装依赖需要网络，数值验证本身离线。源码安装不是原单元0的浮动main安装，两者明确分开。

原notebook SHA256为`501575b319fcbedcd97b10f80173d3433b2c3d21fe06b3e78138614ab415f1d1`，来源与边界见[来源清单](../来源清单.json)。固定官方实现见[PEFT提交](https://github.com/huggingface/peft/tree/7120c9a2636f93c8db3fc4ae466e02f338ecead8)，生成策略参见[Transformers v4.30文档](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation)。

未运行原7B权重、原LLaMA词表、远程Alpaca adapter、bitsandbytes 8bit、CUDA/多卡auto分派、交互演示、吞吐或质量评测。真实小模型调用链只能支持这里的结构/API/shape/后处理结论。


## 2026-10-17：基础说明、原图、模板与生成边界

本轮原样提取解释.md中两块Python代码执行：原generate_prompt函数加教学调用；独立随机小型LLaMA的CPU生成例子。模板函数AST与原notebook单元24一致，有/无input、空字符串、None及末尾Response留空通过。

新模型仅为1层H8、intermediate16、2头、词表8的随机CPU FP32 LLaMA；不加载Alpaca或adapter，不使用真实LLaMA分词器。输入为明确的教学ID[1,3,4]，num_beams4、do_sample=False、max_new_tokens3；EOS配置None使例子固定生成3步。输入shape[1,3]→返回[1,6]；3步scores各[4,8]，返回前缀ID逐个一致，按长度切出3个新增ID。只核对接口和边界，不保存随机文本作为能力证据。结果、环境和正文代码指纹见[本轮记录](outputs/基础改写验证.json)。

```powershell
& 'G:/zzz/Personal ChatGPT 系列/共享运行环境/torch-cpu/Scripts/python.exe' './lessons/06 Alpaca LoRA 推理/复现/scripts/verify_foundation_examples.py'
& 'C:/Users/hgfhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' './lessons/06 Alpaca LoRA 推理/复现/scripts/extract_paper_images.py' --pdf 'G:/zzz/Personal ChatGPT 前七讲/02 LoRA - Low-Rank Adaptation of Large Language Models/资料/LoRA_2106.09685v2.pdf'
```

使用既有Python3.12.14、torch2.5.1+cpu、Transformers4.44.2环境，设置HF离线变量，不下载模型/数据。原远程7B配置没有运行；上文更完整的SentencePiece/本地随机adapter保存重载及原版/注释版执行属于既有历史证据，本轮未重跑。

Figure1直接从固定LoRA v2 PDF第1页渲染后裁剪，已实际查看，x/h、W/A/B、两路箭头和相加位置完整。正文说明B=0属于初始化，不能把已训练adapter的B当成必为零；也区分论文结构图与实际保存配置。来源、坐标与哈希见[截图记录](outputs/论文截图来源.json)。其他机器重做时从记录中的官方URL取得同版本PDF，替换--pdf路径。原notebook、注释代码及历史输出均未修改。
