# -*- coding: utf-8 -*-
# 第06讲纯文本代码阅读版，无JSON和历史输出。
# 源码安装指令保留为注释；7B加载/.cuda()/交互输入仍是原代码，勿直接整文件运行。
# 使用notebook逐格核对环境；本次CPU教学替代及验证边界见复现/README.md。

# %% [markdown]
# ## 单元 0（code）：两种源码安装方式
#
# 两条`!pip install -q git+...`分别用HTTPS和SSH从Git安装Transformers；`!`是IPython调用shell，`pip install`安装，`-q`减少输出，`git+`声明源码来源。
# 它们是替代方式，通常选一条，不需要顺序安装两次。SSH依赖已有身份/配置；原命令没固定SHA或tag，不能据此重建老师4.30.0.dev0。
# 本次不执行此格。注释版保留原指令，Python阅读版把它们变成注释，重做验证的固定环境见复现文档。此格会修改环境，不是模型推理代码。
#
# ```ipython
# # https 协议
# !pip install -q git+https://github.com/huggingface/transformers.git
# # ssh 协议
# !pip install -q git+ssh://git@github.com/huggingface/transformers.git
# ```

# %% 原单元 0
# 原两条安装指令是HTTPS/SSH替代方式，未固定版本；本次不执行。
# !是IPython shell语法，阅读版保留为注释；-q只减少安装输出。
# https 协议
# IPython原指令：!pip install -q git+https://github.com/huggingface/transformers.git
# ssh 协议
# IPython原指令：!pip install -q git+ssh://git@github.com/huggingface/transformers.git

# %% [markdown]
# ## 单元 1（code）：导入Transformers模块
#
# import把模块绑定为transformers；不会在此下载LLaMA或加载7B权重。后面版本属性和类导入才引用它；与模型推理是不同步骤。

# %% 原单元 1
# 导入transformers模块供版本属性和后续类使用；不下载模型。
import transformers

# %% [markdown]
# ## 单元 2（code）：查看实际安装版本
#
# `transformers.__version__`是模块属性str，notebook显示最后一个表达式结果。作者历史输出是4.30.0.dev0；本次教学环境为4.44.2。
# 版本名称没有包含开发版Git SHA，不能精确锁老师实现。此行不设置版本，也不更改模型。

# %% 原单元 2
# 访问版本字符串，不设置版本；原历史4.30.0.dev0与本次4.44.2分开记录。
transformers.__version__

# %% [markdown]
# ## 单元 3（markdown）：基础概念标题
#
# 本节串起基模型、Alpaca指令数据、LoRA参数增量与推理。标题本身不创建模型或调用函数。

# %% [markdown] 原单元 3
# ## basics

# %% [markdown]
# ## 单元 4（markdown）：LLaMA到Alpaca LoRA
#
# 基模型提供预训练语言能力；Alpaca提供指令训练语境；LoRA是微调参数的一种方式，本notebook只读取已有adapter。
# “llama→alpaca”是方法背景，不意味着先下载一个完整Stanford Alpaca模型再叠adapter。标准prompt是文字输入格式，不是网络结构；本篇没有训练loss或optimizer。

# %% [markdown] 原单元 4
# - hf（huggingface）中使用 llama
# - llama => alpaca
# - lora on alpaca
# - inference：推理
#     - alpaca 标准 prompt 格式

# %% [markdown]
# ## 单元 5（markdown）：进入模型与分词器加载
#
# 模型把token ID变成logits；tokenizer把字符串变成ID/attention mask。两者分别加载，不是同一个对象。
# 本格是章节标题；真实下载发生在10、14格。

# %% [markdown] 原单元 5
# ## load model/tokenizer

# %% [markdown]
# ## 单元 6（markdown）：外部生成示例链接
#
# 此链接指向Alpaca-LoRA的generate.py，作为背景。该链接跟随main分支变化；本文使用的固定源码版本见复现说明。
# 实际解释和AST核对以本仓库固定提交的28格源码为准；读取链接不是自动授权运行外部脚本。

# %% [markdown] 原单元 6
# - https://github.com/tloen/alpaca-lora/blob/main/generate.py

# %% [markdown]
# ## 单元 7（code）：三个类各自负责什么
#
# LlamaTokenizer是分词器类；LlamaForCausalLM是有词表输出头的LLaMA类；GenerationConfig是生成配置类。from...import把三个类名绑定当前作用域，不在此创建对象。
# 后续from_pretrained是类方法，可能读远程/本地文件；配置构造器只是创建参数对象。导入之后，后续单元会依次创建配置、加载模型和分词器。

# %% 原单元 7
# 分别导入分词器类、带LM输出头的模型类、生成配置类；此行只导入。
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# %% [markdown]
# ## 单元 8（code）：原代理配置保持注释
#
# 原三行均为注释：没import os，也没写HTTP_PROXY或HTTPS_PROXY。127.0.0.1:7890是示例代理地址，不是模型端口。
# 如果自行去掉#，会改变进程网络环境；本次不启用代理，也不执行附件里的口述命令。空操作也纳入覆盖审计。

# %% 原单元 8
# 原代理示例以注释保留；使用时把地址改为本机代理的实际地址。
# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# %% [markdown]
# ## 单元 9（markdown）：进入基模型加载
#
# 先建立基模型，再装adapter。此标题没有变量变化；下一格的model尚未变成PEFT包装。

# %% [markdown] 原单元 9
# ### load model

# %% [markdown]
# ## 单元 10（code）：加载原7B基模型
#
# `LlamaForCausalLM.from_pretrained(...)`输入历史Hub标识str及关键字参数，返回LlamaForCausalLM实例，绑定model；不传训练数据。
# load_in_8bit=True使用旧式bitsandbytes量化加载，可能保留embedding/norm等浮点参数；device_map='auto'交给Accelerate自动分派，不等于两张卡各放一半或数据并行。
# 预期input_ids为int64[B,S]，模型内部embedding为浮点[B,S,H]，LM logits为[B,S,V]。历史7B常见H4096、V32000；本次未实测原权重。
# 本格可能下载大文件并分配内存。本次不执行它，教学补充用本地随机小型LLaMA FP32模型，绝不把它写成原8bit7B复现。

# %% 原单元 10
# 类方法加载历史基模型标识；此格可能下载7B权重，本次未执行。
# load_in_8bit是旧量化加载；device_map=auto自动分派，不保证两卡均分。
# 返回基模型对象model，尚未读Alpaca adapter；历史shape不作本次实测。
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf",
    # 旧式bitsandbytes量化加载路线；不是AMP上下文。
    load_in_8bit=True,
    # 自动安排子模块设备，不等于两卡固定均分。
    device_map="auto",
)

# %% [markdown]
# ## 单元 11（code）：展示基模型结构
#
# 单独写model，notebook显示repr：可以看embedding、层、attention投影和LM head。它不调用forward/generate，不复制权重，不训练。
# 在本次教学上下文里展示的是随机小型LLaMA，不能把其shape当成历史7B的参数。

# %% 原单元 11
# 展示模型repr，不forward、不训练；教学上下文模型明确另设。
model

# %% [markdown]
# ## 单元 12（code）：逐个参数检查device和dtype
#
# named_parameters()是model的方法，返回(name,Parameter)迭代器；enumerate给每项加从0起的序号i。
# para是二元tuple：para[0]为名称str，para[1]为Parameter/tensor。原打印只显示device和dtype，名称打印行仍被注释。
# i是枚举序号，不是层号。device可为cpu/cuda:N；dtype是元素类型，不是shape。遍历与print不搬设备、不改dtype、不更新参数。
# 历史多卡分布是当时机器输出，本次FP32 CPU小模型会显示cpu/torch.float32。读取实际保存输出不等于当前再次测得原分派。

# %% 原单元 12
# named_parameters返回(name,Parameter)；enumerate给每项加序号。
# para[1]是参数tensor；device/dtype是设备/元素类型，不是shape或层号。
# 原名称打印行仍被注释；只读检查和print，不搬设备、不改权重。
# i是枚举序号；para是(name,Parameter)二元tuple。
for i, para in enumerate(model.named_parameters()):
#     print(f'{i}, {para[0]}\t {para[1].device} \t{para[1].dtype}')
    # para[1]取参数，.device/.dtype只读查看；	是输出制表符。
    print(f'{i}, \t {para[1].device} \t{para[1].dtype}')

# %% [markdown]
# ## 单元 13（markdown）：进入tokenizer加载
#
# 模型处理整数ID，tokenizer建立文本与ID的规则。必须与基模型词表兼容；模型结构相同并不能保证两个tokenizer可互换。

# %% [markdown] 原单元 13
# ### load tokenizer

# %% [markdown]
# ## 单元 14（code）：加载分词器文件
#
# `LlamaTokenizer.from_pretrained(model_id)`读取SentencePiece词表、特殊token和tokenizer配置，返回分词器实例tokenizer。
# 同一个Hub名字不是重新下载另一份7B权重。本次不执行远程加载；教学补充用真实SentencePiece训练的小词表，再通过LlamaTokenizer本地加载，不冒充原decapoda词表。

# %% 原单元 14
# 读取分词器文件并返回tokenizer，不再加载一份7B网络。
# 模型与tokenizer词表及特殊ID必须匹配；本次不运行原远程加载。
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

# %% [markdown]
# ## 单元 15（code）：展示tokenizer配置
#
# 最后表达式tokenizer显示词表规模、特殊token等repr；此格没有编码某条指令。
# vocab_size通常是基础词表规模，len(tokenizer)还可能包括新增token；embedding ID范围要匹配。实际小词表在复现记录单独列出。

# %% 原单元 15
# 展示tokenizer配置，不在此编码指令；词表对象不是input_ids。
tokenizer

# %% [markdown]
# ## 单元 16（markdown）：进入LoRA装载
#
# 从已建基模型进入已有adapter推理。这里不是第03讲创建并训练新adapter的步骤。标题不改变model。

# %% [markdown] 原单元 16
# ## with lora

# %% [markdown]
# ## 单元 17（markdown）：Alpaca-LoRA项目背景
#
# 项目链接是方法背景和来源归属，原notebook没有固定其main版本。本篇只核对本notebook与固定官方PEFT源码，不照链接自动运行项目或下载权重。

# %% [markdown] 原单元 17
# - https://github.com/tloen/alpaca-lora

# %% [markdown]
# ## 单元 18（code）：导入PEFT包装类
#
# PeftModel是PEFT包装入口类，import不会注入LoRA。实际读取配置/参数在下一格的类方法from_pretrained。
# 本次验证直接导入官方PEFT v0.3.0固定提交7120c9a，而不是只根据安装包名称猜版本。

# %% 原单元 18
# 导入PEFT包装类；实际注入/读取参数发生在from_pretrained。
from peft import PeftModel

# %% [markdown]
# ## 单元 19（code）：读取已有adapter
#
# `PeftModel.from_pretrained(model, adapter_id)`第一参数是现有基模型对象，第二参数是adapter目录/Hub标识。返回任务对应的PEFT对象，再绑定model。
# 在固定v0.3.0，adapter_name默认'default'，is_trainable默认False：读取配置、根据任务选包装、注入目标层、加载adapter权重并进入推理状态；不是新训练步骤。
# LoRA目标层复用基权重，并增A/B低秩支路。对4096→4096、rank r的投影，A[r,4096]、B[4096,r]，x[B,S,4096]→[B,S,r]→[B,S,4096]。
# 本次不调用原远程adapter标识；补充明确加载自编非零adapter的本地文件，核对保存/重载后前向一致、default键及冻结，不把随机adapter当成Alpaca训练成果。

# %% 原单元 19
# 现有基模型+已训练adapter标识→PEFT对象，再绑定model。
# 读取已保存配置与A/B权重，不是get_peft_model创建新adapter，也不训练。
# 固定v0.3.0默认adapter_name=default、is_trainable=False；本次远程格不执行。
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

# %% [markdown]
# ## 单元 20（code）：展示装载后模型
#
# model现在是PEFT包装，repr可看LoRA层和基模型。不产生回答、不证明已训练、更不是将A/B合并到基础权重。
# 推理需要后面的prompt→tokenizer→generate。

# %% 原单元 20
# 展示PEFT结构；不生成文本，不合并权重，也不证明训练成功。
model

# %% [markdown]
# ## 单元 21（code）：区分保存目标与库默认目标
#
# 从peft导入mapping模块、从peft.utils导入other模块，都是内部调试接口，升级版本时路径或导出可能变化。
# model.config.model_type是架构名str；model.peft_config是adapter名字到配置的字典，['default']取当前配置对象，.target_modules读取保存的目标列表。
# 历史输出q_proj/k_proj/v_proj/o_proj四种目标来自adapter配置；other字典里的llama默认q_proj/v_proj只用于未指定目标时的推断，两者不矛盾。
# 最后一行是属性/字典展示，不是再次注入。mapping虽被导入，这个单元没有调用它。实际小型adapter也显式存四种目标，并核对默认字典仍为两种。

# %% 原单元 21
# 读取已加载adapter目标与库默认字典，二者含义不同。
# default是adapter字典键；q/k/v/o为历史保存目标，默认llama仅q/v。
# 内部模块导入路径随版本变化；本次直接核对固定PEFT0.3.0。
# 导入内部模块供调试；下面实际没有调用mapping函数。
from peft import mapping
# 固定版本的内部工具模块，默认目标字典在其中。
from peft.utils import other

# 读取基模型架构字符串，例如llama；不是任务名。
print('model_type', model.config.model_type)
# 从default配置读已保存目标，优先于默认推断。
print(model.peft_config['default'].target_modules)

#默认的 target module
# 最后表达式展示默认字典，不再次注入层。
other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

# %% [markdown]
# ## 单元 22（markdown）：进入指令示例
#
# 接下来的模板、生成配置和函数只是推理接口；此标题没有产生训练样本或做标准评估。

# %% [markdown] 原单元 22
# ## alpaca examples

# %% [markdown]
# ## 单元 23（markdown）：Stanford Alpaca背景
#
# 链接给出指令训练项目背景；原notebook的模板是本次逐字符核对的实现。没有额外下载Stanford数据集，没有声明已重现原项目评估。

# %% [markdown] 原单元 23
# - https://github.com/tatsu-lab/stanford_alpaca

# %% [markdown]
# ## 单元 24（code）：两个prompt分支与字符串语法
#
# generate_prompt是本notebook定义的普通函数；instruction预期str，input默认None，返回完整str，不改模型状态。
# if input是真值判断：None/''走无Input模板，非空字符串走带Input模板。input参数在函数局部遮蔽内置input，外部单元26仍调用Python内置输入。
# f三引号保留换行，大括号插入instruction/input。英文引导句分别要求完成指令，或结合补充上下文完成指令；Instruction/Input/Response标签与换行都是tokenizer会接收的输入。
# return立即结束该分支；Response冒号后没有示例答案，让模型续写。本次执行原函数两个分支、空字符串分支，逐字核对模板；没有把注释塞进f字符串里改变prompt。

# %% 原单元 24
# 普通函数：instruction为指令str，input可选；返回完整prompt str。
# f三引号内部文本保持原样；注释不能插入字符串内部，否则会改变模型输入。
# None和空字符串走无Input分支；Response结尾留空供续写。
def generate_prompt(instruction, input=None):
    # 真值判断，None/空字符串为False。
    if input:
        # 返回多行f字符串；其内部原文本和换行保持不变。
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    # 没有非空input的模板分支，只含Instruction/Response。
    else:
        # 返回多行f字符串；其内部原文本和换行保持不变。
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

# %% [markdown]
# ## 单元 25（code）：配置、完整数据流及输出风险
#
# GenerationConfig对象保存temperature1.5、top_p0.8、num_beams4；原代码没设置do_sample，所以默认False、num_return_sequences默认1，实际为beam search，不能仅凭top_p说已随机采样。
# inference是普通函数，instruction/input与上一格相同；读取全局model、tokenizer、generation_config，主要打印Response，隐式返回None。函数定义本身不生成回答。
# prompt(str)→tokenizer(...,return_tensors='pt')得到BatchEncoding；input_ids为int64[1,S]，attention_mask也为[1,S]。原代码只取ID再.cuda()，没有传mask；单条无padding输入不等于批量padding场景安全。
# cuda()是tensor方法，默认当前CUDA设备；CPU包在调用函数时于此失败。分派模型的入口设备应核对，auto不是保证。
# generate接收配置和ID，max_new_tokens256只限制新增G，0<G≤256；结构化generation_output.sequences为int64[R,S+G]，原默认R1。beam内部处理4条候选，output_scores保留逐步候选分数，可能增加内存；后文没有使用scores。
# for s按第0维取一条序列，s为[S+G]。decode默认不跳过特殊token且包含输入前缀；split('### Response:')[1]按字符串找第二段，缺标记会IndexError，输入含同标记可能提错位置；strip仅去首尾空白。print输出而无return值。
# 本次原函数定义/配置原样执行，并记录调用的真实CUDA失败；CPU补充仅把.cuda()改成.to('cpu')，其余表达式保留，真实生成、解码和打印另记。随机小模型输出不用于能力评估。

# %% 原单元 25
# 构造配置后定义inference；定义函数本身不执行推理。
# 原do_sample默认False、返回数默认1：4beam不等于4个答案，top_p不自动启用采样。
# 完整shape流与CUDA失败、未传mask、保留scores和split风险见本格说明。
generation_config = GenerationConfig(
    # 仅在相应采样路径调分布；原配置没有do_sample=True。
    temperature=1.5,
    # nucleus sampling
    # 核采样候选阈值；不自动启动采样，原注释意图与实际策略需区分。
    top_p=0.8,
    # 内部保留4条束；num_return_sequences仍默认1。
    num_beams=4,
)

# 读取全局模型/分词器/配置，打印结果，隐式返回None。
def inference(instruction, input=None):
    # 指令/可选上下文→完整模板str。
    prompt = generate_prompt(instruction, input)
#     print(prompt)
    # 调用分词器，return_tensors=pt得到PyTorch BatchEncoding。
    inputs = tokenizer(prompt, return_tensors="pt")
    # 取int64[1,S]并移动到当前CUDA；原代码未把attention_mask传入generate。
    input_ids = inputs["input_ids"].cuda()
    # 调用模型/PEFT的生成方法，包含多步解码和缓存，不是一次forward。
    generation_output = model.generate(
        # 输入token ID，不是embedding或原始文字。
        input_ids=input_ids,
        # 传入上面配置，原默认beam search。
        generation_config=generation_config,
        # 返回结构化生成对象，通过.sequences取序列。
        return_dict_in_generate=True,
        # 保留逐步分数，后文未用；beam分数通常为[4,V]。
        output_scores=True,
        # 只限制新增token上限，输出总长S+G。
        max_new_tokens=256
    )
    # 按返回条数遍历，s为一维int64[S+G]。
    for s in generation_output.sequences:
        # ID→字符串，默认不跳过特殊token，包含prompt前缀。
        output = tokenizer.decode(s)
        # split取第一个Response标记后的片段；标记缺失/重复有风险，strip去首尾空白。
        print("Response:", output.split("### Response:")[1].strip())

# %% [markdown]
# ## 单元 26（code）：交互输入会等待人
#
# Python先调用内置input('Instruction: ')，向终端要字符串，再把返回值作为inference第一个位置参数。不是给函数局部input参数重新赋值。
# 本格会等待键盘，可能随后生成256token，本次不自动执行交互格。核心函数用明确测试字符串调用；静态审计仍保留并核对本格AST。

# %% 原单元 26
# 内置input会等键盘；返回str传给inference，本次不自动执行交互格。
inference(input("Instruction: "))

# %% [markdown]
# ## 单元 27（code）：空代码单元
#
# 原格为空，没有额外算法、输出或变量变化。说明版保留空格，注释版仅添加解释注释，AST仍为空；不能漏掉它来凑“全部代码”覆盖。

# %% 原单元 27
# 原空代码格；只加注释，无可执行表达式，保留原元数据/输出。
