# -*- coding: utf-8 -*-
# 第12讲原代码纯文本阅读版；历史输出未加入Python。
# 原远程/4bit/FP16配置保留；CPU教学验证另见复现README。

# %% [markdown]
# ## 单元 0（code）：保留四条安装注释
#
# 每行#!pip均以#开头，不执行安装。取消#后!为IPython shell命令而非普通Python；git+ssh需要SSH访问，主分支安装没有锁版本，-q安静输出、-U/--upgrade升级。bitsandbytes处理低比特CUDA层，trl提供SFTTrainer；本次真实环境另外固定版本，未照原注释升级全局环境。

# %% 原单元 0
# 保留四条安装注释；原语句与历史输出保留，实测另存。
#!pip install -q git+ssh://git@github.com/huggingface/transformers.git
#!pip install -U git+ssh://git@github.com/huggingface/accelerate.git
#!pip install --upgrade bitsandbytes
#!pip install trl

# %% [markdown]
# ## 单元 1（code）：导入数据、LoRA与SFT训练对象
#
# torch为模块；load_dataset模块函数；PEFT的LoraConfig类/get_peft_model函数/int8历史alias；transformers工厂类和TrainingArguments类；SFTTrainer继承Trainer。导入不会下载7B权重或训练。原作者版本未锁，本次旧版参考为TRL0.4.7、PEFT0.4.0、Transformers4.30.0。

# %% 原单元 1
# 导入数据、LoRA与SFT训练对象；原语句与历史输出保留，实测另存。
# 导入Tensor/自动求导模块，导入不启动GPU训练。
import torch
# 数据加载模块函数，返回Dataset/分区对象而非Tensor。
from datasets import load_dataset
# 配置类、真实PEFT包装函数与旧k-bit准备alias，依赖固定版本。
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
# 配置驱动的模型/tokenizer工厂与训练参数类。
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# 旧版SFTTrainer继承Trainer，管理文本与packing，不自动回答mask。
from trl import SFTTrainer

# %% [markdown]
# ## 单元 2（markdown）：数据章节入口
#
# 先读文本字段与模板，再追踪分词、packing与labels，不只记训练类名称。

# %% [markdown] 原单元 2
# ## dataset

# %% [markdown]
# ## 单元 3（code）：作者的本机代理设置
#
# os.environ为当前进程环境；HTTP_PROXY/HTTPS_PROXY写入后影响后续网络请求，7890必须有真实代理服务。不是模型设备参数。本次不执行，不修改用户代理；原格保留。

# %% 原单元 3
# 作者的本机代理设置；原语句与历史输出保留，实测另存。
# 导入当前进程环境操作模块。
import os
# 只改变进程后续HTTP代理；本次验证跳过该格。
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# HTTPS代理需真实本机服务，不是模型设备配置。
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# %% [markdown]
# ## 单元 4（code）：从Hub加载Alpaca训练分区
#
# load_dataset函数第一个字符串为数据集名，split=train选择训练分区，返回HF Dataset，不是Tensor。历史52002行，instruction/input/output/text四列；本次不用当前Hub行数替代历史。实际教学用明确自建的六行同字段数据，不称为Alpaca全量下载。

# %% 原单元 4
# 从Hub加载Alpaca训练分区；原语句与历史输出保留，实测另存。
# Hub数据集标识及train分区；历史52002行，本次未下载全量。
train_dataset = load_dataset("tatsu-lab/alpaca", split="train")

# %% [markdown]
# ## 单元 5（code）：查看Dataset对象
#
# 单独表达式在notebook显示列和行数，不遍历/训练全部样本。历史52002行属于原输出；教学小表有6行，分别记录。

# %% 原单元 5
# 查看Dataset对象；原语句与历史输出保留，实测另存。
# 在notebook显示表结构，未训练或分词。
train_dataset

# %% [markdown]
# ## 单元 6（code）：查看无额外输入的样本
#
# 索引0返回单行dict；input空字符串表示没有额外上下文，instruction不为空。text已含任务说明/Instruction/Response及答案，output仅回答。原注释without input不改变数据。

# %% 原单元 6
# 查看无额外输入的样本；原语句与历史输出保留，实测另存。
# without input
# 按0起始索引取第一行dict，原例input为空而instruction非空。
train_dataset[0]

# %% [markdown]
# ## 单元 7（code）：打印首行完整训练文本
#
# [0]取行，[text]取字符串，print只显示内容；这不是tokenizer调用。完整text含prompt和回答，默认全token监督，不自动只训Response。

# %% 原单元 7
# 打印首行完整训练文本；原语句与历史输出保留，实测另存。
# 取首行完整模板字符串，print仅显示，不计算token IDs。
print(train_dataset[0]['text'])

# %% [markdown]
# ## 单元 8（code）：查看带Input模板的样本
#
# 索引5为第六行，历史样本有附加input；本次六行教学表也提供带input行。数字索引不是类别标签或split划分。

# %% 原单元 8
# 查看带Input模板的样本；原语句与历史输出保留，实测另存。
# with input
# 取第六行dict，原例带附加input。
train_dataset[5]

# %% [markdown]
# ## 单元 9（code）：打印第六行完整文本
#
# 展示Instruction/Input/Response模板；输入条件必须保留，单改dataset_text_field为output会丢prompt。

# %% 原单元 9
# 打印第六行完整文本；原语句与历史输出保留，实测另存。
# 打印含Instruction/Input/Response的完整text；回答仅output字段。
print(train_dataset[5]['text'])

# %% [markdown]
# ## 单元 10（markdown）：分词与模型入口
#
# tokenizer文本→整数ID，不包含7B权重；AutoModel加载配置与参数，两者需核对有效ID范围。

# %% [markdown] 原单元 10
# ## tokenizer & model

# %% [markdown]
# ## 单元 11（code）：Xgen标识、远程tokenizer与pad状态
#
# model_ckpt为Hub标识；AutoTokenizer.from_pretrained类接口依配置选择自定义XgenTokenizer，trust_remote_code允许仓库Python代码。这会执行对应仓库的Python代码，运行前应查看实现并固定revision。pad_token=eos_token修改分词对象状态，复用EOS而不新增ID。历史len50313/EOS50256；固定参考revision的词表应实测，不能假定不变。教学分词对象与原Xgen加载分开记录。

# %% 原单元 11
# Xgen标识、远程tokenizer与pad状态；原语句与历史输出保留，实测另存。
# 保存模型Hub标识字符串，不是已加载对象。
model_ckpt = "Salesforce/xgen-7b-8k-base"
# 按仓库配置选择自定义Tokenizer，允许远程Python；本次不照原格联网执行。
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
# 修改pad状态复用EOS，不新增一个词表项。
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# ## 单元 12（code）：查看分词对象
#
# 显示Tokenizer类、词表与特殊token配置，不计算模型logits。vocab_size与len(tokenizer)在有added tokens时可能不同，resize用的是len。

# %% 原单元 12
# 查看分词对象；原语句与历史输出保留，实测另存。
# 显示分词对象/词表/特殊token配置，无模型forward。
tokenizer

# %% [markdown]
# ## 单元 13（code）：原7B四比特模型加载
#
# from_pretrained读配置/权重；load_in_4bit交给bitsandbytes量化线性层；torch_dtype=float16不意味着所有存储或算子FP16。device_map=auto分配模型模块，不自动DDP或翻倍batch。原uint8为打包容器，非普通8bit矩阵；没有显式NF4/double-quant/compute_dtype，不能称完整QLoRA配置。本次保留此格但不下载7B，不用假Linear4bit代替实际量化。

# %% 原单元 13
# 原7B四比特模型加载；原语句与历史输出保留，实测另存。
# 构造并读取CausalLM基础权重；原7B量化加载本次未执行。
model = AutoModelForCausalLM.from_pretrained(
        # 4bit存储、非量化FP16加载与模块自动分配是三个不同配置；不是DDP。
        model_ckpt, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto"
    )

# %% [markdown]
# ## 单元 14（code）：遍历参数名、dtype和device
#
# named_parameters迭代(name,Parameter)，enumerate给0起始序号，元组解包到name/layer；layer在这里是参数而不是模块。print只读存储信息，没有forward，也不打印requires_grad；必须另审计冻结/可训练集合。原量化设备输出为历史，CPU小模型输出另存。

# %% 原单元 14
# 遍历参数名、dtype和device；原语句与历史输出保留，实测另存。
# 遍历参数名字/Parameter并给序号；layer变量不是一个模块实例。
for i, (name, layer) in enumerate(model.named_parameters()):
    # 打印参数存储dtype/设备，不执行forward，也不显示冻结状态。
    print(i, name, layer.dtype, layer.device)
#     print(i, layer.dtype, layer.device)

# %% [markdown]
# ## 单元 15（code）：查看模型结构
#
# 对象repr显示模块树。历史LlamaForCausalLM类/Linear4bit/51200×4096 embedding对应Xgen名称；repr不执行训练或测推理质量。

# %% 原单元 15
# 查看模型结构；原语句与历史输出保留，实测另存。
# 显示模块树，历史Llama类与Linear4bit不可冒称本次CPU结构。
model

# %% [markdown]
# ## 单元 16（markdown）：Embedding章节入口
#
# 输入ID必须落在embedding行数范围；词表与输出head也需一致。

# %% [markdown] 原单元 16
# ### embedding

# %% [markdown]
# ## 单元 17（markdown）：核对词表与Embedding容量
#
# 原笔记说未必需要resize，正确条件是合法ID小于embedding行数，不要求两者完全相等。历史51200行可容纳50313 ID。是否缩小要结合基础checkpoint与adapter保存/重载；不能以文字笔记替代实际shape检查。

# %% [markdown] 原单元 17
# - 未必是必要的，需要check tokenizer 的 vocab_size 与 model embedding layer 是否一致

# %% [markdown]
# ## 单元 18（code）：再次查看tokenizer
#
# 与12格同对象，不重新加载。用于对比len/token ID和后面的embedding容量；历史词表数不覆盖本次教学值。

# %% 原单元 18
# 再次查看tokenizer；原语句与历史输出保留，实测另存。
# 显示分词对象/词表/特殊token配置，无模型forward。
tokenizer

# %% [markdown]
# ## 单元 19（code）：访问LLaMA输入Embedding模块
#
# model为CausalLM，model.model为内部LlamaModel，embed_tokens为Embedding实例；weight逻辑shape历史[51200,4096]。表达式不进行lookup，更不是print全部权重。包装PEFT前后访问层级可能改变，本格在包装前。

# %% 原单元 19
# 访问LLaMA输入Embedding模块；原语句与历史输出保留，实测另存。
# 在未包装LLaMA中访问输入Embedding，不做lookup或复制权重。
model.model.embed_tokens

# %% [markdown]
# ## 单元 20（code）：按tokenizer长度resize
#
# 模型方法改输入Embedding/未绑输出head与config.vocab_size，返回新的输入Embedding模块。历史51200→50313，少887行，是缩小。pad=eos不增加词。resize在冻结准备之前；adapter-only保存不自动保证以后基础词表同形，需要重载验证。教学以明确小词表容量差演示，不复刻7B尺寸。

# %% 原单元 20
# 按tokenizer长度resize；原语句与历史输出保留，实测另存。
# 方法修改Embedding/输出head/config，返回输入Embedding；历史是缩小887行。
model.resize_token_embeddings(len(tokenizer))

# %% [markdown]
# ## 单元 21（markdown）：PEFT包装入口
#
# 先处理基础权重requires_grad/dtype，再注入adapter；二者职责不同。

# %% [markdown] 原单元 21
# ###  peft model

# %% [markdown]
# ## 单元 22（code）：int8名称的k-bit准备alias
#
# PEFT0.4.0的prepare_model_for_int8_training转调kbit准备，旧名字不代表把4bit改成8bit。先冻结基础参数，部分FP16/BF16提升FP32；真正低比特模型可开启checkpointing与输入梯度路径。输入激活可求梯度≠embedding权重解冻。CPU全精度例子演示参数冻结和dtype提升；四位模型的准备路径见对应加载配置。

# %% 原单元 22
# int8名称的k-bit准备alias；原语句与历史输出保留，实测另存。
# 旧alias转kbit准备：冻结基础参数，按真实加载状态决定其他处理。
model = prepare_model_for_int8_training(model)

# %% [markdown]
# ## 单元 23（code）：LoRA配置与真实包装
#
# r16、alpha32使缩放2；dropout.05作用于adapter输入；bias=none不额外训练基础bias；CAUSAL_LM为任务包装。原没写target_modules，PEFT0.4.0的llama默认q_proj/v_proj；A[r,d_in]、B[d_out,r]。原逻辑每4096投影131072、32层两目标共8388608参数。CPU小LLaMA真实注入应另数adapter参数，不用原计数描述小模型。

# %% 原单元 23
# LoRA配置与真实包装；原语句与历史输出保留，实测另存。
# r16/alpha32缩放2，默认llama q/v目标；dropout作用于adapter输入，基础bias不训练。
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
# 返回真实PeftModel包装并注入可训练LoRA；基础参数保持冻结。
model = get_peft_model(model, peft_config)

# %% [markdown]
# ## 单元 24（code）：全部训练配置
#
# output_dir为本地checkpoint路径；batch4指每进程packed块数；AdamW非8bit优化器；logging每10更新；lr2e-4；fp16为训练混合精度，与4bit存储分开；warmup前10%计划更新，linear后续衰减；epoch1/save_epoch；push_to_hub=False不发布模型。原未指定report_to，环境可能启用W&B；教学显式none。原FP16/CUDA设定与CPU变体分别记录，不更改公开原格。

# %% 原单元 24
# 全部训练配置；原语句与历史输出保留，实测另存。
# 构造训练配置对象，构造时未forward；原GPU/FP16与CPU教学变体分开。
training_args = TrainingArguments(
        # 本地模型产物目录，不是Hub仓库名。
        output_dir="xgen-7b-tuned-alpaca-l1",
        # 每训练进程4个packed块；device_map两卡不自动翻倍。
        per_device_train_batch_size=4,
        # PyTorch AdamW优化器，不是8bit optimizer。
        optim="adamw_torch",
        # 每10个更新步记录训练日志，不是每10个token。
        logging_steps=10,
        # 学习率初值0.0002，作用于可训练参数。
        learning_rate=2e-4,
        # 训练混合精度，独立于基础权重4bit；CPU教学显式改False。
        fp16=True,
        # 计划更新步前10%warmup，再进入线性衰减。
        warmup_ratio=0.1,
        # 指定warmup后的线性学习率调度。
        lr_scheduler_type="linear",
        # 一轮训练计划，packing块数与原行数不简单一一对应。
        num_train_epochs=1,
        # epoch边界保存；PEFT保存可能仅adapter。
        save_strategy="epoch",
        # 不推送训练模型到Hub，与本教材Git提交不同。
        push_to_hub=False,
    )

# %% [markdown]
# ## 单元 25（code）：SFTTrainer构造、packing与train
#
# text字段已经是完整训练字符串；max_seq_length1024是块长度，不等于8192模型容量；packing=True用ConstantLengthDataset拼接EOS再切等长块，可跨样本且丢短尾。TRL0.4.7检查已是PeftModel就不再注入，虽然又传peft_config；应审计包装身份。默认packed路径labels=input_ids、padding collator保留真实EOS标签；非packed LM collator pad=eos会遮真实EOS，两路径不同。model内shift一次，prompt也参与loss，EOS不自动阻隔样本attention。train做真实forward/backward/optimizer/save；历史仅198/13001步，未证明整轮完成。CPU教学训练、标签、梯度、保存重载另核对，非7B能力复现。

# %% 原单元 25
# SFTTrainer构造、packing与train；原语句与历史输出保留，实测另存。
# 构造真实训练器，准备packing与collator，再交父类Trainer处理loop。
trainer = SFTTrainer(
    # 传入已经包装的PeftModel，旧版guard防止重复LoRA。
    model=model,
    # 传HF训练表，text为完整指令文本。
    train_dataset=train_dataset,
    # 取text列；不自动只监督Response。
    dataset_text_field="text",
    # 拼接切块长度1024，不代表模型最大8192上下文。
    max_seq_length=1024,
    # 提供分词对象及EOS/pad配置，避免隐式重新加载。
    tokenizer=tokenizer,
    # 绑定前一格batch/optimizer/epoch等配置。
    args=training_args,
    # 拼接样本与EOS后切等长块，跨样本attention仍存在。
    packing=True,
    # 传入配置；已PeftModel时0.4.7不再注入一层。
    peft_config=peft_config,
)
# 实际进入forward/loss/backward/更新/保存，返回训练统计；历史未完成整轮。
trainer.train()

# %% [markdown]
# ## 单元 26（code）：保留原空代码格
#
# 原source为空；保留编号/元数据/历史输出，注释版仅加中文说明，AST仍为空。没有补写原作者未保存的生成/评测代码。

# %% 原单元 26
# 保留原空代码格；原语句与历史输出保留，实测另存。
