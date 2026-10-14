# -*- coding: utf-8 -*-
# 第14讲Python阅读版，保留原代理/远程7B/FP16设置。
# CPU验证及外部训练示例运行改动见复现README。

# %% [markdown]
# ## 单元 0（markdown）：基础概念入口
#
# 【必须会】区分base/chat、权重格式与推理/训练两条路径；下方生成来自chat权重，不是后面Guanaco微调后的效果。

# %% [markdown] 原单元 0
# ## basics

# %% [markdown]
# ## 单元 1（markdown）：发布入口与上下文
#
# 【必须会】7B/13B/70B规模×原始/HF格式×base/chat，共12入口；hf是格式/生态，不是能力等级。4K指输入与生成的总token预算。官方blog/模型页为来源链接，不自动执行或获取权限；7B/13B不能混称70B的GQA设置。

# %% [markdown] 原单元 1
# - 参考
#     - https://huggingface.co/blog/llama2
# - 4k tokens of context
# - 几个版本：3\*2\*2 == 12
#     - https://huggingface.co/meta-llama
#     - 7B，13B，70B
#     - 是否是 hf
#     - 是否是 chat
#         - meta-llama/Llama-2-7b
#         - meta-llama/Llama-2-7b-hf
#         - meta-llama/Llama-2-7b-chat
#         - meta-llama/Llama-2-7b-chat-hf

# %% [markdown]
# ## 单元 2（markdown）：文本生成章节
#
# 【必须会】字符串→分词ID→chat模型generate→decode。pipeline默认返回完整prompt加生成后续，普通字符串不自动变成INST模板。

# %% [markdown] 原单元 2
# ## text-generation

# %% [markdown]
# ## 单元 3（code）：导入分词、模型、量化配置与Torch
#
# 【必须会】AutoTokenizer/AutoModelForCausalLM为工厂类，transformers和torch为模块。BitsAndBytesConfig导入不自动量化；本notebook推理没有用它加载4bit。导入不会下载7B。

# %% 原单元 3
# 导入分词、模型、量化配置与Torch；原语句保留，实际CPU范围另记。
# 导入按配置选分词/因果LM实现的工厂类。
from transformers import AutoTokenizer, AutoModelForCausalLM
# 导入模块，后续pipeline是模块函数。
import transformers
# 量化配置类，本notebook推理加载没有使用它。
from transformers import BitsAndBytesConfig
# 导入Tensor/自动求导模块，不启动训练。
import torch

# %% [markdown]
# ## 单元 4（code）：保存chat模型Hub标识
#
# 【必须会】字符串指向7b-chat-hf；训练Markdown15使用7b-hf基础模型，是两个checkpoint。本次原格保留执行，CPU上下文另把model_name指向同一随机小模型的本地目录。

# %% 原单元 4
# 保存chat模型Hub标识；原语句保留，实际CPU范围另记。
# chat推理权重标识字符串，与后续base训练模型不同。
model_name = "meta-llama/Llama-2-7b-chat-hf"

# %% [markdown]
# ## 单元 5（code）：作者的本地代理
#
# 【建议会】HTTP_PROXY/HTTPS_PROXY写进程网络配置，7890应有真实代理；不是GPU配置。本次跳过，不修改用户代理环境。

# %% 原单元 5
# 作者的本地代理；原语句保留，实际CPU范围另记。
# 导入进程环境操作模块。
import os
# 当前进程HTTP代理，需本地真实服务；本次跳过该格。
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# HTTPS代理设置，不是GPU或模型参数。
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# %% [markdown]
# ## 单元 6（code）：tokenizer与生成pipeline加载
#
# 【必须会】from_pretrained返回分词对象；pipeline工厂返回TextGenerationPipeline，内部还会自己加载tokenizer，原代码未传外部tokenizer。torch_dtype FP16是浮点加载，device_map auto按容量分配模块，不是4bit或DDP。本次CPU变体只把FP16→FP32、auto→None，并在上下文提供本地随机小模型路径；真实对象与原7B分开。

# %% 原单元 6
# tokenizer与生成pipeline加载；原语句保留，实际CPU范围另记。
# 类方法加载外部分词对象，不加载7B模型权重。
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 工厂函数构造TextGenerationPipeline，内部持有model/tokenizer。
pipeline = transformers.pipeline(
    # 任务名选择生成pipeline，不自动构造聊天模板。
    "text-generation",
    # 按标识加载模型；原没传外部tokenizer，所以内部另加载分词对象。
    model=model_name,
    # 原FP16浮点权重加载，不是4bit量化；CPU变体另改FP32。
    torch_dtype=torch.float16,
    # 按容量分配模块，不自动DDP；CPU变体另改None。
    device_map="auto"
)

# %% [markdown]
# ## 单元 7（code）：显示外部分词对象
#
# 【建议会】表达式显示词表和special tokens，不执行模型。外部tokenizer与pipeline.tokenizer是不同实例，改外部pad不自动同步内部。

# %% 原单元 7
# 显示外部分词对象；原语句保留，实际CPU范围另记。
# 显示外部分词对象，可能与pipeline内部实例状态不同。
tokenizer

# %% [markdown]
# ## 单元 8（code）：显示pipeline模型
#
# 【必须会】访问真实model属性，repr展示模块树而不forward。原历史LLaMA hidden4096、32层、vocab32000；小模型结构另记录。

# %% 原单元 8
# 显示pipeline模型；原语句保留，实际CPU范围另记。
# 访问生成pipeline持有的模型，不执行forward。
pipeline.model

# %% [markdown]
# ## 单元 9（code）：统计参数与requires_grad
#
# 【必须会】named_parameters给(name,Parameter)，numel数元素；if requires_grad只计允许求导参数；+=累加。1.强制浮点比例。原6,738,415,616和比值1不等于正在训练；pipeline推理没有optimizer更新。4bit打包权重的存储numel另有语义，不能把7B误称3.5B。本次真实小模型统计，另用meta结构核对7B逻辑计数。

# %% 原单元 9
# 统计参数与requires_grad；原语句保留，实际CPU范围另记。
# 初始化允许求导参数的元素计数。
trainable_params = 0
# 初始化全部参数元素计数。
all_param = 0
# 遍历参数名/Parameter；不会更新权重。
for name, param in pipeline.model.named_parameters():
#     print(name, param.requires_grad, param.numel())
    # numel计元素并累加，量化打包容器需另辨认逻辑shape。
    all_param += param.numel()
    # 判断是否允许求导，不表示已有loss或optimizer。
    if param.requires_grad:
        # 只将允许求导参数加入分子。
        trainable_params += param.numel()
# 打印元素计数及float比例；推理中比值1也不执行训练。
print(trainable_params, all_param, trainable_params*1./all_param)

# %% [markdown]
# ## 单元 10（code）：英文随机生成三候选
#
# 【必须会】调用pipeline实例的__call__。do_sample随机采样；top_k每步候选10；num_return_sequences3是同prompt三个候选，不是三轮聊天。eos为停止ID；max_length200包括输入而不是新增200。字符串普通推荐问句，没有INST。CPU小模型整格保留这些生成参数，随机输出不当聊天能力。

# %% 原单元 10
# 英文随机生成三候选；原语句保留，实际CPU范围另记。
# 调用真实pipeline对象，返回候选list[dict]。
sequences = pipeline(
    # 普通英文prompt，没有[INST]模板；输入长度计入200token预算。
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    # 每一步按概率随机抽样，需固定seed才能受控比较。
    do_sample=True,
    # 只从该步分数最高的10个token中采样，不是输出10条回答。
    top_k=10,
    # 同一prompt三份候选，不是三轮历史对话。
    num_return_sequences=3,
    # 外部分词对象提供EOS停止ID，原Llama2为2。
    eos_token_id=tokenizer.eos_token_id,
    # 输入加输出总长上限200，而不是新增200。
    max_length=200,
)

# %% [markdown]
# ## 单元 11（code）：打印英文候选
#
# 【必须会】sequences为list[dict]，for逐个取seq，[generated_text]取字符串，+两个换行排版。默认full_text包含prompt；print不运行新模型。原历史样例保留，CPU随机输出另记录。

# %% 原单元 11
# 打印英文候选；原语句保留，实际CPU范围另记。
# 逐个取候选字典用于显示。
for seq in sequences:
    # 取默认完整文本并加换行排版，prompt重复不必是模型再次生成。
    print(seq['generated_text'] + '\n\n')

# %% [markdown]
# ## 单元 12（code）：中文随机生成和打印
#
# 【必须会】Unicode中文prompt同样先分词，字符数不等token数；三个候选、top_k10、总长200与英文格相同。异常续写不能仅凭样例归因于中文词表损坏或微调失败。原格真实CPU执行，结果只核对流程及注释等价。

# %% 原单元 12
# 中文随机生成和打印；原语句保留，实际CPU范围另记。
# 调用真实pipeline对象，返回候选list[dict]。
sequences = pipeline(
    # 中文字符串先分词，字符长度不同于token长度。
    '我比较喜欢《肖申克的救赎》、《星际穿越》这样的电影，你有其他电影推荐吗？\n',
    # 每一步按概率随机抽样，需固定seed才能受控比较。
    do_sample=True,
    # 只从该步分数最高的10个token中采样，不是输出10条回答。
    top_k=10,
    # 同一prompt三份候选，不是三轮历史对话。
    num_return_sequences=3,
    # 外部分词对象提供EOS停止ID，原Llama2为2。
    eos_token_id=tokenizer.eos_token_id,
    # 输入加输出总长上限200，而不是新增200。
    max_length=200,
)
# 逐个取候选字典用于显示。
for seq in sequences:
    # 显示中文候选；CPU随机模型结果不是语言质量评价。
    print(seq['generated_text'])

# %% [markdown]
# ## 单元 13（markdown）：微调章节入口
#
# 【必须会】切换到base权重与Guanaco监督数据；不能把前面的chat生成当作后续微调评估。

# %% [markdown] 原单元 13
# ## fine-tune

# %% [markdown]
# ## 单元 14（markdown）：TRL训练器入口
#
# 【建议会】SFTTrainer管理数据准备/训练，不自动只监督Assistant；需读脚本和collator默认值。

# %% [markdown] 原单元 14
# - trl
#     - SFTTrainer

# %% [markdown]
# ## 单元 15（markdown）：安装、clone与完整微调命令
#
# 【必须会】三条shell逻辑命令：pip install trl安装库；git clone取示例；python运行示例。原未锁tag，Bash反斜杠续行不能直接照搬PowerShell；注释不插入续行链中，避免破坏语义。原flags模型base7b、Guanaco数据、4bit、PEFT、micro16、累积2，有效单卡batch32。固定参考TRL0.5示例默认seq512、epoch3、r64/alpha16缩放.25、packingFalse；作者改稿未找到。原4090约5小时与字幕3090矛盾，是估计非实测。本次解析原命令并在本地明确CPU参数下执行真实固定示例，不执行原pip/clone或7B量化。9846*3/(16*2)实际923.0625，按每epoch616micro/308更新才得到924。

# %% [markdown] 原单元15 shell命令
# ```
# # 安装TRL库，原命令未锁版本；本次使用独立固定环境。
# pip install trl
# # 取示例仓库，原未固定tag；不要假定与已安装库同版本。
# git clone https://github.com/lvwerra/trl
# # Python解释器 + 示例路径；下方反斜杠是Bash续行，PowerShell语法不同。
# # --model_name：base 7b-hf，不是前面的chat checkpoint。
# # --dataset_name：原始Guanaco多轮text字段。
# # --load_in_4bit：量化基础权重；--use_peft：训练LoRA。
# # --batch_size16与--gradient_accumulation_steps2：单卡有效batch32。
# # 没写seq_length/epochs/r/alpha：参考v0.5.0默认512/3/64/16。
# python trl/examples/scripts/sft_trainer.py \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --dataset_name timdettmers/openassistant-guanaco \
#     --load_in_4bit \
#     --use_peft \
#     --batch_size 16 \
#     --gradient_accumulation_steps 2
# ```
#
# - 对于 4090，大概是需要 5个小时
#     - 3 个 epochs
#     - 16*2 的 batch size
#     - (9846 * 3) / (16*2) == 924 个 optimizer steps

# %% [markdown]
# ## 单元 16（markdown）：Guanaco数据章节入口
#
# 【必须会】原timdettmers/openassistant-guanaco，不是别的同名1k/INST重排子集；text是多轮对话字符串。

# %% [markdown] 原单元 16
# ### guanaco dataset

# %% [markdown]
# ## 单元 17（code）：导入数据加载函数
#
# 【建议会】模块函数load_dataset，导入不下载数据。本次原格完整执行，加载用固定五行本地样本变体另记录。

# %% 原单元 17
# 导入数据加载函数；原语句保留，实际CPU范围另记。
# 导入HF数据加载模块函数，不在import时下载。
from datasets import load_dataset

# %% [markdown]
# ## 单元 18（code）：从Hub取训练分区
#
# 【必须会】数据集标识与split=train，返回Dataset。原历史9846行；本次不重新取全量，CPU变体改为json/local文件读取已有固定revision五行。样本文件来源和hash另核对，不把五行当全体。

# %% 原单元 18
# 从Hub取训练分区；原语句保留，实际CPU范围另记。
# 原Hub训练分区，历史9846行；本次五行本地变体另记。
dataset = load_dataset('timdettmers/openassistant-guanaco', split='train')

# %% [markdown]
# ## 单元 19（code）：查看数据集
#
# 【建议会】独立表达式显示列和行数，不自动训练。原9846与教学5行是不同范围。

# %% 原单元 19
# 查看数据集；原语句保留，实际CPU范围另记。
# 显示Dataset结构，不训练/分词所有样本。
dataset

# %% [markdown]
# ## 单元 20（code）：打印首行完整text
#
# 【必须会】索引0取字典，text取多轮字符串。原首行Human→Assistant→Human，结尾请求没有回答；不能认为每一行都是一问一答。

# %% 原单元 20
# 打印首行完整text；原语句保留，实际CPU范围另记。
# 取首行多轮text并打印，末尾Human不一定已有回答。
print(dataset[0]['text'])

# %% [markdown]
# ## 单元 21（code）：按###拆字符串
#
# 【必须会】str.split返回list[str]，每项没有###，首项可能空；for/print只是显示。内容内出现###也会被误拆，不是可靠messages解析器或loss mask。真实分词后的监督边界要另核对。

# %% 原单元 21
# 按###拆字符串；原语句保留，实际CPU范围另记。
# 字符串分隔后遍历，非tokenizer/messages解析或监督mask。
for part in dataset[0]['text'].split('###'):
    # 显示分隔片段，可能包含空首项。
    print(part)

# %% [markdown]
# ## 单元 22（code）：粗算优化步数
#
# 【必须会】前两行#只是解释；/产生float923.0625，不自动向上取整。需先ceil(9846/16)=616micro/epoch，再累积2得308更新×3=924；最后有效窗口只有22样本，非32。本次原表达式保留执行；未启动原924更新。

# %% 原单元 22
# 粗算优化步数；原语句保留，实际CPU范围另记。
# optimization steps
# len(dataset) * epochs / (batch_size * gradient_accumulation_steps * n_gpus)
# 普通除法得923.0625；实际924更新需按每epoch批次取整推导。
9846*3/(16*2*1)
