# -*- coding: utf-8 -*-
# 原课程代码阅读版；原GPU/大模型加载保留，CPU变体另见复现脚本。

# %% 原代码单元 0
# datetime是类，now()返回当前本地datetime对象，不参与模型运算；Image稍后展示原图。历史时间2024年4月不是本次运行时间。
# 从模块导入指定对象；后续调用名由此绑定。
from datetime import datetime
# 从模块导入指定对象；后续调用名由此绑定。
from IPython.display import Image
# 返回当前本地时间，验证只检查类型，不要求与历史相同。
datetime.now()

# %% 原代码单元 1
# 本格改进程http/https代理到7890；不属于模型算法。离线CPU验证明确跳过，避免把无服务端口写进环境。
# 导入依赖；不等于下载或构造模型。
import os
# 修改当前进程网络代理环境；离线验证跳过本格。
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# 修改当前进程网络代理环境；离线验证跳过本格。
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# %% 原代码单元 2
# 两行都以#开头，不会执行pip；-U会升级依赖，正式复现使用版本记录而不是盲目运行安装注释。
# !pip install -U wandb
# !pip install -U accelerate

# %% 原代码单元 3
# 导入Transformers适配库与PyTorch张量库；import本身不下载权重，不产生GPU模型。
# 导入依赖；不等于下载或构造模型。
import transformers
# 导入依赖；不等于下载或构造模型。
import torch

# %% 原代码单元 5
# Image读取本地PNG，width400只控制显示宽度。相对路径../../imgs从代码/tutorials/llama解析到代码/imgs，不是从仓库根解析。
# 返回显示对象，读取相对路径PNG，宽度仅控制展示。
Image('../../imgs/meta-llama-model.png', width=400)

# %% 原代码单元 6
# blobfile是读取tiktoken词表依赖；这是注释，不安装。验证在单独runtime-libs目录准备tiktoken0.7.0/blobfile2.1.1。
# !pip install blobfile

# %% 原代码单元 8
# 8.03e9个参数乘2bytes再除1024³，实际单位GiB，字符串写GB不准确；不含KV缓存、激活、加载临时副本或训练优化器。
# 参数数×每参数2bytes÷1024³；输出标签应理解为GiB。
'8.03B, {:.2f}GB'.format(8.03 * 1_000_000_000 * 2 / (1024*1024*1024))

# %% 原代码单元 14
# AutoModelForCausalLM是自动选择因果语言模型实现的工厂；from_pretrained还会解析配置及加载checkpoint，和仅初始化配置不同。
# 从模块导入指定对象；后续调用名由此绑定。
from transformers import AutoModelForCausalLM

# %% 原代码单元 15
# 原格下载2-7B并自动放置FP16权重。CPU教学明确替换为相同完整配置在meta设备构造，只验证真实模块shape/参数量，不伪称加载预训练权重。
# 保存Hub仓库ID字符串；赋值本身不下载。
model_id = "meta-llama/Llama-2-7b-hf"
# 原逻辑加载7B真实权重，CPU验证明确改为完整meta配置。
llama2_7b = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# %% 原代码单元 16
# 交互环境显示模型repr；只展示模块树，不执行forward。Q/K/V/O均4096→4096，MLP宽11008，词表32000。
llama2_7b

# %% 原代码单元 17
# 计算MLP相对隐藏宽度11008/4096=2.6875；SwiGLU三条线性层，不是普通两层4D ReLU。
11008/4096

# %% 原代码单元 18
# named_parameters返回名字/Parameter迭代器；numel返回元素数而不是字节。逐个求和6738415616，meta张量也能精确计数。
# 计数器清零，避免将上个模型计数累加。
total_params = 0
# 逐个解包参数名和Parameter对象，非逐层forward。
for name, para in llama2_7b.named_parameters():
    # numel只读取shape算元素数，无需物理权重数据。
    total_params += para.numel()
total_params

# %% 原代码单元 19
# 原格下载3-8B FP16权重。教学同样只替换加载语句为完整配置的meta模型；词表128256/KV8/MLP14336/rope500000。
# 保存Hub仓库ID字符串；赋值本身不下载。
model_id = "meta-llama/Meta-Llama-3-8B"
# 原逻辑加载8B真实权重，CPU验证明确改为完整meta配置。
llama3_8b = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# %% 原代码单元 20
# 交互显示3-8B模块：Q4096→4096、K/V4096→1024、O4096→4096；GQA缩窄KV投影但保留32个Q头。
llama3_8b

# %% 原代码单元 21
# 总计8030261248个参数，包括未共享的输入embedding与输出lm_head；两个独立矩阵各128256×4096。
# 计数器清零，避免将上个模型计数累加。
total_params = 0
# 逐个解包参数名和Parameter对象，非逐层forward。
for name, para in llama3_8b.named_parameters():
    # numel只读取shape算元素数，无需物理权重数据。
    total_params += para.numel()
total_params

# %% 原代码单元 22
# 精确参数量乘2bytes=14.957527160644531GiB权重；BF16和FP16都是2bytes，不代表推理总显存或训练显存。
# FP16/BF16两字节权重存储估计，不含缓存/激活。
8030261248 * 2 / (1024*1024*1024)

# %% 原代码单元 25
# SentencePieceProcessor从原499723bytes模型读取32000词表；vocab_size是方法。ID语义只属于该词表，不能直接把2的ID喂给3。
# llama2
# 从模块导入指定对象；后续调用名由此绑定。
from sentencepiece import SentencePieceProcessor
# 加载真实SentencePiece文件，文件格式不同于tiktoken。
sp_model = SentencePieceProcessor(model_file='./llama2_tokenizer.model')
# 调用方法得到词条数32000。
sp_model.vocab_size()

# %% 原代码单元 26
# 从同目录原tokenizer.py导入Meta Tokenizer类，以原tiktoken rank文件构造实例。构造函数添加256特殊token；本次真实运行，不替换成伪分词器。
# llama3
# 从模块导入指定对象；后续调用名由此绑定。
from tokenizer import Tokenizer
# 读取原tiktoken词表并附加256特殊ID。
llama3_tokenizer = Tokenizer('./llama3_tokenizer.model')

# %% 原代码单元 27
# n_words是整数属性，不是函数；真实128256=128000普通mergeable ranks+256特殊token。
# 读取属性128256，已包括特殊token。
llama3_tokenizer.n_words

# %% 原代码单元 28
# 仅注释，不会展示字典；新增说明记录BOS128000/EOS128001/头128006与128007/EOT128009。pad_id=-1为哨兵，不可直接喂embedding。
# llama3_tokenizer.special_tokens

# %% 原代码单元 34
# text-generation为任务；model为基础模型ID；model_kwargs只传给模型加载，BF16是权重dtype；device_map auto需相应加载支持。教学用真实随机CPU小Llama与原词表转换的Fast tokenizer，并显式限制4个新增token，不比较语言质量。
# 保存Hub仓库ID字符串；赋值本身不下载。
model_id = "meta-llama/Meta-Llama-3-8B"

# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# model-00003-of-00004.safetensors
# model-00004-of-00004.safetensors
# 构造生成流水线，返回可调用TextGenerationPipeline对象。
pipeline = transformers.pipeline("text-generation",
                                 # 原为Hub模型ID；教学变体显式传真实CPU随机小模型。
                                 model=model_id,
                                 # 该字典在原调用用于模型加载，非采样参数。
                                 model_kwargs={"torch_dtype": torch.bfloat16},
                                 # 让加载器安排设备，auto不保证一定使用GPU。
                                 device_map="auto")
# 调用流水线得到含generated_text的列表，默认包括输入前缀。
pipeline("Hey how are you doing today?")

# %% 原代码单元 36
# messages是list[dict]，apply_chat_template返回带角色头的str；add_generation_prompt补assistant开头。terminators含EOS和EOT，任一可结束；max_new_tokens是新增预算，temperature/top_p控制采样。输出默认含prompt，用字符len(prompt)切掉前缀；本次模型随机且预算明确改4，不冒称海盗回答复现。

# 保存Hub仓库ID字符串；赋值本身不下载。
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 构造生成流水线，返回可调用TextGenerationPipeline对象。
pipeline = transformers.pipeline(
    # 任务名决定分词→generate→decode这条路径。
    "text-generation",
    # 原为Hub模型ID；教学变体显式传真实CPU随机小模型。
    model=model_id,
    # 该字典在原调用用于模型加载，非采样参数。
    model_kwargs={"torch_dtype": torch.bfloat16},
    # 原指定cuda；CPU教学替换构造，不谎称GPU运行。
    device="cuda",
)

# 按顺序保存角色与文本；尚未变成ID。
messages = [
    # role控制模板角色头，content为消息正文。
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    # role控制模板角色头，content为消息正文。
    {"role": "user", "content": "Who are you?"},
]

# 套用Jinja对话模板返回str，再由流水线分词。
prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        # 返回文本而非ID列表，便于检查特殊标记与头部。
        tokenize=False,
        # 末尾追加assistant开头，让模型接着生成正文。
        add_generation_prompt=True
)

# 列表里两种结束ID是或关系，不要求连续生成两者。
terminators = [
    # 读取结束全文ID128001。
    pipeline.tokenizer.eos_token_id,
    # 把EOT标记映射成128009，结束一个对话轮次。
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# model-00001-of-00004.safetensors
# model-00002-of-00004.safetensors
# model-00003-of-00004.safetensors
# model-00004-of-00004.safetensors
# 生成结果列表；随机教学变体只生成至多4个新token。
outputs = pipeline(
    prompt,
    # 原预算256个新增token，不是输入加输出总长。
    max_new_tokens=256,
    # 覆盖结束条件为EOS/EOT两个ID。
    eos_token_id=terminators,
    # True按分布采样，False才是此示例常见greedy路径。
    do_sample=True,
    # 正温度缩放logits，0.6使分布相对更集中。
    temperature=0.6,
    # 保留累计概率达到阈值的候选后再归一化采样。
    top_p=0.9,
)
# 默认输出含prompt，用字符长度切片仅保留续写字符串。
print(outputs[0]["generated_text"][len(prompt):])

# %% 原代码单元 37
# 打印完整序列化提示；这不是生成回答。逐消息以EOT结尾而不是EOS，最后assistant头后有两个换行，BOS仅出现一次。
# 显示格式化输入，不是模型新生成文本。
print(prompt)
