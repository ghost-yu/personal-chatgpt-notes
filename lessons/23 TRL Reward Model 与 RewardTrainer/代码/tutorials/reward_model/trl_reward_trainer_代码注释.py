# -*- coding: utf-8 -*-
# 原notebook全部29代码格的中文注释阅读版；运行前阅读复现范围，不自动执行网络/GPU/1000步训练。

# %% 原代码单元 0
# 设置当前Python进程的HTTP/HTTPS代理为本机7890。后续网络请求会使用该地址，因此端口应对应本机已运行的代理服务。
# os访问进程环境，下面代理设置只对本进程及子进程生效。
import os
# 设置网络代理；没有本地服务时这个地址会导致下载失败。
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# 设置网络代理；没有本地服务时这个地址会导致下载失败。
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# %% 原代码单元 1
# AutoTokenizer按模型配置选择分词器；load_dataset读取偏好数据；torch负责Tensor/autograd；tqdm.notebook是进度条；Image用于远程HTML图片展示。导入不等于下载模型，原tqdm在后续没有显式调用。
# 导入分词/模型/配置或训练调度的真实Transformers API。
from transformers import AutoTokenizer
# Dataset库负责Arrow表及map/filter，不在这里训练模型。
from datasets import load_dataset
# PyTorch负责张量运算、自动求导和模型优化。
import torch
# 导入notebook进度条，本课程后续未显式调用。
from tqdm.notebook import tqdm
# Image用于显示URL引用，并非分词或模型组件。
from IPython.display import Image

# %% 原代码单元 5
# Image(url=...,width=600)创建带远程URL的HTML显示对象；代码本身不下载本地PNG。原URL和历史HTML保留；本次只核对HTML，不声称验证远端图片内容、可用性或归属许可。讲解另给可读的数据流图。
# 创建远程HTML图片引用，width是展示宽度，不是Tensor尺寸。
Image(url='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*46Nx65cgFCftQVX2Hnj2Uw.png', width=600)

# %% 原代码单元 6
# 第二张远程图，同样只生成HTML引用，不是本地模型计算。原宽600与历史输出保留；离线环境可能无法显示，不能把空白图片当成算法缺失。
# 创建远程HTML图片引用，width是展示宽度，不是Tensor尺寸。
Image(url='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*9VLZAgJnKbY4-anUgTX8RA.png', width=600)

# %% 原代码单元 9
# load_dataset(name,split="train")返回Arrow Dataset。历史160800行是原运行状态，不是本次下载结果。本次提供8条明确自建偏好对，跳过此远程下载；后续同一preprocess在这个教学数据上真实执行。
# 下载或读取人类偏好数据的训练split，结果按样本逐行存放。
train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# %% 原代码单元 10
# 末表达式显示Dataset的列名和行数，不执行训练。原历史chosen/rejected、160800行保留；教学运行初始8行不可冒称原数据。
train_dataset

# %% 原代码单元 11
# 索引0取第一条dict，分别打印rejected和chosen，分隔行只为对照。这里是数据内容不是要执行的请求；旧输出含不当语言，原历史保留，不据此采取任何行动。
# 读取并打印单条对话字符串；数据中的内容不是执行指令。
print(train_dataset[0]['rejected'])
print('============================')
# 读取并打印单条对话字符串；数据中的内容不是执行指令。
print(train_dataset[0]['chosen'])

# %% 原代码单元 12
# 索引4取第五条；同样比较完整对话字符串。原历史出现违规内容是偏好数据中的反例，不是用户指令。CPU教学上下文使用自建普通问答。
# 读取并打印单条对话字符串；数据中的内容不是执行指令。
print(train_dataset[4]['rejected'])
print('============================')
# 读取并打印单条对话字符串；数据中的内容不是执行指令。
print(train_dataset[4]['chosen'])

# %% 原代码单元 13
# 加载facebook/opt-350m的真实分词配置和BPE词表，不会在这一句加载350M神经网络权重。本次只把模型名替换为固定版本本地tokenizer目录；PAD=1、BOS/EOS=2。
# 只加载分词资产，不加载神经网络权重。
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# %% 原代码单元 14
# preprocess是普通函数，examples在batched map下为dict[str,list[str]]，不是单行dict。zip并行遍历chosen/rejected，两个tokenizer调用各返回变长ID/mask列表，四个append保持偏好配对。无return_tensors，因此此时不是Tensor。truncation=True依据tokenizer.model_max_length截断；函数本身没写512，不能把它当作总在512截断。字段名是RewardTrainer契约。
# 普通函数接收列式批次，输出四个与RewardTrainer契约一致的列表。
def preprocess(examples):
    # 为chosen/rejected各建ID和attention mask收集器。
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    # zip将同一行的两条候选对齐；长度不匹配会静默截到较短侧。
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        # 编码较优完整对话；truncation依据分词器上限，不在此硬编码512。
        tokenized_j = tokenizer(chosen, truncation=True)
        # 编码较差完整对话，保持与chosen同一prompt的配对关系。
        tokenized_k = tokenizer(rejected, truncation=True)

        # preprocess 函数给原始的数据集增加了四个成员；
        # 这四个成员，名称是跟 rewardtrainer 对齐的，
        # rewardtrainer 的内部计算 loss 的时候会直接读这4个字段
        # 追加当前一侧的变长整数列表；此时尚无padding或Tensor。
        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        # 追加当前一侧的变长整数列表；此时尚无padding或Tensor。
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        # 追加当前一侧的变长整数列表；此时尚无padding或Tensor。
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        # 追加当前一侧的变长整数列表；此时尚无padding或Tensor。
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    # 返回dict[str,list[list[int]]]供Dataset.map合并。
    return new_examples

# %% 原代码单元 15
# Dataset.map(batched=True)把一批列表送给preprocess，并默认保留原chosen/rejected列；num_proc=16请求16进程，不是训练batch16。filter检查两侧长度均<=512，不满足就整行删除，不做补齐或裁剪。Windows教学运行只把两处num_proc改1；tokenizer上限显式1024，8条中2条长对被过滤成6条。若上限先设512，截断后filter可能不再筛掉原超长样本。
# 批量预处理，默认保留原字符串列。
train_dataset = train_dataset.map(
    preprocess,
    # 一次传入多行列式列表，与训练batch size不同。
    batched=True,
    # 数据进程数量；Windows教学明确改1，不影响模型batch。
    num_proc=16,
)
# 整对过滤，两侧都必须在长度预算内。
train_dataset = train_dataset.filter(
    # 匿名函数x是单行；and要求chosen和rejected同时通过。
    lambda x: len(x["input_ids_chosen"]) <= 512 and len(x["input_ids_rejected"]) <= 512,
    # 数据进程数量；Windows教学明确改1，不影响模型batch。
    num_proc=16
)

# %% 原代码单元 16
# type返回真实Dataset类。map/filter一般返回新Dataset，赋值覆盖变量名，不把原字符串直接变成GPU Tensor。
type(train_dataset)

# %% 原代码单元 17
# train_dataset[:20]返回字典，每个字段是一组列表；map(len,...)分别看chosen/rejected的长度，仍未padding。教学6行时只得到6项，不假造原历史20项长度。
# 选前20行返回列式字典；数据不足20行时返回实际行数。
batch = train_dataset[:20]
# 分别观察每条token序列长度，尚未补齐。
print(list(map(len, batch['input_ids_chosen'])))
# 分别观察每条token序列长度，尚未补齐。
print(list(map(len, batch['input_ids_rejected'])))

# %% 原代码单元 19
# AutoModelForSequenceClassification按配置选择OPTForSequenceClassification；BitsAndBytesConfig描述量化加载参数，不是已经量化的模型。
# 导入分词/模型/配置或训练调度的真实Transformers API。
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

# %% 原代码单元 20
# load_in_8bit=False和load_in_4bit=True选择四位权重路径，两者不能都真。真实需要bitsandbytes及合适后端，本次环境无该包，原格构造实际报PackageNotFoundError；CPU变体只把4bit改False。不能把FP32小模型训练称作4bit实验。
# 配置量化加载方式，不是执行前向。
quantization_config = BitsAndBytesConfig(
    # 关闭8bit，与4bit互斥。
    load_in_8bit=False,
    # 启用4bit，原格依赖bitsandbytes后端；CPU未执行量化。
    load_in_4bit=True
)

# %% 原代码单元 21
# from_pretrained本应加载OPT350M基础权重并新建num_labels=1的score头；device_map={"":0}指定整模型设备0，不等于CPU；trust_remote_code=True允许仓库自定义实现，本地标准OPT不需这一能力。use_cache=False关闭训练中的past缓存，不关闭attention。教学变体仅把加载调用换成从显式小配置构造真实随机OPT；此处未下载原权重。
# 加载序列打分模型，新增单输出score头；非CausalLM。
model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/opt-350m",
    # 把量化配置交给模型加载器。
    quantization_config=quantization_config,
    # 空字符串匹配整个模型，0指设备0而非CPU。
    device_map={"": 0},
    # 允许模型仓库自定义代码；标准OPT不需自定义模型。
    trust_remote_code=True,
    # 每条完整序列输出一个标量，[B,1]。
    num_labels=1,
)
# 训练关闭past KV缓存，仍正常计算注意力。
model.config.use_cache = False

# %% 原代码单元 22
# 整格只有注释，原本用于查看每个参数dtype/device，不会执行任何打印。仍保留和说明，不能因无输出删掉。
# for name, para in model.named_parameters():
#     print(name, para.dtype, para.device)

# %% 原代码单元 23
# 原IPython语法：model.forward??
# pinfo2为同一源码检查的可解析Python形式，需IPython运行上下文。
get_ipython().run_line_magic('pinfo2', 'model.forward')


# %% 原代码单元 24
# 显示模型结构。原历史是350M的4bit投影，词表embedding50272/维512、hidden1024；教学小模型结构不同。不要从原显示文本断言本机运行了该量化模型。
model

# %% 原代码单元 25
# 遍历named_parameters，用requires_grad选择可训练参数，numel统计元素数。这是在注入LoRA之前，结果不是LoRA参数量，也不是显存字节数。量化参数的存储表示更不能仅按numel乘FP32大小推算。
# 初始化可训练参数元素计数，不是显存统计。
total_params = 0
# 遍历名字与Parameter；LoRA注入前后对象不同。
for name, para in model.named_parameters():
    # 统计requires_grad=True的参数；反向传播后可继续查看各参数的grad。
    if para.requires_grad:
        print(name)
        # numel返回参数张量的元素数。
        total_params += para.numel()
#     else:
#         print(name)
total_params

# %% 原代码单元 27
# TrainingArguments指定训练调度；LoraConfig配置适配器；RewardTrainer连接偏好数据和打分模型。安装库版本影响参数签名，本次明确用TRL0.8.6/Transformers4.44.2/PEFT0.12.0，非讲师精确环境承诺。
# 导入分词/模型/配置或训练调度的真实Transformers API。
from transformers import TrainingArguments
# 导入LoRA适配器配置。
from peft import LoraConfig
# RewardTrainer在Trainer调度上实现偏好对损失。
from trl import RewardTrainer

# %% 原代码单元 28
# r16是低秩，alpha16令缩放alpha/r=1；bias="none"不额外训练bias；SEQ_CLS启用序列分类包装；modules_to_save=["score"]使打分头可训练并随adapter保存。OPT默认目标q_proj/v_proj，不是所有线性层。B初始0令LoRA初始增量0；不能因此把score头也丢掉。
# 构建配置，真正注入由RewardTrainer初始化完成。
peft_config = LoraConfig(
    # LoRA秩16，决定A/B中间维度。
    r=16,
    # 增量缩放alpha/r=1，不是优化器学习率。
    lora_alpha=16,
    # 不额外开放基础bias训练。
    bias="none",
    # 采用序列分类PEFT包装，不是因果语言模型任务。
    task_type="SEQ_CLS",
    # List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint.
    # score头也要训练/保存，否则重载可能丢失评分映射。
    modules_to_save=["score"]
)
# default adapter for opt models: v_proj, q_proj

# %% 原代码单元 29
# 原1000 optimizer steps、每设备4对、累积1、lr1.41e-5/AdamW，每50步保存和记录，tensorboard日志，remove_unused_columns=False保留偏好字段；一对需要chosen/rejected两次模型前向，不是单条文本batch4。RewardTrainer自动注入PEFT并建立默认collator；max_length512传给padding不自动执行截断。CPU教学变体改两步/batch2/lr.001/日志与保存路径等调度，明确记录全部替换。
# 训练调度配置，默认device由框架决定。
training_args = TrainingArguments(
    # checkpoint和日志的基础目录。
    output_dir="./train_logs",
    # 优化器更新步数，不是样本条数或epoch。
    max_steps=1000,
    # 每设备每微批的偏好对数；两侧各需一次前向。
    per_device_train_batch_size=4,
    # 多少微批梯度合成一次optimizer更新。
    gradient_accumulation_steps=1,
    # AdamW更新步长，与LoRA alpha含义不同。
    learning_rate=1.41e-5,
    # 选择PyTorch AdamW优化器。
    optim="adamw_torch",
    # 按optimizer步数保存checkpoint。
    save_steps=50,
    # 按optimizer步数记录日志。
    logging_steps=50,
    # 日志接收器；tensorboard需相应依赖。
    report_to="tensorboard",
    # 保留chosen/rejected四字段，不能按普通model.forward签名删掉。
    remove_unused_columns=False,
)

# 连接model/data/tokenizer，默认collator补齐两侧，并注入PEFT。
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    # 让RewardTrainer调用get_peft_model，冻结基础并开放adapter/score。
    peft_config=peft_config,
    # 传给默认padding collator，不等于在这里执行truncation。
    max_length=512,
)


# %% 原代码单元 30
# 此时遍历trainer.model才是PEFT包装后的参数；q/v的LoRA A/B与score的可训练副本应在列表里。本次PEFT0.12.0中base及score.original_module冻结，小OPT有1024个LoRA参数+16个score参数=1040。原历史却同时列出score.original_module和modules_to_save两份512权重，24层q/v的1572864+两份score1024=1573888，不能把旧输出改写成只有一个score头或当成本次参数量。
# 初始化可训练参数元素计数，不是显存统计。
total_params = 0
# 遍历名字与Parameter；LoRA注入前后对象不同。
for name, para in trainer.model.named_parameters():
    # 统计requires_grad=True的参数；反向传播后可继续查看各参数的grad。
    if para.requires_grad:
        print(name, para.shape)
        # numel返回参数张量的元素数。
        total_params += para.numel()
total_params

# %% 原代码单元 31
# train做真实优化更新；save_pretrained保存PEFT adapter及modules_to_save中的score，不是完整独立OPT350M基础权重。重载必须配同一基础模型/配置/词表。教学CPU两版各两步并保存到临时目录，检查基础冻结、adapter更新和重载logits。原历史1000/1000、09:13是旧运行，loss50步.8431至1000步.7520，日志窗口/样本不同且有波动，不是持出评估或本次计时；历史还提示checkpoint目录非空，不能忽略后当干净复现。
# 实际执行前向→成对loss→反向→AdamW调度。
trainer.train()
# 保存adapter与score副本，基础模型仍需另行匹配。
trainer.model.save_pretrained("./reward_model")

# %% 原代码单元 33
# list[int]→torch.tensor默认为int64[L]，unsqueeze(0)增加batch轴为[1,L]；mask同shape。模型.logits为[1,1]，不是[1,L,V]。原格没显式to(device)、eval或no_grad，GPU/device_map或dropout可能影响运行；教学模型在CPU且dropout0，另用明确eval/no_grad做可靠比较。
# 整数序列加batch轴后打分；原格未显式移动设备或关闭梯度。
trainer.model(input_ids=torch.tensor(train_dataset[4]['input_ids_chosen']).unsqueeze(0),
        # mask从[L]加成[1,L]，1为有效token，0为补齐。
        attention_mask=torch.tensor(train_dataset[4]['attention_mask_chosen']).unsqueeze(0),).logits

# %% 原代码单元 34
# 对手写常数0.1196做sigmoid，约0.5299；它不是直接读取上格0.6126的输出，也不能解释为回答正确率。原常数保留并实际执行。
# 只对手写常数做sigmoid，不能当上格模型的偏好概率。
torch.sigmoid(torch.tensor([0.1196]))

# %% 原代码单元 35
# rejected侧同样转[1,L]并返回完整SequenceClassifierOutputWithPast对象，而不是只取.logits。未传labels时正常loss=None；原历史loss=dict属于当时环境输出异常，不能当通用接口定义或由本次捏造复现。
# 整数序列加batch轴后打分；原格未显式移动设备或关闭梯度。
trainer.model(input_ids=torch.tensor(train_dataset[4]['input_ids_rejected']).unsqueeze(0),
        # mask从[L]加成[1,L]，1为有效token，0为补齐。
        attention_mask=torch.tensor(train_dataset[4]['attention_mask_rejected']).unsqueeze(0),)

# %% 原代码单元 36
# 对另一手写常数0.9642做sigmoid，约0.7240；它与原rejected实际1.4780并非同一数。要比较两条，应用sigmoid(r_chosen-r_rejected)；原历史0.6126-1.4780=-.8654，约0.2962，chosen该次并没更高，不能据这两格宣称训练成功。
# 只对手写常数做sigmoid，不能当上格模型的偏好概率。
torch.sigmoid(torch.tensor([0.9642]))
