# 第03讲 PEFT / BLOOM 代码注释阅读版
# 来源：chunhuizhang/personal_chatgpt，提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。
# 无JSON或历史输出；# %% 分格。建议在配套notebook中按单元检查依赖后执行。
# Python 阅读版把 IPython 的 !/% 命令列为注释，并按原顺序保留普通 Python 语句。
# 离线CPU教学调用链见复现/scripts/verify_pipeline.py。

# %% [markdown]
# # 第03讲中文代码说明
#
# 新增注释只解释原行为；原输出是历史记录。
# 原文件另存。安装/模型下载/原7B训练没有因本注释文件生成而执行。CPU补充验证见复现README。

# %% [markdown]
# ## 单元 0：安装命令
#
# 三条`!pip install`由IPython转交shell；不是标准Python语法。`-q`减少日志；第一条安装bitsandbytes、datasets、accelerate、loralib，后两条从Git安装Transformers的main与PEFT默认分支。main不固定SHA，原watermark的dev版本无法精确重建。
#
# 安装会改环境并联网，不应把本次注释静态核对算作实际安装。原单元没有安装watermark。loralib虽被安装，后面走的是PEFT接口。中文注释说明三条命令各自用途，原命令不改；验证时转换为IPython AST，但不执行这些安装命令。

# %%
# IPython的!将命令交给shell安装依赖；Git默认分支未锁提交，本次核对不执行安装。
# ! 是 IPython 交给 shell 执行的前缀；-q 只减少日志，不改变安装版本。
# 这三行会联网并修改环境；从 main/默认分支安装不等于2023年的历史dev版本。
# 原命令未安装 watermark；loralib 被安装，但后续 adapter 实际由 PEFT 构造。
# 安装量化、数据、训练分派与loralib依赖；后续LoRA实际走PEFT。
# IPython原指令（在notebook中执行）：!pip install -q bitsandbytes datasets accelerate loralib
# 从会变化的main安装Transformers。
# IPython原指令（在notebook中执行）：!pip install -q git+https://github.com/huggingface/transformers.git@main
# 从默认分支安装PEFT，缺少历史SHA。
# IPython原指令（在notebook中执行）：!pip install -q git+https://github.com/huggingface/peft.git

# %% [markdown]
# ## 单元 1：summary 标题
#
# 这个 Markdown 二级标题引出接下来的总体流程。

# %% [markdown]
# ## summary

# %% [markdown]
# ## 单元 2：任务与流程概要
#
# 原任务是把名言映射到标签文本，模型是BLOOM。LoRA冻结原权重、加低秩参数；PEFT为实现库。MLM通常预测被遮盖位置，CLM预测下一token，本实现选择后者。
#
# 原文字把MLM/CLM称作unsupervised，是强调可从文本自动构造token目标，不表示这个有quote/tags关系的任务没有监督信息。pipeline在这里是流程概念，原文件未调用`transformers.pipeline()`。

# %% [markdown]
# - bigscience/bloom-7b1
# - lora fine-tune bloom: 可插拔式的（plugin/adapter）
#     - freeeze original weights
#     - plugin lora adapters (peft)
# - huggingface transformers 库
#     - trainer.train 的参数及过程；
#     - mlm 与 clm 的差异：（都是 unsupervised learning，都可以自动地构建 input/labels）
#         - mlm：bert
#         - clm：gpt（bloom）
#     - pipeline
#         - dataset/tasks
#         - tokenizer
#         - training (fine-tune base lora)
#         - inference

# %% [markdown]
# ## 单元 3：base model 与 adapters 标题
#
# Markdown标题区分共享大模型与任务增量；不是Python代码，不执行冻结或注入。

# %% [markdown]
# ## base model & lora adapters

# %% [markdown]
# ## 单元 4：导入依赖
#
# torch提供Tensor/autograd，`torch.nn as nn`绑定神经网络模块别名；bitsandbytes绑定bnb，但后续没有直接调用bnb函数，量化由Transformers加载路径触发。三个Auto类按配置选对应的tokenizer/config/因果LM；LoraConfig与get_peft_model分别描述adapter配置和执行注入。
#
# 导入依赖包，供后面的模型加载与量化调用使用；bitsandbytes 需要与 CUDA 和相关包版本匹配，环境信息见复现说明。

# %%
# 导入Tensor/神经网络、量化加载、Auto接口及LoRA配置/注入；导入不等于加载7B。
# import torch 绑定模块；import torch.nn as nn 为神经网络模块指定别名。
# AutoTokenizer 加载分词规则；AutoConfig 加载结构配置；AutoModelForCausalLM 构造因果LM。
# LoraConfig 是配置类，get_peft_model 是注入函数；导入这两者尚未添加 adapter。
# bnb 是 bitsandbytes 的模块别名；本文件未直接调用 bnb，量化由加载路径触发。
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# %% [markdown]
# ## 单元 5：加载 watermark 扩展
#
# `%load_ext watermark`调用IPython扩展加载机制，需要另有watermark包。状态是当前kernel加载扩展；不改变模型。标准Python脚本直接解析百分号会失败，所以注释核对先做IPython语法转换。

# %%
# 加载watermark扩展，要求包已安装；不是普通Python语法。
# %load_ext 是 IPython magic，加载已安装扩展到当前内核；不会安装缺失的 watermark。
# 标准 Python 文件不能直接解析百分号语法；Python阅读版把此原指令记为注释。
# IPython原指令（在notebook中执行）：%load_ext watermark

# %% [markdown]
# ## 单元 6：查看已导入库版本
#
# `%watermark --iversions`打印当前会话导入模块的版本，用于环境记录。没有Tensor输入/输出。保存的历史输出来自老师环境，不是这次CPU环境；没有Git SHA的dev版本字符串不能固定源码。

# %%
# 打印当前导入模块的版本，便于选择对应版本的API文档。
# --iversions 报告当前内核导入模块的版本；保存的旧输出不是当前运行环境。
# 版本字符串中的dev0表示开发版；精确定位源码还需要Git提交号。
# IPython原指令（在notebook中执行）：%watermark --iversions

# %% [markdown]
# ## 单元 7：函数方式查看环境
#
# 从 watermark 导入同名函数，packages 接收逗号分隔的包名字符串，返回版本报告供 print 输出。它与前格 magic 提供两种调用方式；量化组件的版本可另行加入 packages 查询。

# %%
# 使用函数方式打印指定包的版本；返回字符串，不做模型计算。
# watermark 此时是从同名包导入的独立函数；packages 接收逗号分隔的 str。
# 返回环境报告字符串，再用print显示所列库的版本。
from watermark import watermark
print(watermark(packages='peft,torch,loralib,transformers,accelerate,datasets'))

# %% [markdown]
# ## 单元 8：加载模型与 tokenizer
#
# AutoModelForCausalLM.from_pretrained为类方法，输入模型ID字符串、load_in_8bit=True、device_map='auto'，返回因果LM模块；可能下载权重、分配设备内存和设置dispatch hook。AutoTokenizer.from_pretrained单独加载匹配的分词器，不加载Transformer权重。
#
# 8bit是旧式bitsandbytes量化入口，不是所有参数/激活均为int8；auto是设备分派策略，不是数据并行副本数。这里未指定revision，无法精确锁定模型/词表快照。三个加载参数与tokenizer调用的作用见相邻注释，运行条件见复现说明。

# %%
# 加载BLOOM因果模型和匹配分词器；可能下载权重并分配设备内存。
# from_pretrained 是类方法；模型ID为 str，可触发配置/权重下载与设备内存分配。
# 返回 model: nn.Module；原 BLOOM-7b1 的隐藏宽度 H=4096、词表 V=250880。
# 尚无输入 token；加载模型不会自动得到 logits 或生成回答。
# tokenizer 是独立分词器对象；其ID及特殊token必须与预训练模型相容。
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    # 旧版bitsandbytes量化加载；不保证全部参数或激活都是int8。
    load_in_8bit=True,
    # 自动设备分派，不是数据并行副本数。
    device_map='auto',
)

# 单独加载词表与分词规则，模型权重不在tokenizer内。
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

# %% [markdown]
# ## 单元 9：查看配置
#
# 第一行`# model.config`未执行；实际调用AutoConfig.from_pretrained返回配置对象。它读取配置并返回独立配置对象，供后面的属性查看使用。参数值应以固定模型revision核对，原文件缺revision。

# %%
# 读取结构配置；注释中的model.config不执行；这里不再次加载全部权重。
# AutoConfig.from_pretrained 返回模型结构配置对象，读取的是配置而不是全部权重。
# 没有赋给变量，notebook会展示最后表达式；# model.config 原注释不执行。
# 配置加载仍可能联网；不是纯粹访问已经存在的 model.config。
# model.config
AutoConfig.from_pretrained("bigscience/bloom-7b1")

# %% [markdown]
# ## 单元 10：查看原模型结构
#
# 末尾表达式 model 让 notebook 展示模型树；普通脚本中可使用 print(model)。沿模型树找到融合的 query_key_value 投影。

# %%
# notebook显示模型的模块结构；普通Python脚本中可用print(model)查看。
# model 是前面创建的模块对象；单独写对象名字会让 notebook 显示 repr。
# 查看模型的模块结构，沿层名寻找embedding、attention和输出层。
model

# %% [markdown]
# ## 单元 11：查看输入 embedding
#
# 注释里的具体属性访问不执行；实际`model.get_input_embeddings()`是实例方法，返回输入Embedding模块，未对任何ID做查表。其权重通常为`[V,H]`，实际lookup将int64输入`[B,S]`变成浮点激活`[B,S,H]`。
#
# 本格不解冻权重；返回模块也不是复制一份参数。默认参数为空，模块无输入Tensor。

# %%
# 返回输入Embedding模块，未对token ID查表，也未解冻权重。
# get_input_embeddings() 是 model 的方法，无显式实参，返回输入词嵌入模块。
# 调用返回模块而不是嵌入结果；真正查表时输入 int64 IDs [B,S]，输出浮点 [B,S,H]。
# 原注释中的属性路径未执行；本调用也不改变 embedding.weight.requires_grad。
# model.transformer.word_embeddings
model.get_input_embeddings()

# %% [markdown]
# ## 单元 12：查看 tokenizer
#
# 展示 tokenizer 的配置、特殊 token 等信息；下一步传入文本时再进行编码。

# %%
# 展示分词器对象，可查看词表和特殊token配置。
# tokenizer是分词器对象；向它传入文本时，会返回对应的token编号。
# 真正分词在36/38/44/45，调用 tokenizer(text) 才会创建 BatchEncoding。
tokenizer

# %% [markdown]
# ## 单元 13：冻结标题
#
# Markdown标题指出接下来冻结基权重。标题本身不改变requires_grad。

# %% [markdown]
# ### freeze original weights

# %% [markdown]
# ## 单元 14：只看首个参数 dtype
#
# model.parameters返回迭代器，list将其收集为参数引用列表，`[0]`取第一个Parameter，`.dtype`返回类型对象。参数必须存在。
#
# 只观察第一项，不能把结果推广到全部Linear8bitLt/norm/adapter。没有Tensor拷贝或dtype转换，后格才实际修改。

# %%
# 查看首个参数的 dtype；需要了解各层精度时，可继续遍历全部参数。
# parameters() 返回参数迭代器；list(...) 将其物化，索引[0]取得第一个 Parameter。
# .dtype 是类型属性；这不是全模型dtype统计，也不证明量化权重、激活和adapter精度一致。
list(model.parameters())[0].dtype

# %% [markdown]
# ## 单元 15：冻结和一维参数转 FP32
#
# `enumerate(model.parameters())`给每个Parameter提供编号i和param；i只用于原注释中的调试打印，没有参与计算。循环将requires_grad=False，再用ndim==1选择一维权重/偏置，把底层data转为float32。shape保持，参数存储/类型改变。
#
# 一维条件不是“所有小参数”或“只有LayerNorm”。二维Embedding仍被冻结而不被此条件cast；冻结不等于跳过其计算。`.data`绕过一般autograd跟踪，是原历史写法；注释版保留，未擅自改算法。本次在小型真实BLOOM上执行该循环，核对所有基参数冻结。

# %%
# 冻结现有权重，将一维参数转为FP32；梯度仍可沿这些运算传回可训练的LoRA参数。
# param 是 Parameter；requires_grad=False 阻止其梯度累积，后面才加入可训练LoRA因子。
# 冻结权重不等于关闭整条计算图，adapter学习仍需要梯度穿过模型中间运算。
# ndim 是维度数，一维参数也可能是 bias；这里不是只选择LayerNorm。
# .data 的历史写法直接改底层存储；to(float32) 保留shape，并不是优化器step。
# 遍历Parameter并给出编号；i不参与计算。
for i, param in enumerate(model.parameters()):
    # 关闭基权重的梯度累积，LoRA会在后面另加可训练参数。
    param.requires_grad = False  # freeze the model - train adapters later
#     print(i, 'param.requires_grad = False')
    # 选择所有一维参数，不只LayerNorm。
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        # 按原历史写法改底层存储dtype，shape保持；绕过一般autograd跟踪。
        param.data = param.data.to(torch.float32)
#         print(i, 'ndim == 1, torch.float16 to torch.float32')

# %% [markdown]
# ## 单元 16：checkpointing 与输入激活梯度
#
# 两个实例方法无显式参数。gradient_checkpointing_enable改变模型的激活保存/重算设置；不把训练checkpoint写入磁盘。enable_input_require_grads安装hook，使Embedding输出激活requires_grad=True；不会把Embedding.weight解冻。
#
# 目的在于冻结模型与激活重算时维持adapter所需梯度链。实际CPU验证核对Embedding权重仍False，而输出激活为True，并检查小模型开启重算标志。

# %%
# 开启激活重算并使embedding输出激活可求梯度；不解冻embedding权重。
# 两个调用均修改模型行为：激活checkpointing保存较少中间值，反向时重新计算。
# enable_input_require_grads 对embedding输出设置梯度hook；权重仍然被冻结。
# 例如IDs [B,S] -> embedding激活 [B,S,H]；激活可求梯度和权重可训练是两件事。
# reduce number of stored activations
# 减少保存的激活，反向时重算，不是写磁盘checkpoint。
model.gradient_checkpointing_enable()
# 开启embedding输出激活梯度，不改变其weight.requires_grad=False。
model.enable_input_require_grads()

# %% [markdown]
# ## 单元 17：lm_head 输出转 FP32
#
# CastOutputToFloat是nn.Sequential子类；forward(self,x)覆盖父方法，super().forward先运行被包装lm_head，再把返回Tensor转float32。它接收`[B,S,H]`，返回`[B,S,V]`，shape不因cast改变。
#
# 把model.lm_head赋为包装对象修改模块树，原head变成其子模块；不是创建另一个词表，也不是解冻head。class/继承/self/super分别是类型定义、复用父实现、当前实例和父方法访问。没有自写__init__，沿用Sequential构造行为。

# %%
# 包装lm_head：先执行原head，再把[B,S,V]输出转FP32。
# class 定义子类；nn.Sequential 管理按顺序调用的子模块，包装器沿用父类构造函数。
# forward 是模型方法，self是当前实例；调用模块时PyTorch经__call__调度forward。
# super() 委托父类执行已有head，不是再次构造模型；[B,S,H] -> logits [B,S,V]。
# .to(torch.float32) 只转换输出dtype，没有把head权重整体转FP32。
# 继承Sequential，沿用其构造函数，原head成为子模块。
class CastOutputToFloat(nn.Sequential):
    # self是当前实例，x为[B,S,H]输入激活。
    def forward(self, x):
        # 调用父类运行head，输出[B,S,V]，只转换dtype。
        return super().forward(x).to(torch.float32)
# 替换模块属性，改变模块树；不是创建另一个词表。
model.lm_head = CastOutputToFloat(model.lm_head)

# %% [markdown]
# ## 单元 18：LoRA 标题
#
# Markdown过渡到adapter部分，本格未注入任何矩阵。

# %% [markdown]
# ### LoRa Adapters

# %% [markdown]
# ## 单元 19：统计可训练参数的函数
#
# print_trainable_parameters(model)是独立函数，无默认参数。named_parameters迭代名字与Parameter，`_`忽略名字；numel统计元素数。all_param计全部，trainable_params仅计requires_grad=True，再用f-string打印比例。返回None，不改参数。
#
# 第一行docstring是函数说明str，不是运行命令；定义时不会调用统计循环，单元21才调用。全空模型会除零，原函数未处理；真实BLOOM有参数。量化参数元素计数不是实际显存字节数。

# %%
# 定义独立统计函数；只读requires_grad/numel并打印，默认返回None。
# model 是唯一实参，没有默认值；函数只统计，不修改参数或返回模型。
# named_parameters() 给出 (名字, Parameter)；numel() 统计元素数，不能当显存字节。
# 函数定义后不会立即运行；21格调用它，print之后隐式返回None。
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # 初始化可训练参数计数。
    trainable_params = 0
    # 初始化全部参数计数。
    all_param = 0
    # 遍历名字和参数；下划线表示忽略名字。
    for _, param in model.named_parameters():
        # numel为元素数，不是量化显存字节数。
        all_param += param.numel()
        # 只累加可训练的Parameter。
        if param.requires_grad:
            # 累计当前可训练Parameter元素数。
            trainable_params += param.numel()
    # 打印计数和比例；全空模型会除零，原函数未处理该边界。
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# %% [markdown]
# ## 单元 20：LoRA 配置
#
# 重复导入一般复用模块缓存，不是重复安装。LoraConfig构造配置对象：r=16，lora_alpha=32，标准scale=32/16=2；dropout=0.05只用于adapter支路；bias='none'不额外训练偏置；task_type='CAUSAL_LM'选择因果LM包装。
#
# target_modules 这一行处于注释中，当前 BLOOM 使用默认的 query_key_value 目标映射。r 决定低秩矩阵的尺寸，alpha 决定修正分支的缩放。

# %%
# 创建LoRA配置：r为瓶颈宽度，alpha/r=2为增量缩放系数。
# config是LoraConfig实例，后面的get_peft_model读取它来注入LoRA模块。
# 融合QKV输入 [B,S,H]；标准LoRA分支经 A:[r,H]、B:[3H,r] 后输出 [B,S,3H]。
# 图与库采用nn.Linear权重[out,in]布局，不能混用上一讲行向量因子的存储方向。
# target_modules 行是注释；核对版本中BLOOM默认映射为 query_key_value，不是 q_proj/v_proj。
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    # 瓶颈维度16，对应因子的共同小维度。
    r=16, #low rank
    # 缩放分子32，标准scale=32/16=2。
    lora_alpha=32, #alpha scaling， scale lora weights/outputs
    # target_modules=["q_proj", "v_proj"], #if you know the
    # 训练时adapter分支输入dropout概率0.05。
    lora_dropout=0.05,
    # none表示不额外训练bias。
    bias="none",
    # 因果语言模型任务；target_modules注释未生效，BLOOM默认映射融合QKV。
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

# %% [markdown]
# ## 单元 21：注入并统计参数
#
# get_peft_model(model,config)返回PEFT包装，并改造原模型内部目标模块；赋回model名字，因此后续调用使用adapter模型。不是对原模型深拷贝。随后调用前格函数打印计数。
#
# 原7B的融合QKV权重方向为`[12288,4096]`，新增A`[16,4096]`、B`[12288,16]`；激活为`[B,S,4096]→[B,S,16]→[B,S,12288]`再与基输出相加。每层新增262144个参数，30层推得7864320；真实目标/计数仍应检查版本与模型。新CPU验证使用2层H=16的随机模型，结果按它的真实尺寸单独记录。

# %%
# 注入并包装目标模块，赋回model；随后统计可训练参数，不是深拷贝。
# 返回外层PEFT模型包装，并在传入基模型的目标层加入adapter；不会深拷贝全部权重。
# 原大权重保留并冻结，新增A/B参与训练；此处配置r=16，标准缩放alpha/r=32/16=2。
# 以原H=4096为例：A权重[16,4096]、B权重[12288,16]；单层因子共262144元素。
model = get_peft_model(model, config)
print_trainable_parameters(model)

# %% [markdown]
# ## 单元 22：查看包装后的模型
#
# 展示注入 LoRA 后的 model，沿 PeftModel 包装找到新增模块。可结合参数遍历和 requires_grad 查看哪些权重参与训练。

# %%
# 显示注入后的模型结构，不能用repr代替训练结果或冻结状态核对。
# 注入后仍需用requires_grad和真实前向核对；结构repr不能代替这些检查。
# 同一个模型以后接收token输入、经BLOOM计算logits，adapter只是选定投影上的增量。
model

# %% [markdown]
# ## 单元 23：pipeline 标题
#
# Markdown流程标题，不调用任何pipeline类或函数。

# %% [markdown]
# ## pipeline

# %% [markdown]
# ## 单元 24：data 标题
#
# Markdown引出数据检查，尚未下载dataset。

# %% [markdown]
# ### data

# %% [markdown]
# ## 单元 25：加载名言数据
#
# import transformers绑定包名，后面trainer接口使用；from datasets导入load_dataset。load_dataset('Abirate/english_quotes')可能联网、缓存并返回DatasetDict；原调用没固定revision，也没指定split，后面取train。
#
# 数据字段quote/author通常为str，tags为list[str]。读取后得到Dataset中的Python对象；配套算例使用5条自编记录展示取行、取列等Dataset操作。

# %%
# 加载名言数据集，并按需访问网络或缓存。原数据集 ID 使用 Hub 默认版本；配套运行数据见复现说明。
# load_dataset 是独立函数；模型与数据加载相互独立，不会因为load_dataset而训练模型。
# 原ID是 str，返回DatasetDict，train是一个split；未固定revision，样本快照可能变化。
# 本次教学用明确新编的5条本地数据，未下载原Abirate数据。
import transformers
from datasets import load_dataset
dataset = load_dataset("Abirate/english_quotes")

# %% [markdown]
# ## 单元 26：查看 DatasetDict
#
# dataset 展示各个 split、字段和行数。DatasetDict 将 split 名称映射到相应 Dataset，随后再进行文本处理和分词。

# %%
# 显示 DatasetDict 的 split 结构，便于选择后续处理的训练集。
# dataset 是 DatasetDict，按split名映射到Dataset；本格只显示结构。
# 此时字段如 quote/author/tags 是数据记录，不是GPU Tensor。
dataset

# %% [markdown]
# ## 单元 27：取得 train split
#
# dataset['train']按名称取Dataset；与list索引不同。必须存在train键，缺少会报KeyError。返回数据集对象，不是某一行，也未转Tensor。

# %%
# 按名称取得train Dataset，返回整个split，不是单行。
# 通过train键取出训练集Dataset，后续再选择行或字段。
# 这里按名称选择已有的数据划分。
dataset['train']

# %% [markdown]
# ## 单元 28：转 pandas 表格
#
# to_pandas返回DataFrame，便于查看列和行；可能物化大量数据到内存。不会改变Dataset字段，也不会把数据送到GPU。大数据不应默认全部转表。

# %%
# 转换为pandas表格以查看数据，可能将数据物化到内存。
# to_pandas 是Dataset的方法，无显式参数；返回 pandas.DataFrame。
# 这是浏览数据的表格，可能物化到内存；后面的Trainer仍接收Dataset，不接收这个表格。
dataset['train'].to_pandas()

# %% [markdown]
# ## 单元 29：quote 列的前四条
#
# 先`['quote']`取列，再`[:4]`切前四项，结果为list[str]。从0开始、不含右端点4。列表的每一项是一条完整名言文本。

# %%
# 先取quote列，再切前四条完整文本；不是前四个token。
# 先按列名quote取得字符串列，再[:4]按行取前四条，结果为list[str]。
# 切片上界4不包含索引4；字符串内容保留，不是切前四个token。
dataset['train']['quote'][:4]

# %% [markdown]
# ## 单元 30：author 列的前四条
#
# 与前格相同的取列再切片，返回作者字符串列表。后面的merge选用quote和tags构造训练文本。

# %%
# 取前四条记录的作者字段，查看数据内容；后续训练文本由 quote 和 tags 拼成。
# author列的前四个字符串仅用于查看；后面merge不把author拼入训练文本。
# 作者字段用于了解样本来源，训练目标由后面的 quote 与 tags 字段组成。
dataset['train']['author'][:4]

# %% [markdown]
# ## 单元 31：同时看前四条的所有列
#
# Dataset的`[:4]`按行批取，结果为dict，各列值是长度4的列表；不是list[dict]。与单元29只取quote列不同。后续batched map同样处理按列组织的批次字典。

# %%
# 批取前四行，返回各字段对应列表的dict，不是list[dict]。
# Dataset切片[:4]返回列名 -> 值列表的dict，而不是四个单行dict组成的list。
# 例如 quote:list[str]、tags:list[list[str]]；后面batched map也使用列列表字典。
dataset['train'][:4]

# %% [markdown]
# ## 单元 32：把标签列表变成字符串
#
# 先取tags列再取第0条list[str]，str将其转成带方括号/引号的Python列表展示文本。不是JSON序列化、不是类别ID，也未tokenize。空列表会成为'[]'，之后模型学的是这类字符串形式。

# %%
# 将第一条tags列表转成Python展示字符串，非JSON或类别ID。
# tags[0] 是第一条标签列表；str(list) 包含方括号和引号，成为文本续写目标的一部分。
# 它不是JSON序列化，也不是每个标签对应一个固定分类类别ID。
str(dataset['train']['tags'][0])

# %% [markdown]
# ## 单元 33：merge 函数与 Dataset.map
#
# merge(row) 接收单行 dict，取 quote，加上固定分隔符 ' ->: ' 和 str(tags)，把拼接结果写入 prediction 字段，再返回该记录。这里 prediction 是整理好的训练文本。
#
# Dataset.map逐行调用函数并返回新Dataset，再赋回train split；原API不是对整个Dataset直接原地添列。其他split若存在不会在此合并，后续DatasetDict.map访问prediction时可能失败。作者字段保留但没有拼进文本。
#
# 本次提取原函数实际执行，例如quote='Be yourself.'、tags=['life']得到`Be yourself. ->: ['life']`，并核对传入dict被修改。后续对新编数据运行真实map。

# %%
# 拼接名言、固定分隔符和标签字符串；prediction是文本列名，不是模型预测。
# row 是 Dataset 的一条 dict 记录，quote 为字符串，tags 为字符串列表；调用时传入这条记录。
# 输入名言和标签经字符串拼接形成prediction:str；prediction不是model的推理结果。
# 返回同一个被添加字段的row；外层Dataset.map返回新Dataset，赋回train split。
# 这个独立函数接收一条 dict 记录。
def merge(row):
    # 给传入dict原地添加prediction字段；author不参与拼接。
    row['prediction'] = row['quote'] + ' ->: ' + str(row['tags'])
    # 返回同一dict，供Dataset.map接收。
    return row
# map返回新Dataset，再赋回train；其他split未在此处理。
dataset['train'] = dataset['train'].map(merge)

# %% [markdown]
# ## 单元 34：查看拼接后前五条
#
# 取prediction列再[:5]，结果为list[str]，比前面检查多一条。要求单元33已成功创建字段；没有model.generate。

# %%
# 查看新增prediction列的前五条；要求merge/map已成功。
# prediction列为list[str]；前五条用于核对merge的格式、分隔符和标签列表文本。
# 这一格展示拼接后的训练文本，可检查问题、名言和标签的排列。
dataset['train']['prediction'][:5]

# %% [markdown]
# ## 单元 35：查看第五条记录
#
# Dataset整数索引[4]返回单行dict，各字段为标量/列表，和[:4]的列列表字典不同。五条以下的数据会越界；本次教学数据明确有5条。

# %%
# 按从0开始的索引取第五条单行dict；与切片[:4]返回形态不同。
# [4] 是第5条单行记录，返回字段 -> 单个值的dict；不同于[:4]的列列表字典。
# 依赖数据至少有5行；新编CPU教学数据也安排5行，以实际覆盖这个索引。
dataset['train'][4]

# %% [markdown]
# ## 单元 36：试分词前四条
#
# 调用tokenizer对象的__call__，输入list[str]，返回BatchEncoding，默认是input_ids/attention_mask等列表，不是torch Tensor。原调用没padding/truncation，长度可不同，未截断超长序列。
#
# 默认可能加入该词表定义的特殊token，精确ID取决于tokenizer快照。试调用没有把返回值赋回dataset；正式转换在38。新教学tokenizer不是BLOOM真实词表，ID仅用于API和loss路径验证。

# %%
# 试分词前四条文本，默认返回列表形式BatchEncoding，未写回dataset。
# tokenizer.__call__ 输入list[str]，返回BatchEncoding映射；默认input_ids等仍为Python列表。
# 未指定return_tensors、padding、truncation；各行长度可能不同，也不主动截断超长输入。
# 返回值未赋回Dataset；这里只试分词，38才正式写入token列。
tokenizer(dataset['train']['prediction'][:4])

# %% [markdown]
# ## 单元 37：tokenize 标题
#
# Markdown标题标记正式数据转换阶段，无函数调用。

# %% [markdown]
# ### tokenize

# %% [markdown]
# ## 单元 38：批量 map 分词
#
# lambda samples为匿名函数，取samples['prediction']字符串列表，调用tokenizer。batched=True让map传入列列表字典，默认非此文件指定的固定微批大小；不等于训练batch_size=4。
#
# dataset为DatasetDict，map对所有split执行并返回新对象；添加input_ids与attention_mask列，原quote/tags等列默认仍保留。因未指定padding/truncation，后面collator负责动态补齐。Trainer通常移除forward不接收的文本列；直接调用collator时应只传模型输入字段。

# %%
# batched map传入字符串列表；添加token列；没有在此padding或truncation。
# lambda定义匿名函数，samples为一批记录的列列表字典；函数返回分词后的字段映射。
# batched=True 是map处理模式，不是Trainer微批大小4；DatasetDict会对每个split执行map。
# 添加input_ids:list[int]、attention_mask:list[int]等字段；原文本列默认保留。
# 若额外split没有prediction，本匿名函数会KeyError；原文只给train做了merge。
dataset = dataset.map(lambda samples: tokenizer(samples['prediction']), batched=True)

# %% [markdown]
# ## 单元 39：查看分词后数据集
#
# 展示分词后的数据结构，重点看 input_ids 与 attention_mask 两列；数据整理器随后将列表组合成训练张量。

# %%
# 展示分词后的字段；当前保存的是token编号和mask列表。
# 注释中的字段名只帮助阅读；实际表达式dataset显示token列和split结构。
# 本格不构造 [B,S] Tensor；转换和补齐发生在后面collator。
# 'input_ids', 'attention_mask'
dataset

# %% [markdown]
# ## 单元 40：training 标题
#
# Markdown转入训练，标题不更新权重。

# %% [markdown]
# ### training

# %% [markdown]
# ## 单元 41：导入训练接口
#
# Trainer组织训练，TrainingArguments保存调度/批次/精度等配置，DataCollatorForLanguageModeling构造语言模型批次和labels。导入不开始训练，也不是定义模型loss公式；BLOOM内部处理CLM位移。

# %%
# 导入训练调度、训练参数和CLM批次构造接口；导入不会启动训练。
# Trainer是训练控制器，TrainingArguments是参数对象；collator是样本到batch的转换器。
# CLM的token位移与交叉熵由BLOOM内部实现，不是在这条import里计算。
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# %% [markdown]
# ## 单元 42：构造 Trainer 并训练
#
# Trainer接收model、train Dataset、TrainingArguments和collator。原参数无独立eval_dataset，不能从训练日志推出验证集效果。
#
# | 参数 | 值与含义 | 容易混淆的边界 |
# |---|---|---|
# | per_device_train_batch_size | 4，每设备微批样本数 | 不是token长度 |
# | gradient_accumulation_steps | 4，累积四个微批再更新 | 单个完整周期单设备有效batch为16 |
# | warmup_steps | 100，优化step预热数 | 不是前100条样本 |
# | max_steps | 200，优化更新上限 | 不是固定200个微批 |
# | learning_rate | 2e-4 | 与LoRA scale=2不同 |
# | fp16 | True，训练混合精度 | 不表示每个参数都为fp16 |
# | logging_steps | 1，更新日志间隔 | 不保证额外评估 |
# | output_dir | outputs，训练输出目录 | 无显式save_pretrained导出流程 |
#
# mlm=False选择CLM collator：动态padding后input_ids/attention_mask为int64`[B,Smax]`，labels复制ID并将padding目标设-100。原BLOOM内部logits`[B,S,V]`向左切一格，与labels右移一格配对计算交叉熵；不会让当前位置预测自己。全段非padding文本计loss，不仅标签回答段。
#
# model.config.use_cache=False修改配置，使训练与checkpointing配合；不冻结参数或关闭autograd。trainer.train实际组织forward/loss/backward/累积与optimizer更新，改变adapter和训练状态并可能写文件。auto设备分派不是数据并行world_size，不能直接按GPU数扩大有效batch。
#
# 配套 CPU 算例使用随机小型 BLOOM 和真实 Trainer 训练2步：fp16=False、微批2、累积2、warmup0。原训练配置及历史输出保留在 notebook 中，运行记录见复现说明。

# %%
# 构造Trainer后执行训练；会改变adapter和训练状态，原7B训练本次未运行。
# 样本列表 -> collator -> int64 input_ids/attention_mask/labels [B,Smax]。
# mlm=False：复制input_ids作为labels，padding目标设-100，让交叉熵忽略补齐位置。
# BLOOM输出logits [B,S,V]；logits[:,:-1,:]与labels[:,1:]配对，预测下一token。
# 原目标对整段非padding文本计算loss，没有只保留回答标签段的mask。
# 单设备完整累积周期的有效batch=4*4=16；多卡数量不能由device_map=auto直接推断。
# 原200步指optimizer更新数；Trainer负责梯度累积、反向与优化器step。
# 本格构造的trainer会持有model/dataset/collator，train会改adapter及训练状态并可能写文件。
# CPU验证另设随机小模型、fp16=False、微批2/累积2/2步；未执行原7B的FP16 200步训练。
trainer = Trainer(
    # 使用已经注入adapter的model对象。
    model=model,
    # 使用train split，未提供eval_dataset。
    train_dataset=dataset['train'],
    args=TrainingArguments(
        # 每设备每微批4条，不是有效batch总数。
        per_device_train_batch_size=4,
        # 累计4个微批再更新，单设备完整周期有效batch=16。
        gradient_accumulation_steps=4,
        # 前100个优化step预热。
        warmup_steps=100,
        # 上限200次优化更新，不是200个样本。
        max_steps=200,
        # 优化器学习率2e-4，与LoRA scale不同。
        learning_rate=2e-4,
        # 混合精度训练，非所有参数强制fp16。
        fp16=True,
        # 每次优化更新记录日志。
        logging_steps=1,
        # 训练文件写入outputs目录。
        output_dir='outputs'
    ),
    # mlm=False：动态padding、创建labels，padding目标设-100。
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
# 关训练生成缓存以配合checkpointing，不关闭autograd。
model.config.use_cache = False
# 进入真实训练调度、前向、loss、反向、梯度累积与优化器更新。
trainer.train()

# %% [markdown]
# ## 单元 43：inference 标题
#
# Markdown开始生成示例，未自动切换eval模式或关闭梯度。

# %% [markdown]
# ### inference

# %% [markdown]
# ## 单元 44：第一条生成示例
#
# tokenizer 接收带有 ' ->: ' 的字符串，return_tensors='pt' 使 BatchEncoding 内的编号成为 [1,S] 整数张量。CUDA autocast 控制浮点运算的精度；输入设备由后续迁移操作决定。
#
# `**batch`把input_ids/attention_mask作为关键字实参传给model.generate；max_new_tokens=50限制新增token，输出整数`[1,S+G]`且G≤50。`output_tokens[0]`取一行；decode(...,skip_special_tokens=True)返回str，通常仍包含输入前缀。print额外输出换行。
#
# 生成前检查 model 的模式、输入设备与 dispatch hook。CPU 配套例子显式调用 eval 和 no_grad，生成3个新增 token，便于观察输入前缀与续写结果。

# %%
# 将第一条prompt分词为Tensor，生成最多50个新token，再decode；autocast不搬设备。
# batch:BatchEncoding，以pt输出int64 input_ids/attention_mask [1,S]，默认仍在CPU。
# with 是上下文管理；CUDA autocast只控制部分浮点计算精度，不负责输入设备搬运。
# **batch 展开关键字实参；generate输出整数token序列 [1,S+G]，G最多50。
# decode的输入是一行token IDs，返回str；skip_special_tokens跳过特殊ID，通常仍含prompt。
# 原文没有显式eval()/inference_mode；此处注释不添加执行逻辑，原CUDA生成未复现。
# return_tensors=pt产生[1,S]整数Tensor，原代码默认在CPU。
batch = tokenizer("“Training models with PEFT and LoRa is cool” ->: ", return_tensors='pt')

# CUDA AMP上下文；不会把batch搬到CUDA或显式设置eval。
with torch.cuda.amp.autocast():
    # **batch展开关键字输入，输出含prompt及最多50个新token。
    output_tokens = model.generate(**batch, max_new_tokens=50)

# 取第一行decode并跳过特殊token；通常仍含输入前缀。
print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# %% [markdown]
# ## 单元 45：第二条生成示例
#
# 与44完全相同的API路径，只把prompt换成更长的论文风格句子。开头空白行保留；长文本使S变化，不是新的模型、独立验证集或分类head。
#
# 字符串中的引号、标点也被tokenize。max_new_tokens仍50，autocast仍不搬设备，decode仍包含前缀。两条prompt都需解释，不能用“同上”替代本格的输入差别及评估边界。

# %%
# 更长的第二条prompt，路径与前格相同；不是独立评估集，原CUDA生成未运行。
# 第二条prompt更长，所以输入长度S与第一条不同；仍使用同一model与tokenizer。
# 输出上限是新增50个token，不是总长度50；更长prompt不会自动产生独立验证集。
# 沿用上一格的autocast、**batch和decode，查看两条提示词的生成结果。

# 第二条更长prompt使S变化；相同tokenizer和同一model。
batch = tokenizer("“An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains.” ->: ", return_tensors='pt')

# 仍是CUDA autocast，不自动搬运输入。
with torch.cuda.amp.autocast():
    # 生成上限仍为50个新增token，未组成独立验证集。
    output_tokens = model.generate(**batch, max_new_tokens=50)

# decode结果为str，两个单例不构成准确率测量。
print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# %% [markdown]
# ## 单元 46：查看 collator
#
# 访问trainer.data_collator并显示其类型和配置。调用这个对象并传入样本后，它才会把样本整理成训练批次。
#
# 实际batch转换是调用该collator(samples)，需要样本列表；新验证通过真实collator检查padding labels=-100、shift与BLOOM返回loss一致。

# %%
# 查看Trainer持有的collator；训练时它负责把样本整理成batch。
# trainer.data_collator 是可调用批次构造对象；本格仅取属性并显示repr，返回该对象。
# 调用collator(samples)时，才会对样本做padding并建立labels。
trainer.data_collator
