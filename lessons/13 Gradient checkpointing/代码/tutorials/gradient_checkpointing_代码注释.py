# -*- coding: utf-8 -*-
# 第13讲Python阅读版；原CUDA/BERT-large设置保留。
# 真实CPU补充与执行改动见本讲复现README。

# %% [markdown]
# ## 单元 0（markdown）：基础概念入口
#
# 先区分激活checkpoint与磁盘模型checkpoint：前者保存计算边界并在backward重算，后者保存权重等训练状态。

# %% [markdown] 原单元 0
# ## basics

# %% [markdown]
# ## 单元 1（markdown）：论文、计算图与复杂度链接
#
# arXiv论文讨论分段重算；两个Medium链接是课程扩展阅读；HF v4.18性能文档是原链接，非精确环境锁。O(n)全保存和O(1)空间/O(n²)朴素重算都依赖简化链/均匀层假设，参数和优化器状态仍需单独保存。computation graph为计算依赖图，节点按需求恢复；未运行链接中的命令。

# %% [markdown] 原单元 1
# - https://arxiv.org/pdf/1604.06174.pdf
# - https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
#     - **computation graph examples**
#     - 节点在必要时才加载进来
#     - At the peak, the algorithm stores all activations,
#         - forward
#         - $O(n)$ memory requirement for network of depth $n$.
#     - recomputing them later.
#         - More generally, this “memory-poor” strategy needs $O(1)$ memory but requires $O(n^2)$ computation steps.
#     - https://yaroslavvb.medium.com/backprop-and-systolic-arrays-24e925d2050
# - https://huggingface.co/docs/transformers/v4.18.0/en/performance

# %% [markdown]
# ## 单元 2（markdown）：激活存储与计算折中
#
# intermediate activations即中间张量，backward求梯度需要其中部分值。第一次forward只保存边界，backward按需重算段内；增加计算来减少保存激活。不能仅靠del变量或empty_cache释放仍被autograd引用的张量，也不是冻结参数。

# %% [markdown] 原单元 2
# - 依然是显存占用优化算法
#     - 当然是 memory usage 与 computation time 之间的 tradeoff ；
# - In deep neural networks, backpropagation requires storing **intermediate activations** for computing gradients during the backward pass.
#     - 但是当层数变多时，存储所有的中间层的激活值（intermediate activations）非常地占用显存；
# - gradient checkpointing 选择性地重新计算（recompute）一部分的 intermediate activations 在反向传播过程中来缓解显存的压力；
#     - Instead of storing all activations (**during the forward pass**), only a **subset** of them, typically those necessary for computing gradients, are cached.
#     - The remaining intermediate activations are recomputed on-the-fly **during the backward pass**. By recomputing rather than storing all intermediate activations, memory usage is reduced at the cost of increased computation time.

# %% [markdown]
# ## 单元 3（markdown）：训练参数章节入口
#
# 后续Trainer把数据、模型、loss和optimizer连起来，gradient_checkpointing是模型开关，gradient_accumulation是更新节奏。

# %% [markdown] 原单元 3
# ## Trainer Arguments

# %% [markdown]
# ## 单元 4（code）：选择作者的单张GPU
#
# os.environ写进程可见CUDA设备，0指物理选择，内部编号重新从0起；应在CUDA初始化前设置。本次CPU测试跳过此格，不修改用户设备环境。不是量化/混合精度/自动并行开关。

# %% 原单元 4
# 选择作者的单张GPU；原语句保留，实际教学运行另记。
# 导入进程环境模块，不启动CUDA。
import os
# 改变CUDA可见设备；原作者单卡选择，本次验证跳过此格。
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %% [markdown]
# ## 单元 5（markdown）：随机数据入口
#
# 本例没有真实文本或tokenizer；随机ID与随机标签适合验证计算/资源流程，不证明分类能力。

# %% [markdown] 原单元 5
# ### dataset

# %% [markdown]
# ## 单元 6（code）：构造随机序列分类Dataset
#
# NumPy模块/数据集类导入；seq_len与dataset_size均512。randint上界不含30000或2，input_ids形状[512,512]，labels[512]，每序列一个0/1标签。括号(dataset_size)仍是整数，不是一元tuple。from_dict类方法把列转HF表，set_format('pt')改取样返回Torch Tensor，不搬GPU。原格完整执行核对；独立教学变体仅512/512→16/8，用相同随机种子。

# %% 原单元 6
# 构造随机序列分类Dataset；原语句保留，实际教学运行另记。
# 导入数值数组模块，np为模块别名。
import numpy as np
# 导入HF Dataset类，from_dict为类方法。
from datasets import Dataset


# 元组解包赋值：每行512token，共512样本。
seq_len, dataset_size = 512, 512
# 创建列式字典，后续两个数组第一维长度一致。
dummy_data = {
    # 产生[512,512]整数ID，上界30000不含在内；没有真实语言文本。
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    # 产生[512]的0/1序列标签；括号整数不是一元tuple。
    "labels": np.random.randint(0, 2, (dataset_size)),
}
# 类方法将两列构造为HF Dataset，返回表对象。
ds = Dataset.from_dict(dummy_data)
# 实例方法修改取样格式为Torch Tensor，仍在CPU，不训练模型。
ds.set_format("pt")

# %% [markdown]
# ## 单元 7（code）：查看两个NumPy列的shape
#
# input_ids.shape是[样本数,序列长]，labels.shape是[样本数]；序列分类labels不是语言模型[B,T]，不做next-token shift。print不创建新模型。

# %% 原单元 7
# 查看两个NumPy列的shape；原语句保留，实际教学运行另记。
# 查看原NumPy列shape，labels每序列一项，不是[B,T]语言模型监督。
print(dummy_data['input_ids'].shape, dummy_data['labels'].shape)

# %% [markdown]
# ## 单元 8（code）：定义NVML查询与训练摘要函数
#
# 星号import导入NVML公开名。def创建函数尚不调用；nvmlInit初始化驱动查询，GetHandleByIndex(0)取物理卡句柄，MemoryInfo.used是整卡当时占用。//整数除法与1024**2换算MiB，原print写MB不精确。f-string从result.metrics读运行时间和样本吞吐，不是准确率。NVML统计整张物理GPU当时的占用，包括其他进程；单次读数应按当前占用量理解。

# %% 原单元 8
# 定义NVML查询与训练摘要函数；原语句保留，实际教学运行另记。
# 星号导入驱动查询API，与Torch训练图独立。
from pynvml import *


# 定义函数，定义时不查询；调用后读取物理GPU整卡占用。
def print_gpu_utilization():
    # 初始化NVML驱动会话，不表示Torch已可用CUDA。
    nvmlInit()
    # 按物理索引取第一张卡句柄，未按训练进程过滤。
    handle = nvmlDeviceGetHandleByIndex(0)
    # 读取显存信息结构体，used包括其他进程/驱动占用。
    info = nvmlDeviceGetMemoryInfo(handle)
    # 整数除法换算MiB，原字符串MB保留；是当时占用而非峰值。
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# 定义接收TrainOutput的摘要函数，返回None。
def print_summary(result):
    # f-string读取秒数，:.2f保留两位小数，不是语言质量。
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    # 读取吞吐样本/秒；不能当准确率。
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # 调用已定义NVML函数，只查询此刻整卡占用。
    print_gpu_utilization()

# %% [markdown]
# ## 单元 9（code）：调用NVML一次
#
# 执行前一格定义的函数，实际物理设备占用随其他进程变化，不能要求原/注释读数相等或当benchmark。本次真实查询记录范围；不是stub打印0。

# %% 原单元 9
# 调用NVML一次；原语句保留，实际教学运行另记。
# 调用已定义NVML函数，只查询此刻整卡占用。
print_gpu_utilization()

# %% [markdown]
# ## 单元 10（markdown）：模型章节入口
#
# AutoModelForSequenceClassification按配置构造序列分类模型，输出[B,2]，与第12讲CausalLM不同。

# %% [markdown] 原单元 10
# ### model

# %% [markdown]
# ## 单元 11（code）：原BERT-large加载与CUDA移动
#
# from_pretrained加载基础checkpoint并构造分类头；原分类头未预训练而新初始化。to('cuda')实例方法移动参数/buffers并返回对象，随后的NVML只查当时整卡占用。原调用默认使用FP32。配套CPU算例用小BertConfig随机初始化模型，并将设备改为cpu，展示同一分类结构的前向与训练过程。

# %% 原单元 11
# 原BERT-large加载与CUDA移动；原语句保留，实际教学运行另记。
# 配置驱动的序列分类工厂类，默认输出两类logits。
from transformers import AutoModelForSequenceClassification
# 原BERT-large加载后移动CUDA；本次CPU随机小模型替换单独记录，不下载原权重。
model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased').to('cuda')
# 调用已定义NVML函数，只查询此刻整卡占用。
print_gpu_utilization()

# %% [markdown]
# ## 单元 12（markdown）：普通训练入口与原拼写
#
# 原标题checkpint拼写保留；下格按作者意图应与checkpoint方案分别运行，不串接训练同一个已更新模型作为公平对比。

# %% [markdown] 原单元 12
# ### training without checkpint

# %% [markdown]
# ## 单元 13（code）：导入Trainer并定义共享参数字典
#
# TrainingArguments配置类、Trainer训练器类、logging模块。字典default_args用**展开成关键字参数；output_dir是产物位置，evaluation_strategy=steps但没有eval_dataset是潜在失败；epoch1、log_level error、report_to none。本次整格保留执行，训练教学上下文明确更新临时路径/评估no/max_steps2/save_strategy no，负例另外实际触发无评估集错误。

# %% 原单元 13
# 导入Trainer并定义共享参数字典；原语句保留，实际教学运行另记。
# 训练配置类、训练器类及日志模块，导入本身不训练。
from transformers import TrainingArguments, Trainer, logging

# 创建共享关键字字典，后续**default_args展开参数。
default_args = {
    # 训练产物路径；CPU教学上下文改临时目录。
    "output_dir": "tmp",
    # 按步评估但没有eval_dataset，触发评估时会失败；补充真实负例。
    "evaluation_strategy": "steps",
    # 一轮计划；教学另设max_steps2限制真实优化步数。
    "num_train_epochs": 1,
    # 控制框架日志级别，不改变数学计算。
    "log_level": "error",
    # 关闭W&B等报告集成，避免隐式登录。
    "report_to": "none",
}

# %% [markdown]
# ## 单元 14（code）：原batch4普通训练流程
#
# 两条#开头行仅注释。构造TrainingArguments的batch4，原累积默认1、checkpoint默认False。Trainer拿model/ds，train真实forward/分类CE/backward/AdamW更新并返回TrainOutput，print_summary读metrics/NVML。本次该格整格执行，受控上下文使用随机小模型/8×16数据，2步。公平比较需要各组恢复相同初始模型，而非接着训练。

# %% 原单元 14
# 原batch4普通训练流程；原语句保留，实际教学运行另记。
# logging.set_verbosity_error()
# 跑 training with checkpint 时，需要把这段代码注释掉
# batch4、原累积1/checkpointFalse；**字典展开，不是新算法。
training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
# 构造真实Trainer，绑定模型、配置与Dataset。
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# 实例方法执行forward/loss/backward/累积/更新，返回TrainOutput。
result = trainer.train()
# 打印运行统计与NVML，不测Torch峰值。
print_summary(result)

# %% [markdown]
# ## 单元 15（markdown）：重算训练入口
#
# checkpoint只减少一部分保存激活，不降低全部参数/optimizer状态内存；下面还同时改micro-batch与累积。

# %% [markdown] 原单元 15
# ### training with checkpoint

# %% [markdown]
# ## 单元 16（code）：原batch1/累积4/checkpoint流程
#
# TrainingArguments per_device_batch1、gradient_accumulation_steps4、gradient_checkpointing=True，有效单卡batch仍4。Trainer在准备时启用模型开关，BertEncoder仅training分支重算。该格整格CPU教学执行；补充四组对照单独切开关分离因素。返回TrainOutput不是保存checkpoint路径。

# %% 原单元 16
# 原batch1/累积4/checkpoint流程；原语句保留，实际教学运行另记。
# 创建第二组训练配置，不立即开始forward。
training_args = TrainingArguments(
    # micro1×累积4有效batch4，checkpointTrue触发模型训练路径重算。
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)
# 构造真实Trainer，绑定模型、配置与Dataset。
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# 实例方法执行forward/loss/backward/累积/更新，返回TrainOutput。
result = trainer.train()
# 打印运行统计与NVML，不测Torch峰值。
print_summary(result)

# %% [markdown]
# ## 单元 17（code）：计算有效更新次数
#
# 原512/((1*1)*4)得到float128.0：512样本/(micro1×单进程1×累积4)，对应更新步数。第二组dataloader有512个micro-batch，不是128个；原注释steps(total batches)需辨认这个区别。教学8样本对应2更新，但不改原表达式。

# %% 原单元 17
# 计算有效更新次数；原语句保留，实际教学运行另记。
# steps(total batches)
# 除法返回float128.0，原512样本的optimizer更新数而非micro-batch数量。
512/((1*1) * 4)

# %% [markdown]
# ## 单元 18（markdown）：PyTorch接口章节入口
#
# 分段sequential API与BERT逐层checkpoint路径有关但不同；原材料没有完整sequential使用示例。

# %% [markdown] 原单元 18
# ## pytorch

# %% [markdown]
# ## 单元 19（code）：只导入checkpoint_sequential
#
# 这是torch.utils.checkpoint模块函数，原格没有调用。segments为段数，旧版前segments-1段重算、最后段普通执行；输入/输出单Tensor。独立教学补充继续展示函数调用、梯度和随机数状态的变化。

# %% 原单元 19
# 只导入checkpoint_sequential；原语句保留，实际教学运行另记。
# 只导入分段函数，原格没有调用；segments参数表示段数。
from torch.utils.checkpoint import checkpoint_sequential

# %% [markdown]
# ## 单元 20（markdown）：HF模型开关章节入口
#
# 追踪Trainer→模型启用方法→Module.apply→BertEncoder flag→checkpoint，而不只记配置名。

# %% [markdown] 原单元 20
# ## huggingface

# %% [markdown]
# ## 单元 21（markdown）：Markdown中的BERT开关类
#
# 原fence是讲解代码，不是notebook执行格；注释版仅在fence内插中文#。class定义继承PreTrainedModel的类；_set_gradient_checkpointing实例方法，self是实例、module是被遍历子模块，value默认False；isinstance判断BertEncoder后赋flag，其他模块不处理。方法返回None，只设置状态；实际重算在encoder.forward的training分支。两份片段AST及真实encoder开关测试另核对。
#
# 原Markdown中的源码（这里加入注释）：
#
# ```python
# # Markdown中的BERT开关类；原语句保留，实际教学运行另记。
# # class创建继承PreTrainedModel的类，此片段仅在原Markdown中。
# class BertPreTrainedModel(PreTrainedModel):
#     # 实例方法接收待检查module；value默认False，返回None。
#     def _set_gradient_checkpointing(self, module, value=False):
#         # 只对 BertEncoder 进行 checkpoint
#         # 内建函数检查对象类型，只处理BERT encoder。
#         if isinstance(module, BertEncoder):
#             # 修改encoder开关状态，不在这里执行重算。
#             module.gradient_checkpointing = value
#
# ```

# %% [markdown] 原21源码，避免误当执行类
# # Markdown中的BERT开关类；原语句保留，实际教学运行另记。
# # class创建继承PreTrainedModel的类，此片段仅在原Markdown中。
# class BertPreTrainedModel(PreTrainedModel):
#     # 实例方法接收待检查module；value默认False，返回None。
#     def _set_gradient_checkpointing(self, module, value=False):
#         # 只对 BertEncoder 进行 checkpoint
#         # 内建函数检查对象类型，只处理BERT encoder。
#         if isinstance(module, BertEncoder):
#             # 修改encoder开关状态，不在这里执行重算。
#             module.gradient_checkpointing = value

# %% [markdown]
# ## 单元 22（code）：原空代码格
#
# 原source空；保留编号/元数据/历史输出，仅注释版增加说明。没有臆造作者未保存的评估/保存代码。

# %% 原单元 22
# 原空代码格；原语句保留，实际教学运行另记。
