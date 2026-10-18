# -*- coding: utf-8 -*-
# 第11讲原代码纯文本阅读版；原语句保留，历史输出不加入Python。
# 代理/远程BERT/CUDA代码保留原样；从代码/tutorials运行以解析原图。
# Markdown伪代码以注释保留；可执行窗口修正见../窗口归一化_教学补充.py。

# %% [markdown]
# ## 单元 0（code）：导入实际使用的库
#
# numpy生成随机ID；Dataset来自HF datasets，原注释的torch.utils.data.Dataset不生效。torch支持autograd，transformers提供分类模型/TrainingArguments/Trainer及日志模块，Image展示图片。导入不下载BERT、不启动训练；Trainer还依赖accelerate。

# %% 原单元 0
# 导入numpy并命名np，用于生成整数随机数组。
import numpy as np
# from torch.utils.data import Dataset
# 这里是HF Dataset，不是torch.utils.data.Dataset；支持from_dict及列格式。
from datasets import Dataset
# 导入PyTorch，Tensor/自动求导与CUDA查询属于该模块。
import torch
# 自动序列分类模型接口，根据配置选择BERT分类实现。
from transformers import AutoModelForSequenceClassification
# 配置对象、实际训练器和transformers日志模块，不是Python标准logging。
from transformers import TrainingArguments, Trainer, logging
# 导入Jupyter图片显示类；导入本身不读取文件。
from IPython.display import Image

# %% [markdown]
# ## 单元 1（markdown）：基础原理入口
#
# 每个micro-batch执行一次forward/backward，凑满有效窗口后再执行一次optimizer更新。

# %% [markdown] 原单元 1
# ## basics

# %% [markdown]
# ## 单元 2（code）：展示累积示意图
#
# Image读取../imgs/grad-accumulation.png，width400为展示像素；要求内核工作目录为代码/tutorials。三个蓝色Batch箭头汇入绿色Weight update：累积的是梯度，不是图。图为示意，三批与后面K=4不矛盾。返回Image对象，无模型参数或loss。

# %% 原单元 2
# 读取相对工作目录的PNG，返回400像素宽的显示对象，不执行训练。
Image('../imgs/grad-accumulation.png', width=400)

# %% [markdown]
# ## 单元 3（markdown）：原课程的性能与收敛表述
#
# 实际提升的是有效更新batch，物理micro-batch不变。省的主要是同时存在的激活，不是全部参数/梯度/Adam状态。原文说小batch必定不稳定/更慢/精度低，条件过强；最终收敛还依赖目标、学习率、噪声与总更新数，见主解释第3节。

# %% [markdown] 原单元 3
# - 显存占用优化算法
#     - 可以在 `batch_size = 1` （或者是很小时）的时候，还能事实上提升真正的 `batch_size`
#         - `batch_size = 1` （或者很小时）会导致训练过程的不稳定，收敛更慢，精度更低；

# %% [markdown]
# ## 单元 4（markdown）：普通更新标题
#
# Vanilla指每个物理批次都更新一次，作为下一段累积循环的对照。

# %% [markdown] 原单元 4
# ### Vanilla

# %% [markdown]
# ## 单元 5（markdown）：Markdown中的普通训练循环
#
# 这段是Markdown伪代码，不会被Jupyter自动执行。data_loader产生(inputs,labels)，to(device)移动数据；with开启autograd，model调用forward，criterion返回标量loss；backward向.grad相加，step更新，zero_grad清空。model/criterion/optimizer/device未在块内定义。循环前也应明确清零；原循环每批更新。新增注释保留所有原代码行，真实执行用明确的CPU线性模型上下文，不声称原BERT运行。
#
# 原框架循环与新增中文注释：
#
# ```
# # loop through batches
# # 普通循环每批取inputs/labels；这里假定loader返回二元组。
# for (inputs, labels) in data_loader:
#
#     # extract inputs and labels
#     # 将输入移指定设备；不自动移动model。
#     inputs = inputs.to(device)
#     # 将目标移同设备；分类标签通常为int64[b]。
#     labels = labels.to(device)
#
#     # passes and weights update
#     # 上下文开启autograd，退出恢复；不替代model.train()。
#     with torch.set_grad_enabled(True):
#
#         # forward pass
#         # 调用模型forward，得到预测Tensor或模型输出；此示意需与criterion对齐。
#         preds = model(inputs)
#         # criterion将预测和标签归约成标量loss；默认mean需按窗口缩放。
#         loss  = criterion(preds, labels)
#
#         # backward pass
#         # 对当前micro图求导并加到参数.grad；通常释放这一次forward图。
#         loss.backward()
#
#         # weights update
#         # 按已有累计grad更新参数和优化器状态；只应在窗口边界执行。
#         optimizer.step()
#         # 清除本窗口grad；当前PyTorch默认set_to_none=True。
#         optimizer.zero_grad()
# ```

# %% [markdown] 原单元 5
# ```
# # loop through batches
# # 普通循环每批取inputs/labels；这里假定loader返回二元组。
# for (inputs, labels) in data_loader:
#
#     # extract inputs and labels
#     # 将输入移指定设备；不自动移动model。
#     inputs = inputs.to(device)
#     # 将目标移同设备；分类标签通常为int64[b]。
#     labels = labels.to(device)
#
#     # passes and weights update
#     # 上下文开启autograd，退出恢复；不替代model.train()。
#     with torch.set_grad_enabled(True):
#
#         # forward pass
#         # 调用模型forward，得到预测Tensor或模型输出；此示意需与criterion对齐。
#         preds = model(inputs)
#         # criterion将预测和标签归约成标量loss；默认mean需按窗口缩放。
#         loss  = criterion(preds, labels)
#
#         # backward pass
#         # 对当前micro图求导并加到参数.grad；通常释放这一次forward图。
#         loss.backward()
#
#         # weights update
#         # 按已有累计grad更新参数和优化器状态；只应在窗口边界执行。
#         optimizer.step()
#         # 清除本窗口grad；当前PyTorch默认set_to_none=True。
#         optimizer.zero_grad()
# ```

# %% [markdown]
# ## 单元 6（markdown）：累积循环标题
#
# accumulation指延后参数更新，在一个窗口内累加多个micro梯度。

# %% [markdown] 原单元 6
# ### accumulation

# %% [markdown]
# ## 单元 7（markdown）：Markdown中的累积循环与尾窗口
#
# K=4；enumerate给0起始索引，batch_idx+1为批次计数；每批loss/=K后backward，整4批或epoch最后批才step/zero_grad。窗口内参数不变，且每次重新forward，不需retain_graph。最后条件能避免漏step，但尾窗口不足4批仍除4，会低估梯度。不等大小batch还要按样本/有效token数归一化。原代码保留，修正版单独提供；原循环与修正版实际CPU执行见复现记录。
#
# 原框架循环与新增中文注释：
#
# ```
# # 指定完整累积窗口4批，不增加物理micro大小。
# gradient_accumulation_steps = 4
#
# # loop through batches
# # enumerate返回0起始索引与该批数据。
# for batch_idx, (inputs, labels) in enumerate(data_loader):
#
#     # extract inputs and labels
#     # 将输入移指定设备；不自动移动model。
#     inputs = inputs.to(device)
#     # 将目标移同设备；分类标签通常为int64[b]。
#     labels = labels.to(device)
#
#     # passes and weights update
#     # 上下文开启autograd，退出恢复；不替代model.train()。
#     with torch.set_grad_enabled(True):
#
#         # forward pass
#         # 调用模型forward，得到预测Tensor或模型输出；此示意需与criterion对齐。
#         preds = model(inputs)
#         # 每micro计算标量平均损失，随后进行窗口归一化。
#         loss = criterion(preds, labels)
#
#         # 完整等大小窗口除K求平均；短窗口/不等样本数需要另外加权。
#         loss /= gradient_accumulation_steps
#
#         # backward pass => compute gradient
#         # 对当前micro图求导并加到参数.grad；通常释放这一次forward图。
#         loss.backward()
#
#         # 整K批或最后一批时更新，防止尾梯度遗漏，但未修正尾窗口分母。
#         if ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx+1) == len(data_loader)):
#             # weights update
#             # 按已有累计grad更新参数和优化器状态；只应在窗口边界执行。
#             optimizer.step()
#             # 清除本窗口grad；当前PyTorch默认set_to_none=True。
#             optimizer.zero_grad()
# ```

# %% [markdown] 原单元 7
# ```
# # 指定完整累积窗口4批，不增加物理micro大小。
# gradient_accumulation_steps = 4
#
# # loop through batches
# # enumerate返回0起始索引与该批数据。
# for batch_idx, (inputs, labels) in enumerate(data_loader):
#
#     # extract inputs and labels
#     # 将输入移指定设备；不自动移动model。
#     inputs = inputs.to(device)
#     # 将目标移同设备；分类标签通常为int64[b]。
#     labels = labels.to(device)
#
#     # passes and weights update
#     # 上下文开启autograd，退出恢复；不替代model.train()。
#     with torch.set_grad_enabled(True):
#
#         # forward pass
#         # 调用模型forward，得到预测Tensor或模型输出；此示意需与criterion对齐。
#         preds = model(inputs)
#         # 每micro计算标量平均损失，随后进行窗口归一化。
#         loss = criterion(preds, labels)
#
#         # 完整等大小窗口除K求平均；短窗口/不等样本数需要另外加权。
#         loss /= gradient_accumulation_steps
#
#         # backward pass => compute gradient
#         # 对当前micro图求导并加到参数.grad；通常释放这一次forward图。
#         loss.backward()
#
#         # 整K批或最后一批时更新，防止尾梯度遗漏，但未修正尾窗口分母。
#         if ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx+1) == len(data_loader)):
#             # weights update
#             # 按已有累计grad更新参数和优化器状态；只应在窗口边界执行。
#             optimizer.step()
#             # 清除本窗口grad；当前PyTorch默认set_to_none=True。
#             optimizer.zero_grad()
# ```

# %% [markdown]
# ## 单元 8（code）：构建标量叶子与平方图
#
# torch.tensor(1.,requires_grad=True)创建默认float32标量[]，x为叶子；y=x**2为新的计算图节点，导数2x。y不是参数更新，不会自动给x.grad赋值；需要backward。

# %% 原单元 8
# 构造float32标量叶子x，shape[]；请求记录梯度，初始grad为None。
x = torch.tensor(1., requires_grad=True)
# 平方建立本次计算图；dy/dx=2x，x=1时贡献为2。
y = x**2

# %% [markdown]
# ## 单元 9（code）：第一次backward并保留图
#
# Tensor.backward方法沿y的图求导，标量不必传gradient；retain_graph=True保留中间图供同一y第二遍使用。返回None，副作用是x.grad=2；最后表达式在Jupyter显示Tensor。真实执行确认shape[]/float32/value2。

# %% 原单元 9
# 求导累加到叶子x.grad，并保留此forward图供下一格再反向。
y.backward(retain_graph=True)
# 读取叶子累计梯度Tensor，在notebook显示；不是返回backward的值。
x.grad

# %% [markdown]
# ## 单元 10（code）：第二次backward累加.grad
#
# 没清零，第二次贡献2累加为4；不加retain_graph时这次通常释放图中保存量。再对同一y反向会报已释放图错误。真实累积通常重新forward每个micro，所以不必保留旧图。

# %% 原单元 10
# 再反向贡献2，加上已有2得到4；本次不再保留图。
y.backward()
# 读取叶子累计梯度Tensor，在notebook显示；不是返回backward的值。
x.grad

# %% [markdown]
# ## 单元 11（markdown）：BERT演示入口
#
# 下面比较随机二分类演示的物理batch4与batch1×累积4；不是自然语言任务评测。

# %% [markdown] 原单元 11
# ## demo

# %% [markdown]
# ## 单元 12（code）：定义真实NVML占用与训练摘要函数
#
# 原注释!pip install nvidia-ml-py3不会执行；pynvml是模块名，实际补充安装nvidia-ml-py12.560.30。星号导入把模块公开API放入当前名字空间。print_gpu_utilization调用nvmlInit、遍历torch可见CUDA设备、查info.used并求和；//1024**2把bytes取整成MiB，原打印写MB。函数无参数/返回None，只打印，未采集峰值也未限定本进程。print_summary接收TrainOutput，访问metrics字典后调用占用函数；f字符串.2f只格式化显示。本次CPU torch可见设备0，但真实NVML能见物理卡，原函数会误导性打印0，需单独说明。

# %% 原单元 12
# !pip install nvidia-ml-py3
# 星号导入NVML驱动查询API；需要pynvml模块与真实NVIDIA驱动。
from pynvml import *

# 定义无参数打印函数，返回None；此时不调用驱动。
def print_gpu_utilization():
    # 初始化NVML驱动查询接口；可能因驱动/库缺失报错。
    nvmlInit()
    # 用Python整数累计所有torch可见CUDA设备的used字节数。
    total_used = 0
    # torch可见卡数决定循环；CPU构建为0，不等于物理卡不存在。
    for i in range(torch.cuda.device_count()):
        # 按NVML索引取设备句柄；复杂可见设备映射还需额外核对。
        handle = nvmlDeviceGetHandleByIndex(i)
        # 查询真实设备总显存使用，含其他进程，不是本模型张量峰值。
        info = nvmlDeviceGetMemoryInfo(handle)
        # 将该设备used bytes累加到total_used。
        total_used += info.used
    # 向下取整换算MiB，原字符串写MB；训练后时点值不是peak。
    print(f"GPU memory occupied: {total_used//1024**2} MB.")

# 接收Trainer返回的TrainOutput，通过.metrics字典读取计时/吞吐。
def print_summary(result):
    # 读取秒数并格式化两位小数，显示不改变原指标。
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    # 读取训练吞吐并打印；小随机CPU运行不可对比原GPU速度。
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # 实际调用占用函数；依赖其定义及NVML，未限定当前进程。
    print_gpu_utilization()

# %% [markdown]
# ## 单元 13（code）：训练前调用GPU占用函数
#
# 调用上格定义的函数；依赖驱动/NVML可用。原历史输出是讲师记录，不是本次结果。CPU构建中循环0次，即使NVML检测到卡也不能据0MB认定物理GPU空闲；本次单独对比两种设备计数。

# %% 原单元 13
# 实际调用占用函数；依赖其定义及NVML，未限定当前进程。
print_gpu_utilization()

# %% [markdown]
# ## 单元 14（markdown）：随机数据标题
#
# 演示用合成token ID和标签，不涉及真实文本下载、tokenizer或准确率数据集。

# %% [markdown] 原单元 14
# ### dataset

# %% [markdown]
# ## 单元 15（code）：生成随机ID与HF Dataset
#
# num_seqs1024为样本数，seq_len512为长度。randint的high不包含：ID100–29999、标签0/1；(num_seqs)为int而非tuple但size允许整数。input_ids为二维numpy整数数组，labels为一维。Dataset.from_dict按列创建HF表；set_format(pt)修改取出格式，取单行时IDs为torch.int64[512]、label为标量；不训练/不移动GPU。原格1024×512实际执行，CPU Trainer补充另把常量改为8×16。

# %% 原单元 15
# 原演示样本数1024；与每条序列长度分开。
num_seqs = 1024
# 每条随机ID序列512位置；不是batch或词表大小。
seq_len = 512
# 创建按列组织的dict，用于构建HF Dataset。
dummy_data = {
    # 生成二维整数数组[N,S]；上界30000不包含，没有真实tokenizer。
    'input_ids': np.random.randint(100, 30000, (num_seqs, seq_len)),
    # 生成整数分类标签[N]；(num_seqs)只是int，不是tuple。
    'labels': np.random.randint(0, 2, (num_seqs))
}
# 类接口按列建表，返回新的HF Dataset，未进行训练。
ds = Dataset.from_dict(dummy_data)
# 方法修改取出列的格式为PyTorch Tensor；数据仍在CPU。
ds.set_format('pt')

# %% [markdown]
# ## 单元 16（code）：检查标签取值集合
#
# ds[labels]先取全部列，pt格式返回int64[1024]；.numpy()得到CPU数组，set去重为{0,1}。这里用set查看标签类别；大语料可以分批统计，减少一次读入整列的内存开销。

# %% 原单元 16
# 取整列CPU标签→numpy数组→去重set；不计算准确率。
set(ds['labels'].numpy())

# %% [markdown]
# ## 单元 17（markdown）：加载模型标题
#
# AutoModel选择序列分类架构；与只加载tokenizer或基础BertModel不同。

# %% [markdown] 原单元 17
# ### load model

# %% [markdown]
# ## 单元 18（code）：设置进程HTTP代理
#
# os.environ是当前进程环境映射；两项赋值影响后续网络请求，127.0.0.1:7890必须有真实代理服务。它不分配GPU或修改模型。本次跳过，未改用户/系统代理；课程原语句仍保留。

# %% 原单元 18
# 导入进程环境操作模块。
import os
# 为当前进程后续HTTP请求指定本机代理，本次跳过该格。
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# 为当前进程后续HTTPS请求指定代理，需实际端口服务。
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# %% [markdown]
# ## 单元 19（code）：远程BERT-large加载并移到CUDA
#
# AutoModelForSequenceClassification.from_pretrained类接口可能下载配置/权重，返回分类模型；新分类头可随机初始化。to(cuda)原地移动参数/缓冲并返回自身。原模型hidden1024、序列512、分类2：IDs[b,512]→hidden[b,512,1024]→logits[b,2]；分类标签[b]产生标量交叉熵。原GPU/远程加载未运行；CPU教学另从BertConfig建小随机模型，不替换公开原格。

# %% 原单元 19
# 远程加载BERT-large分类架构/权重再移CUDA；本次仅在明确CPU教学变体中替换此表达式。
model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased').to('cuda')

# %% [markdown]
# ## 单元 20（code）：模型加载后占用读数
#
# 同一NVML函数的时点观测，非峰值或本进程精确测量；受他进程/allocator/旧对象影响。本次独立执行但没有原BERT权重，CPU构建可见CUDA数量0。

# %% 原单元 20
# 实际调用占用函数；依赖其定义及NVML，未限定当前进程。
print_gpu_utilization()

# %% [markdown]
# ## 单元 21（code）：原注释代码格
#
# source仅# model，没有执行AST语句；历史输出/元数据保留。不取消注释或擅自把它变成模型打印代码。

# %% 原单元 21
# 原格只有注释，没有执行语句；保留原编号与历史输出，不取消注释。
# model

# %% [markdown]
# ## 单元 22（markdown）：训练参数与Trainer标题
#
# 配置对象管理batch、epoch与评估；Trainer构造器保存model/args/train_dataset，train才进入loop。

# %% [markdown] 原单元 22
# ### training args & trainer

# %% [markdown]
# ## 单元 23（code）：物理batch4的Trainer基线
#
# logging属于transformers，只影响日志。default_args字典由**展开关键字：output_dir=tmp，evaluation_strategy=steps，epoch1，log_level=error，report_to=none。TrainingArguments本格batch4，累积默认1；默认learning_rate5e-5、AdamW、线性scheduler及seed42为实测4.44.2接口参照。未提供eval_dataset；默认评估触发间隔约500更新，短实验未触发，不能据此称配置适于长期运行。Trainer.train返回TrainOutput(global_step,training_loss,metrics)，实际更新/裁剪/调度在Trainer内；print_summary只打印结果。CPU补充以同数据/初始化的小BERT真实执行整个原格，性能不可与历史GPU比较。

# %% 原单元 23
# 只改变transformers日志级别，不改变训练梯度或batch。
logging.set_verbosity_error()
# 共享TrainingArguments配置字典，后面**展开为关键字。
default_args = {
    # 训练文件输出目录，原为相对cwd的tmp；教学运行在临时cwd。
    "output_dir": "tmp",
    # 按更新步评估；原没有eval_dataset，达到触发间隔可能失败。
    "evaluation_strategy": "steps",
    # 遍历数据一遍，不等于仅执行一个optimizer更新。
    "num_train_epochs": 1,
    # 配置日志等级error，减少显示。
    "log_level": "error",
    # 关闭默认外部实验报告集成，不发送训练日志到外部服务。
    "report_to": "none",
}
# 基线物理batch4，累积默认1；**把dict展开为命名参数。
training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
# 构造真实Trainer，保存模型与数据/配置；尚未执行训练loop。
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# 执行训练，内部forward/backward/累积与边界更新；返回TrainOutput。
result = trainer.train()
# 打印本次TrainOutput摘要及设备占用，不重新训练。
print_summary(result)

# %% [markdown]
# ## 单元 24（markdown）：Markdown保存的历史基线指标
#
# 这段不是可运行Python，是历史train_runtime38.0194s、samples/s26.934、steps/s3.367、loss0.7216147780、epoch1，以及占用21296MB文本。原NVML换算实际MiB、非peak；没有eval准确率。原文与输出原样保留；本次小BERT数据不填回历史格。

# %% [markdown] 原单元 24
# ```
# {'train_runtime': 38.0194, 'train_samples_per_second': 26.934, 'train_steps_per_second': 3.367, 'train_loss': 0.7216147780418396, 'epoch': 1.0}
# Time: 38.02
# Samples/second: 26.93
# GPU memory occupied: 21296 MB.
# ```

# %% [markdown]
# ## 单元 25（markdown）：累积设置对照标题
#
# 接下来重建数据/模型；原未控制随机seed，所以不构成严格同数据同初始化实验。

# %% [markdown] 原单元 25
# #### gradient accumulation

# %% [markdown]
# ## 单元 26（code）：完整重建与batch1累积4训练
#
# 重复导入、NVML函数、随机数据与BERT加载，再设置物理batch1/K4训练。重复定义会覆盖当前函数名，新的随机数据/模型对象取代旧引用，但旧Trainer仍可能持有旧对象。有效batch等于基线只在同设备并行设置下成立；不自动保证训练逐位等价。原整格CUDA/远程BERT未运行；CPU派生实验只明确替换8/16两个数据常量与模型加载表达式，保留Trainer和NVML原语句，真实train完成2次更新。

# %% 原单元 26
# 导入numpy并命名np，用于生成整数随机数组。
import numpy as np
# from torch.utils.data import Dataset
# 这里是HF Dataset，不是torch.utils.data.Dataset；支持from_dict及列格式。
from datasets import Dataset
# 导入PyTorch，Tensor/自动求导与CUDA查询属于该模块。
import torch
# 自动序列分类模型接口，根据配置选择BERT分类实现。
from transformers import AutoModelForSequenceClassification
# 配置对象、实际训练器和transformers日志模块，不是Python标准logging。
from transformers import TrainingArguments, Trainer, logging
# 导入Jupyter图片显示类；导入本身不读取文件。
from IPython.display import Image


# 星号导入NVML驱动查询API；需要pynvml模块与真实NVIDIA驱动。
from pynvml import *

# 定义无参数打印函数，返回None；此时不调用驱动。
def print_gpu_utilization():
    # 初始化NVML驱动查询接口；可能因驱动/库缺失报错。
    nvmlInit()
    # 用Python整数累计所有torch可见CUDA设备的used字节数。
    total_used = 0
    # torch可见卡数决定循环；CPU构建为0，不等于物理卡不存在。
    for i in range(torch.cuda.device_count()):
        # 按NVML索引取设备句柄；复杂可见设备映射还需额外核对。
        handle = nvmlDeviceGetHandleByIndex(i)
        # 查询真实设备总显存使用，含其他进程，不是本模型张量峰值。
        info = nvmlDeviceGetMemoryInfo(handle)
        # 将该设备used bytes累加到total_used。
        total_used += info.used
    # 向下取整换算MiB，原字符串写MB；训练后时点值不是peak。
    print(f"GPU memory occupied: {total_used//1024**2} MB.")

# 接收Trainer返回的TrainOutput，通过.metrics字典读取计时/吞吐。
def print_summary(result):
    # 读取秒数并格式化两位小数，显示不改变原指标。
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    # 读取训练吞吐并打印；小随机CPU运行不可对比原GPU速度。
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # 实际调用占用函数；依赖其定义及NVML，未限定当前进程。
    print_gpu_utilization()

# 原演示样本数1024；与每条序列长度分开。
num_seqs = 1024
# 每条随机ID序列512位置；不是batch或词表大小。
seq_len = 512
# 创建按列组织的dict，用于构建HF Dataset。
dummy_data = {
    # 生成二维整数数组[N,S]；上界30000不包含，没有真实tokenizer。
    'input_ids': np.random.randint(100, 30000, (num_seqs, seq_len)),
    # 生成整数分类标签[N]；(num_seqs)只是int，不是tuple。
    'labels': np.random.randint(0, 2, (num_seqs))
}
# 类接口按列建表，返回新的HF Dataset，未进行训练。
ds = Dataset.from_dict(dummy_data)
# 方法修改取出列的格式为PyTorch Tensor；数据仍在CPU。
ds.set_format('pt')

# 远程加载BERT-large分类架构/权重再移CUDA；本次仅在明确CPU教学变体中替换此表达式。
model = AutoModelForSequenceClassification.from_pretrained('bert-large-uncased').to('cuda')

# 只改变transformers日志级别，不改变训练梯度或batch。
logging.set_verbosity_error()
# 共享TrainingArguments配置字典，后面**展开为关键字。
default_args = {
    # 训练文件输出目录，原为相对cwd的tmp；教学运行在临时cwd。
    "output_dir": "tmp",
    # 按更新步评估；原没有eval_dataset，达到触发间隔可能失败。
    "evaluation_strategy": "steps",
    # 遍历数据一遍，不等于仅执行一个optimizer更新。
    "num_train_epochs": 1,
    # 配置日志等级error，减少显示。
    "log_level": "error",
    # 关闭默认外部实验报告集成，不发送训练日志到外部服务。
    "report_to": "none",
}
# 物理batch1、累积4；相同数据并行设备数下有效batch与基线相同。
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
# 构造真实Trainer，保存模型与数据/配置；尚未执行训练loop。
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
# 执行训练，内部forward/backward/累积与边界更新；返回TrainOutput。
result = trainer.train()
# 打印本次TrainOutput摘要及设备占用，不重新训练。
print_summary(result)

# %% [markdown]
# ## 单元 27（markdown）：Markdown保存的历史累积指标
#
# 历史102.8176s、9.959samples/s、1.245steps/s、loss0.7096406817、13166MB；占用较基线少8130MiB/38.18%，时间2.70倍、吞吐下降63.02%。随机数据与缺验证集，不支持准确率提升；该观测非原论文训练或本次benchmark。

# %% [markdown] 原单元 27
# ```
# {'train_runtime': 102.8176, 'train_samples_per_second': 9.959, 'train_steps_per_second': 1.245, 'train_loss': 0.7096406817436218, 'epoch': 1.0}
# Time: 102.82
# Samples/second: 9.96
# GPU memory occupied: 13166 MB.
# ```
