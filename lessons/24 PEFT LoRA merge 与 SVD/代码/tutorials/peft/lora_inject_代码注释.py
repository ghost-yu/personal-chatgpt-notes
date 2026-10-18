# -*- coding: utf-8 -*-
# 原课程代码中文注释阅读版；不是直接运行原GPU/远程权重的承诺。

# %% 原代码单元 1
# 导入inspect检查继承/源码，torch和nn构造张量/模块，inject_adapter_in_model直接改模型内部结构；导入本身不下载模型。
# 导入模块命名空间供后续使用。
import inspect
# 导入模块命名空间供后续使用。
import torch
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from torch import nn
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from peft import inject_adapter_in_model, LoraConfig

# %% 原代码单元 2
# DummyModel继承nn.Module。__init__注册embedding、linear、lm_head；super调用父初始化；forward定义调用流程。输入int64[B,L]→Embedding float32[B,L,1000]→Linear[B,L,1000]→lm_head[B,L,100]。没有激活、softmax或loss，不是训练完的语言模型。
# 定义nn.Module子类；类是结构模板，实例才持有实际参数。
class DummyModel(torch.nn.Module):
    # 构造方法，实例创建时注册子层。
    def __init__(self):
        # 先初始化父nn.Module内部注册结构。
        super().__init__()
        # 100词映射到1000维；权重[100,1000]。
        self.embedding = torch.nn.Embedding(100, 1000)
        # 目标层权重[1000,1000]、偏置[1000]。
        self.linear = torch.nn.Linear(1000, 1000)
        # 1000维投影为100个未归一化词logit。
        self.lm_head = torch.nn.Linear(1000, 100)

    # 实例被model(input_ids)调用时执行此方法。
    def forward(self, input_ids):
        # int64[B,L]查表得到float32[B,L,1000]。
        x = self.embedding(input_ids)
        # 在末维做线性变换，保留batch/序列轴。
        x = self.linear(x)
        # 末维1000变100，得到[B,L,100]。
        x = self.lm_head(x)
        # 返回最后张量，不额外softmax或计算loss。
        return x

# %% 原代码单元 3
# LoraConfig是配置对象。r64、alpha16令s=alpha/r=.25；dropout .1只在训练分支随机丢输入；bias none不额外训练偏置；仅名字linear匹配，不注入embedding和lm_head。A[64,1000]、B[1000,64]，新增128000参数。
# 创建配置，本行尚未替换模型内部层。
lora_config = LoraConfig(
    # 普通LoRA缩放分子alpha，与r相除得到支路系数。
    lora_alpha=16,
    # 训练时随机丢输入比例；eval时禁用，不能与density混淆。
    lora_dropout=0.1,
    # LoRA中间秩，决定A/B中间轴和新增参数量。
    r=64,
    # none表示不额外训练基础层bias。
    bias="none",
    # 指定注入目标名字，不能只按打印文本猜目标路径。
    target_modules=["linear"],
)

# %% 原代码单元 4
# 展示配置，不发生第二次注入。默认普通LoRA且非RSLoRA，不能套alpha/sqrt(r)。
lora_config

# %% 原代码单元 6
# 实例化三个随机层。nn.Module的注册让parameters/state_dict能递归查找子模块；此时没有预训练权重。
# 实例化三个随机基础层，不加载预训练权重。
model = DummyModel()
model

# %% 原代码单元 7
# type返回注入前的torch.nn.Linear类，不执行前向。
# 读取实际Python类，注意torch和PEFT的Linear是不同类。
type(model.linear)

# %% 原代码单元 8
# LongTensor是int64词ID；[1,8]→[1,8,100]。0–7均在100词表内，最后100维是未归一化logits；本格只看shape。
# 构造有效范围内的int64词ID张量[1,8]。
dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
# 触发Module.__call__→forward，只查看结果shape。
model(dummy_inputs).shape

# %% 原代码单元 10
# inject_adapter_in_model(config,model)在原模型上替换目标层，返回同一个DummyModel，不产生PeftModel外层。旧linear被保存在base_layer中，基础参数冻结，只有A/B可训练。新分支初始B0，所以初始输出不变。
# 在原对象内替换匹配层，返回同一模型实例。
model = inject_adapter_in_model(lora_config, model)
model

# %% 原代码单元 11
# 对比类型：embedding仍是Embedding，linear为PEFT的Linear；完整模块路径可区分同名类。
# 读取实际Python类，注意torch和PEFT的Linear是不同类。
type(model.embedding), type(model.linear)

# %% 原代码单元 12
# isinstance检查继承关系；LoRA包装仍然是nn.Module，可以参与注册、调用和autograd。
# 测试对象是否属于该类或其子类。
isinstance(model.linear, nn.Module)

# %% 原代码单元 13
# inspect.getmro返回方法解析顺序元组，解释多继承查找次序，不是模型执行顺序或层数。
# 读取类的方法解析顺序，不是神经网络计算顺序。
inspect.getmro(type(model.linear))

# %% 原代码单元 14
# forward??是IPython源码查询。核心为base_layer(x)+B(A(dropout(x)))*s；普通Python不能直接解析双问号，阅读版采用真实IPython转换。本次捕获pinfo2源码文本，不启动外部分页器。
# 原IPython双问号语句，以下是其Python转换。
# result = result + lora_B(lora_A(dropout(x))) * scaling
get_ipython().run_line_magic('pinfo2', 'model.linear.forward')


# %% 原代码单元 16
# get_peft_model_state_dict是函数，按模型的peft_config筛选适配器权重。
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from peft import get_peft_model_state_dict

# %% 原代码单元 17
# 返回字典含linear.lora_A.weight[64,1000]和linear.lora_B.weight[1000,64]，默认adapter名在导出键中去掉。没有embedding/base_linear/lm_head完整权重，低层API也不自动提供PeftModel.save_pretrained。
# 筛选adapter状态，不包含完整基础模型参数。
get_peft_model_state_dict(model)
