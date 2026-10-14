# -*- coding: utf-8 -*-
# 第17讲原代码阅读版；未改变语句，裸表达式在脚本中不会自动显示。

# %% 0
# ## 单元 0（markdown）：基础资料章节
#
# 【建议会】标题用于组织材料，不执行模型。
# ## basics

# %% 1
# ## 单元 1（markdown）：来源与术语
#
# 【建议会】原链接可能指向变化的main/stable；固定课程与Meta版本见来源清单。Swish在β=1时等于SiLU；SwiGLU还需要门控双分支和输出投影。Medium是补充材料，核心公式以原论文和官方源码为准。
# - llama2: https://github.com/facebookresearch/llama
# - rmsnorm: https://github.com/bzhangGo/rmsnorm
#     - layernorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
# - Swish: SwiLU, SiLU
#     - https://arxiv.org/abs/1710.05941v1
#     - https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html
# - 参考：
#     - https://akgeni.medium.com/llama-concepts-explained-summary-a87f0bd61964

# %% 2
# ## 单元 2（markdown）：归一化章节
#
# 【必须会】关注最后通道维度的统计，不能将batch或序列混合求均值。
# ## RMSNorm

# %% 3
# ## 单元 3（markdown）：pre-norm、公式与类预览
#
# 【必须会】a为一个token的n维向量，g为可学习通道增益。RMS不减均值，norm位于attention/FFN之前。原文√n结论只在非零输入、epsilon=0且g全1时成立，不能无条件套用。这里的Markdown类和单元11重复，下面注释同样覆盖它；Markdown展示不执行。原文关于省时间是背景结论，本机没有测速证据。
# - Pre-normalization Using RMSNorm  llama
#     - root mean square norm
#     - before attn
#     - before ffn
#     - 相比较 layernorm（re-centering，re-scaling），优化了计算时间；
#         - rms norm 只关注 re-scaling
#
#         $$ \begin{align} \begin{split} & \bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} \odot g_i, \quad \text{where}~~ \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}. \end{split}\nonumber \end{align} $$
#
#     - 我们自然地可以得到一个结论就是，经过 rms norm 之后的 $\bar{\mathbf{a}}$ 其 $\ell_2$ norm 为 $\sqrt n$
#
# $$
# \|\bar{\mathbf{a}}\|_2=\sqrt{\frac{\sum a_i^2}{\frac1n\sum a_i^2}}=\sqrt{n}
# $$
#
# ```
# # 继承Module以注册参数、使用forward调用和autograd。
# class RMSNorm(torch.nn.Module):
#     # 构造方法：dim为通道数，eps默认1e-6；类型提示不强制验证输入shape。
#     def __init__(self, dim: int, eps: float = 1e-6):
#         # 初始化父类Module，之后Parameter赋值才会正确注册。
#         super().__init__()
#         # 保存稳定项配置，不是优化器参数。
#         self.eps = eps
#         # 注册可训练通道增益[dim]，初始全1；没有beta偏置。
#         self.weight = nn.Parameter(torch.ones(dim))
#
#     # 内部方法：沿最后通道轴计算均方根，不减均值。
#     def _norm(self, x):
#         # 平方→通道均值[B,T,1]→加eps→倒数平方根→广播乘x[B,T,d]。
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#
#     # 模块调用的前向方法，返回同shape，参数更新不在这里发生。
#     def forward(self, x):
#         # 内部用FP32统计；归一化中间结果再转回输入dtype。
#         output = self._norm(x.float()).type_as(x)
#         # 逐通道乘共享增益；最终dtype还取决于weight与类型提升。
#         return output * self.weight
# ```

# %% 4
# ## 单元 4（code）：导入数组与张量框架
#
# 【必须会】np是NumPy模块别名；torch提供Tensor/随机数/自动求导；nn提供Module、Parameter与LayerNorm。导入不产生训练数据。
# NumPy模块用于sqrt、数组和手写激活；别名np。
import numpy as np
# Torch模块提供Tensor、随机数、rsqrt和自动求导。
import torch
# 导入神经网络子模块，后面使用LayerNorm和Parameter。
from torch import nn

# %% 5
# ## 单元 5（code）：构造隐藏状态
#
# 【必须会】多变量赋值给bs=20、seq_len=5、embedding_dim=10；torch.randn生成标准正态浮点Tensor[20,5,10]，三轴为batch、位置、通道。原格未设随机种子，所以新结果不能要求等于历史数值。
# 同一语句解包三个整数：batch、序列位置数、隐藏通道数。
bs, seq_len, embedding_dim = 20, 5, 10
# 标准正态Tensor[20,5,10]，原格未固定seed。
x = torch.randn(bs, seq_len, embedding_dim)

# %% 6
# ## 单元 6（markdown）：LayerNorm小节
#
# 【必须会】先减每个token通道均值，再按总体方差缩放。
# ### layer norm

# %% 7
# ## 单元 7（code）：创建LayerNorm实例
#
# 【必须会】nn.LayerNorm(10)只处理最后10维，默认eps=1e-5；weight[10]全1、bias[10]全0均可训练。ln是模块对象，不是已经归一化的数据。
# 构造只归一化最后10维的模块；eps默认1e-5，带weight/bias。
ln = nn.LayerNorm(embedding_dim)

# %% 8
# ## 单元 8（code）：调用LayerNorm
#
# 【必须会】ln(x)通过nn.Module调用forward，返回x_ln[20,5,10]，不覆盖x，不更新参数。
# Module调用forward：[20,5,10]→[20,5,10]，不覆盖x。
x_ln = ln(x)

# %% 9
# ## 单元 9（code）：检查两个token的统计
#
# 【必须会】[1,3,:]与[1,2,:]是两个不同token的[10]向量。mean()返回0维均值；std(unbiased=False)使用分母10而不是9。epsilon使标准差略小于1；学习后的增益和偏置不保证均值0/标准差1。print只展示数值。
# 取样本1/位置3的10通道均值；结果0维，初始化时约0。
print(x_ln[1, 3, :].mean())
# 位置2是另一个token；总体标准差分母10，加eps后不严格为1。
print(x_ln[1, 2, :].std(unbiased=False))

# %% 10
# ## 单元 10（markdown）：RMSNorm小节
#
# 【必须会】使用均方根而不是中心化标准差。
# ### RMSNorm

# %% 11
# ## 单元 11（code）：实现RMSNorm类
#
# 【必须会】class继承torch.nn.Module，__init__构造实例；self代表当前实例，super初始化基类。dim:int、eps:float=1e-6是类型提示/默认值，不是运行时shape检查。Parameter将ones[dim]注册成可训练weight；eps是常量。_norm中平方[ B,T,d ]→最后轴mean[ B,T,1 ]→加eps→rsqrt→广播乘x。forward先float32统计，再type_as回输入dtype，最后与weight相乘；最终dtype还受weight类型提升影响。返回同shape，不在forward更新参数；梯度和optimizer更新是后续步骤。
# 继承Module以注册参数、使用forward调用和autograd。
class RMSNorm(torch.nn.Module):
    # 构造方法：dim为通道数，eps默认1e-6；类型提示不强制验证输入shape。
    def __init__(self, dim: int, eps: float = 1e-6):
        # 初始化父类Module，之后Parameter赋值才会正确注册。
        super().__init__()
        # 保存稳定项配置，不是优化器参数。
        self.eps = eps
        # 注册可训练通道增益[dim]，初始全1；没有beta偏置。
        self.weight = nn.Parameter(torch.ones(dim))

    # 内部方法：沿最后通道轴计算均方根，不减均值。
    def _norm(self, x):
        # 平方→通道均值[B,T,1]→加eps→倒数平方根→广播乘x[B,T,d]。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 模块调用的前向方法，返回同shape，参数更新不在这里发生。
    def forward(self, x):
        # 内部用FP32统计；归一化中间结果再转回输入dtype。
        output = self._norm(x.float()).type_as(x)
        # 逐通道乘共享增益；最终dtype还取决于weight与类型提升。
        return output * self.weight

# %% 12
# ## 单元 12（code）：创建RMSNorm实例
#
# 【必须会】dim=10，eps默认1e-6，只有10个weight参数，无bias或running统计。模块train/eval采用同一公式。
# 创建dim10/eps1e-6实例，只有10个weight参数。
rms_norm = RMSNorm(embedding_dim)

# %% 13
# ## 单元 13（code）：调用RMSNorm
#
# 【必须会】x_rms[20,5,10]与x同shape，每token有独立均方[20,5,1]，所有token共享通道增益[10]。
# 真实RMSNorm forward：[20,5,10]，每个token独立统计。
x_rms = rms_norm(x)

# %% 14
# ## 单元 14（code）：查看一个原始token
#
# 【必须会】[0,0,:]索引第一个样本/位置，返回[10]视图；Notebook末尾表达式会显示，普通Python脚本不会自动显示。
# 查看第一个token的10维原始向量，Notebook会显示末表达式。
x[0, 0, :]

# %% 15
# ## 单元 15（code）：查看对应归一化token
#
# 【必须会】同一索引取x_rms[10]，幅度统一调整但不减均值；不能由显示精度判断公式严格相等。
# 查看同一token归一化后的10维向量。
x_rms[0, 0, :]

# %% 16
# ## 单元 16（code）：不带epsilon的手算
#
# 【必须会】pow(2)逐元素平方；torch.sum全部10项→0维，除embedding_dim=10求均方，再sqrt开方，广播除回[10]。此格省略eps，非零普通输入近似匹配RMSNorm，零输入会0/0，近零时与带eps版本明显不同。
# 手算x/RMS，省略eps；零输入会0/0，不能声称与类严格相等。
x[0, 0, :] / torch.sqrt(torch.sum(x[0, 0, :].pow(2)) / embedding_dim)

# %% 17
# ## 单元 17（code）：查看输出L2长度
#
# 【必须会】torch.norm对该[10]向量给0维欧氏长度。eps>0、weight初值1时通常略小于√10；学习weight后不保证此长度。
# 欧氏L2长度标量；eps>0或学习增益时不保证√10。
torch.norm(x_rms[0, 0, :])

# %% 18
# ## 单元 18（code）：比较理论√d
#
# 【必须会】np.sqrt(10)返回NumPy浮点标量，约3.162277660；这是理想条件比较值，不是模型参数或第二个归一化操作。
# 理想g1/eps0/非零输入的理论长度√10。
np.sqrt(embedding_dim)

# %% 19
# ## 单元 19（markdown）：SwiGLU章节
#
# 【必须会】标题虽为SwiGLU，后面课程代码只演示激活SiLU，没有完整三矩阵FFN；官方完整模块另在复现补充验证。
# ## SwiGLU

# %% 20
# ## 单元 20（markdown）：SiLU公式与命名纠正
#
# 【必须会】σ(z)=1/(1+exp(-z))，SiLU(z)=zσ(z)，任意shape逐元素运算。原文“也叫SiLU”不能作为SwiGLU同义解释：SwiGLU=SiLU(XW1ᵀ)⊙XW3ᵀ再乘W2ᵀ。β=1的Swish与SiLU相同，门控结果并非概率。
# - Inspiration of using SwiGLU in LLaMA is taken from PaLM.
# - 也叫 SiLU (Sigmoid Linear Unit)，
#
# $$
# \text{silu}(x)=x\cdot\sigma(x)=x\cdot\frac{1}{1+\exp(-x)}=\frac{x}{1+\exp(-x)}
# $$

# %% 21
# ## 单元 21（code）：NumPy sigmoid与swish
#
# 【必须会】def只定义普通函数，return返回结果；np.exp逐元素指数，两个函数保留输入数组shape。swish调用sigmoid再逐元素乘x。这里没有训练参数或β，极大负输入的朴素exp可能溢出。
# 定义普通函数：标量或NumPy数组到同shape sigmoid。
def sigmoid(x):
    # 逐元素1/(1+exp(-x))；极大负数时朴素exp可溢出。
    return  1/(1 + np.exp(-x))

# 定义β1 Swish，也就是这里的SiLU激活，而非完整SwiGLU。
def swish(x):
    # 逐元素x乘sigmoid(x)，无矩阵或可学习参数。
    return x*sigmoid(x)

# %% 22
# ## 单元 22（code）：准备绘图
#
# 【建议会】plt是pyplot别名；rcParams是全局绘图配置，figure.dpi=120设置显示/保存密度，不改变数据。验证采用Agg后端以便无GUI保存。
# 导入绘图接口别名plt，不执行模型。
import matplotlib.pyplot as plt
# 修改全局图像dpi配置，不改变曲线数值。
plt.rcParams['figure.dpi'] = 120

# %% 23
# ## 单元 23（code）：绘制SiLU曲线并改变x类型
#
# 【必须会】np.arange(-5,5,.01)生成约1000个float64值，右端5不包含。x从先前Torch三维Tensor被重新赋成NumPy[1000]；plt.plot返回Line2D列表并修改当前图。横轴输入、纵轴激活值，负侧下凹不是loss曲线或性能实验。
# x重新赋成float64 NumPy[1000]；右端5不包含。
x = np.arange(-5, 5, .01)
# 横轴输入，纵轴SiLU；返回Line2D列表，更新当前图。
plt.plot(x, swish(x))

# %% 24
# ## 单元 24（code）：再次改变x类型
#
# 【必须会】torch.randn(5)重新赋值，x变成float32 Tensor[5]，不再是绘图数组。独立抽取5个随机数，Notebook末表达式显示。
# x再次赋成float32 Tensor[5]，不再是绘图数组。
x = torch.randn(5)
# Notebook显示x；普通.py的裸表达式不会自动打印。
x

# %% 25
# ## 单元 25（code）：Torch手算SiLU
#
# 【必须会】exp(-x)、加1、逐元素除法都保留[5]；不修改x或产生可学习参数。此处没有三矩阵门控。
# 手写Torch SiLU，逐元素运算保持[5]且不覆盖x。
x/(1+torch.exp(-x))

# %% 26
# ## 单元 26（code）：导入functional接口
#
# 【建议会】F是torch.nn.functional模块别名；F.silu是函数而不是需实例化的nn.Module。
# functional模块别名F；silu是函数。
import torch.nn.functional as F

# %% 27
# ## 单元 27（code）：使用官方SiLU函数
#
# 【必须会】F.silu(x,inplace=False默认)返回[5]，不原地修改x；数值/梯度与β=1手算对应。训练和推理无不同分支，没有持久参数。
# 默认inplace=False，返回[5]激活；完整SwiGLU还需要三投影。
F.silu(x)
