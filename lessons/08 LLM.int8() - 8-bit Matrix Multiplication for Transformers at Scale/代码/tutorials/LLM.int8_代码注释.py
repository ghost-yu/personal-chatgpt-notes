# -*- coding: utf-8 -*-
# 第08讲 LLM.int8 Python阅读版；原执行语句未修改。
# 按# %%分格阅读；最后OPT6.7B段可能下载权重，原代理格会修改环境。
# 实际CPU验证范围和跳过单元见复现/README.md。

# %% [markdown]
# ## 单元 0（code）：导入显示、数组与张量库
#
# `from IPython.display import Image` 导入图片显示类；`import numpy as np` 给NumPy模块起别名；`torch`提供Tensor算术；`from torch import nn`导入神经网络命名空间，但本数值示例没有用它定义网络。导入执行模块初始化，不会加载OPT权重。后续先用torch演示单向量，再用NumPy演示矩阵。

# %% 原单元 0
# 导入显示、数组与张量库：详见本格中文说明，原执行语句保持不变。
# IPython图片展示类；这里只导入，不读图。
from IPython.display import Image
# 数组模块别名np，后面用二维ndarray演示。
import numpy as np
# 张量库，先做CPU单向量运算。
import torch
# 神经网络命名空间；本数值段未定义nn.Module。
from torch import nn

# %% [markdown]
# ## 单元 1（markdown）：基础量化入口
#
# basics即基础。先弄清缩放、舍入、重建，再理解矩阵量化；当前Markdown不执行Python。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：

# %% [markdown] 原单元 1
# ## basics

# %% [markdown]
# ## 单元 2（code）：浮点位布局配图
#
# `Image`是IPython显示对象，filename为相对当前工作目录的`../imgs/floats-rep.png`；`width=500`指定页面上的展示宽度。图片返回对象由Jupyter显示；脚本运行时最后一个表达式不会自动显示。应从`代码/tutorials`作为工作目录运行，三张原图已放到相邻`代码/imgs`。 图中蓝色sign为符号位；FP32为1+8指数+23显式尾数位，FP16为1+5+10，BF16为1+8+7。TF32图表示计算精度/范围，不是可以直接创建的19bit PyTorch存储dtype；详见第05讲。

# %% 原单元 2
# 浮点位布局配图：详见本格中文说明，原执行语句保持不变。
# 从当前工作目录读取原图；width控制展示像素，不影响量化算法。
Image('../imgs/floats-rep.png', width=500)

# %% [markdown]
# ## 单元 3（markdown）：论文、库与异常特征
#
# LLM.int8论文组合vector-wise quantization（向量级量化）和mixed-precision decomposition（混合精度分解）。bitsandbytes是相关开源实现；QLoRA是相关作者后续工作，不能把本讲等同于QLoRA。原英文emergent features指随模型规模涌现、幅度很大且系统出现的特征；少量这样的特征可能对注意力和预测重要。论文、源码的固定版本与实验边界见主解释；本格链接不是安装命令。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# - LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
#     - vector-wise quantization & mixed precision decomposition
#     - https://arxiv.org/abs/2208.07339
# - 作者也是开源库：bitsandbytes 的作者
#     - https://github.com/TimDettmers/bitsandbytes
#     - Tim Dettmers，也是 QLora 的作者
# - emergent features
#     - highly systematic emergent features in transformer language models that dominate attention and transformer
# predictive performance.
#     - emergent outliers vs. Regular values

# %% [markdown] 原单元 3
# - LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
#     - vector-wise quantization & mixed precision decomposition
#     - https://arxiv.org/abs/2208.07339
# - 作者也是开源库：bitsandbytes 的作者
#     - https://github.com/TimDettmers/bitsandbytes
#     - Tim Dettmers，也是 QLora 的作者
# - emergent features
#     - highly systematic emergent features in transformer language models that dominate attention and transformer
# predictive performance.
#     - emergent outliers vs. Regular values

# %% [markdown]
# ## 单元 4（markdown）：单向量演示标题
#
# 从一个向量的最大幅度开始，观察一套尺度如何影响每个元素；不是网络训练。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# ### 简单过程展示

# %% [markdown] 原单元 4
# ### 简单过程展示

# %% [markdown]
# ## 单元 5（code）：量化与信息损失配图
#
# 读取`../imgs/quant-freeze.png`，宽度400像素；展示量化示意，不执行压缩内核，不证明推理准确率。配图与第9–12格算术对照。 从左到右：8项向量→absmax5.4→放大倍数127/5.4≈23.5185→整数编码→除相同倍数重建。图把23.5185显示成23.5、重建保留一位小数，不能据下排相同文字断言无损；例如编码28实际重建28*5.4/127≈1.19055，不等于1.2。

# %% 原单元 5
# 量化与信息损失配图：详见本格中文说明，原执行语句保持不变。
# 读取原量化示意图，返回可供Jupyter展示的Image对象。
Image('../imgs/quant-freeze.png', width=400)

# %% [markdown]
# ## 单元 6（markdown）：缩放与逆缩放公式
#
# 设x为浮点元素，x_max为向量absmax，factor=127/x_max。第一式把x缩放成y，第二式逆缩放；原公式没有写round，因此仅缩放再逆缩放可代数抵消。实际有损量化是q=round(127*x/x_max)，再重建x_hat=q*x_max/127。x_max=0时要单独处理，原格没有保护；127对应对称范围[-127,127]，不是int8全部256种编码。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# $$
# y=\frac{x}{x_\max}\cdot 127\\
# x = \frac{y\cdot x_\max}{127}
# $$
#
# - $\text{factor}=\frac{127}{x_\max}$

# %% [markdown] 原单元 6
# $$
# y=\frac{x}{x_\max}\cdot 127\\
# x = \frac{y\cdot x_\max}{127}
# $$
#
# - $\text{factor}=\frac{127}{x_\max}$

# %% [markdown]
# ## 单元 7（code）：生成随机向量
#
# `torch.randn(5)`函数默认生成CPU浮点Tensor，shape=[5]、标准正态分布；本环境默认float32。未设seed时每次值不同。末行`a`由notebook展示；赋值将变量绑定到新Tensor，下一格求尺度。

# %% 原单元 7
# 生成随机向量：详见本格中文说明，原执行语句保持不变。
# 标准正态5项，默认CPU float32[5]；每次运行重新采样。
a = torch.randn(5)
# 裸表达式在Jupyter显示，不改变数据。
a

# %% [markdown]
# ## 单元 8（code）：求整个向量absmax
#
# `torch.abs(a)`逐元素取绝对值；结果的`.max()`是Tensor方法，不传dim时归约所有元素，返回0维Tensor[]。`a_max`是幅度，不是最大带符号值；不改变原a。它会广播到第9格的所有5个元素。

# %% 原单元 8
# 求整个向量absmax：详见本格中文说明，原执行语句保持不变。
# 逐元素绝对值再全局归约，标量absmax[]供共享缩放。
a_max = torch.abs(a).max()
# 展示尺度；max不是最大带符号值。
a_max

# %% [markdown]
# ## 单元 9（code）：把幅度缩到127范围
#
# `a/a_max`逐元素除以共享尺度，非零a_max时落在[-1,1]；乘127得到`scaled`，shape仍[5]、float32。标量[]广播到向量，不需要复制成5项；本格没有舍入和int8存储转换。

# %% 原单元 9
# 把幅度缩到127范围：详见本格中文说明，原执行语句保持不变。
# a/a_max [-1, 1] ==> [-127, 127]
# 标量广播到5项，缩放到[-127,127]；仍float32。
scaled = (a/a_max)*127
# 展示缩放向量；尚未舍入。
scaled

# %% [markdown]
# ## 单元 10（code）：舍入数值但保留浮点容器
#
# `torch.round(scaled)`按最近整数舍入，中点采用ties-to-even（例如0.5→0，1.5→2）；返回与输入相同shape/dtype的Tensor。`int8val`只是变量名，仍为float32[5]。真正每元素1字节需要`.to(torch.int8)`，此处保留原程序不修正。

# %% 原单元 10
# 舍入数值但保留浮点容器：详见本格中文说明，原执行语句保持不变。
# rounded
# round舍入数值、保留float32容器；变量名int8val不等于int8存储。
int8val = torch.round(scaled)
# 查看舍入后的浮点Tensor[5]。
int8val

# %% [markdown]
# ## 单元 11（code）：逆缩放重建
#
# Tensor方法`.float()`返回/转换到float32，不会撤销round的损失；本例输入已经float32。乘a_max/127恢复尺度，`fp32val`是近似重建向量[5]。零尺度保护仍未加入。

# %% 原单元 11
# 逆缩放重建：详见本格中文说明，原执行语句保持不变。
# rescaling
# 逆缩放到原幅度，float()不能恢复舍入损失，输出float32[5]。
fp32val = int8val.float() * a_max / 127

# %% [markdown]
# ## 单元 12（code）：平均绝对重建误差
#
# 先计算`fp32val-a`，abs消除符号，mean归约5项返回float32标量[]；这是单向量MAE，不是语言模型loss或准确率。随机值改变会导致误差变化。

# %% 原单元 12
# 平均绝对重建误差：详见本格中文说明，原执行语句保持不变。
# 5项绝对误差的均值，输出标量[]，不是模型准确率。
torch.abs(fp32val - a).mean()

# %% [markdown]
# ## 单元 13（code）：合并单向量流程并打印
#
# 重新randn覆盖a和后续变量，不能与第7格同一个样本混用。流程仍是absmax→缩放→round→反量化→MAE；三个print返回None并输出原向量、重建向量和误差。int8val没有int8转换，fp32val只沿用输入float32。

# %% 原单元 13
# 合并单向量流程并打印：详见本格中文说明，原执行语句保持不变。
# 重新采样，覆盖旧a。
a = torch.randn(5)
# 整个新向量共用标量absmax。
a_max = torch.abs(a).max()
# 浮点缩放，shape[5]不变。
scaled = a/a_max * 127
# 最近整数舍入，dtype仍float32。
int8val = torch.round(scaled)
# 逆缩放重建；没有真正int8编码。
fp32val = int8val * a_max / 127
# 打印新原向量。
print(a)
# 打印近似重建向量。
print(fp32val)
# 计算并打印平均绝对误差。
print(torch.mean(torch.abs(a - fp32val)))

# %% [markdown]
# ## 单元 14（markdown）：异常值为何影响普通值
#
# outliers即异常幅度值。共享一个最大尺度时，一个大幅度元素会增大所有普通元素的量化步长；不能仅删除它，因为它可能承担重要模型信息。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# ### outliers 对量化的影响

# %% [markdown] 原单元 14
# ### outliers 对量化的影响

# %% [markdown]
# ## 单元 15（markdown）：量化误差的直觉
#
# 原句指出异常值会令量化误差变大。精确说，这是共享尺度下普通值分辨率下降的风险，并非任意两次随机采样必定误差单调上升；第16、17格重新采样，不构成严格同输入对照。补充验证固定其余元素、只改变末项。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# - 异常值会显著地增加量化误差；

# %% [markdown] 原单元 15
# - 异常值会显著地增加量化误差；

# %% [markdown]
# ## 单元 16（code）：原位设置末项异常值
#
# `a[-1]`是最后一个元素，负索引从尾部数；赋值`=-5`修改现有Tensor的内容，不增加维度。重新randn得到float32[5]，这里仅展示它，还没有重建。其余随机项理论上也可能超过5，不能一概断言absmax必然等于5。

# %% 原单元 16
# 原位设置末项异常值：详见本格中文说明，原执行语句保持不变。
# 新随机float32[5]。
a = torch.randn(5)
# 手动地添加一个异常值
# 负索引-1是最后一项，赋值原位修改为-5。
a[-1] = -5
# 展示含大幅度末项的向量。
a

# %% [markdown]
# ## 单元 17（code）：含异常值的完整量化演示
#
# 再次创建a并把末项改成-5；max尺度同时作用于全部元素。`int7val`是原拼写，实际同样round到127范围，名称中的7是原变量拼写，这里使用的量化范围仍以127为界。返回重建及MAE；本次用固定seed重跑，与原保存值分别记录。

# %% 原单元 17
# 含异常值的完整量化演示：详见本格中文说明，原执行语句保持不变。
# 再采样，和第16格不是同一随机向量。
a = torch.randn(5)
# 手动地添加一个异常值
# 原位改末项；共享尺度可能增大量化步长。
a[-1] = -5
# absmax由所有元素决定，随机其他项也可能更大。
a_max = torch.abs(a).max()
# 仍是浮点缩放，不改变dtype。
scaled = a/a_max*127
# int7val是原拼写，round并不代表7bit。
int7val = torch.round(scaled)
# 重建float32向量。
fp32val = int7val * a_max / 127
# 原向量输出。
print(a)
# 重建输出。
print(fp32val)
# 单次MAE；不能与不同随机输入当成严格误差对照。
print(torch.mean(torch.abs(a - fp32val)))

# %% [markdown]
# ## 单元 18（markdown）：LLM.int8算法部分
#
# 接下来将单向量尺度推广到矩阵的每行/每列，再把异常特征交给浮点支路；并非整个LLM都成为整数。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：

# %% [markdown] 原单元 18
# ## LLM.int8

# %% [markdown]
# ## 单元 19（markdown）：向量级量化标题
#
# vector-wise意味着每个输入行/权重列各有尺度，比整个矩阵共用一套尺度更适合内积。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# ### vector-wise quantization

# %% [markdown] 原单元 19
# ### vector-wise quantization

# %% [markdown]
# ## 单元 20（markdown）：每个向量自己的normalization constant
#
# 输入行向量和paper权重列向量分别保存最大幅度。若用absmax，则量化放大倍数127/absmax和反量化尺度absmax/127互为倒数；不要把不同文献里同名constant当成同一方向。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# - 每一个 vector（row vector/column vector）计算 normalization constant

# %% [markdown] 原单元 20
# - 每一个 vector（row vector/column vector）计算 normalization constant

# %% [markdown]
# ## 单元 21（markdown）：矩阵乘法的维度契约
#
# A[b,h]与B[h,o]沿共享h求和，得到C[b,o]。b为输入行/token数，h为输入特征，o为输出特征。每个C[i,j]是A第i行和B第j列的内积。PyTorch Linear保存weight[o,h]，paper方向需要转置。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# $$
# A@B=C\\
# [b\times h] @ [h\times o]\rightarrow [b\times o]
# $$

# %% [markdown] 原单元 21
# $$
# A@B=C\\
# [b\times h] @ [h\times o]\rightarrow [b\times o]
# $$

# %% [markdown]
# ## 单元 22（markdown）：Markdown中的完整量化伪代码
#
# 本格不是可直接执行的Python：`A16.absmax(1)`并非标准Tensor方法；原`B:[b,o]`维度应为[h,o]。正确尺度为a=abs(A).max(axis=1,keepdims=True)[b,1]、d=abs(B).max(axis=0,keepdims=True)[1,o]；零尺度须保护。先round并存成int8，转int32累加Q_A@Q_B，再乘a*d/127²得到浮点[b,o]。原absmax_A_vec@absmax_B_vec仅在明确二维列/行形状时才是外积；两个1D向量@会做内积或维度失败。原文字缺少round、类型转换和零尺度处理，中文注释说明纠正但不改原单元。公式对应论文式(7)，参数/返回shape见主解释第3节。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# - A, B 都是 fp16，现在要做的是转化为 int8
# - normailze A & B
#     - 对于 A16
#         - A: [b, h] => [b]
#         - absmax_A_vec = A16.absmax(1)
#         - A8 = A16/absmax_A_vec * 127
#     - 对于 B16
#         - B: [b, o] => [o]
#         - absmax_B_vec = B16.absmax(0)
#         - B8 = B16/absmax_B_vec * 127
# - 计算 C32 = A8 @ B8
# - C16f = C32 \* (absmax_A_vec @ absmax_B_vec) / (127\*127)

# %% [markdown] 原单元 22
# - A, B 都是 fp16，现在要做的是转化为 int8
# - normailze A & B
#     - 对于 A16
#         - A: [b, h] => [b]
#         - absmax_A_vec = A16.absmax(1)
#         - A8 = A16/absmax_A_vec * 127
#     - 对于 B16
#         - B: [b, o] => [o]
#         - absmax_B_vec = B16.absmax(0)
#         - B8 = B16/absmax_B_vec * 127
# - 计算 C32 = A8 @ B8
# - C16f = C32 \* (absmax_A_vec @ absmax_B_vec) / (127\*127)

# %% [markdown]
# ## 单元 23（markdown）：矩阵例子入口
#
# 接下来固定3行、5特征、2输出，能手算两条支路并与完整矩阵乘法对照；不涉及OPT权重。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：

# %% [markdown] 原单元 23
# ## 一个计算示例

# %% [markdown]
# ## 单元 24（code）：论文两支路示意图
#
# 原图`../imgs/llm-int8.png`：输入与权重按同一个输入特征维拆分，普通支路量化/整数乘法/反量化，异常支路浮点乘法，两者同shape后相加。图中的dtype是算法设计，后面的NumPy变量名不保证实际达到图中dtype。

# %% 原单元 24
# 论文两支路示意图：详见本格中文说明，原执行语句保持不变。
# 原图展示两条支路；读图不执行CUDA内核。
Image('../imgs/llm-int8.png')

# %% [markdown]
# ## 单元 25（code）：构建固定输入与paper方向权重
#
# `np.asarray`函数将嵌套整数列表转为二维ndarray：X[3,5]、W[5,2]。未传dtype，平台默认整数，本次Windows NumPy1.26.4实际int32；不预设float16。X列1/3有大幅度45/12/37与-17/-63/-83。方括号是列表，逗号分隔行，最后一个尾逗号不增加额外参数。

# %% 原单元 25
# 构建固定输入与paper方向权重：详见本格中文说明，原执行语句保持不变。
# 列为特征，行为样本
# 整数嵌套列表→X[3,5]；未指定float16，实际dtype由NumPy平台决定。
X = np.asarray([[2, 45, -1, -17, -1],
                # 第二输入行，共5特征。
                [0, 12, 3, -63, 2],
                # 第三输入行；尾逗号不新增矩阵轴。
                [-1, 37, -1, -83, 0]], )
# paper方向权重W[5,2]，共享特征维5。
W = np.asarray([[-1, 0],
                # 特征1的两项权重。
                [2, 0],
                # 特征2的两项权重。
                [0, -2],
                # 特征3的两项权重。
                [3, -2],
                # 特征4的两项权重，结束矩阵。
                [-1, 2]])

# %% [markdown]
# ## 单元 26（code）：完整矩阵乘法基准
#
# ndarray方法`X.dot(W)`在二维时做矩阵乘法，输出[3,2]；不是逐元素`*`。本例整数基准[[38,34],[-167,124],[-174,168]]，值小没有本次整数溢出。下一步拆分仍必须相加恢复这个输出空间。

# %% 原单元 26
# 完整矩阵乘法基准：详见本格中文说明，原执行语句保持不变。
# (3, 5) * (5, 2) ==> (3, 2)
# ground truth
# 二维dot是矩阵乘法，整数基准[3,2]。
X.dot(W)

# %% [markdown]
# ## 单元 27（code）：按特征维拆普通与异常部分
#
# 冒号保留该轴全部；索引列表做高级索引，返回选取的新数组。X取列，W取行：reg[0,2,4]给X_reg[3,3]/W_reg[3,2]；out[1,3]给X_out[3,2]/W_out[2,2]。两组互斥且合并覆盖5特征，避免丢失/重复求和。原程序手工选列；补充按max(abs(X),axis=0)>6自动识别。

# %% 原单元 27
# 按特征维拆普通与异常部分：详见本格中文说明，原执行语句保持不变。
# X取普通特征列0/2/4→[3,3]；高级索引产生选取数组。
X_reg = X[:, [0, 2, 4]]
# W取相同特征行0/2/4→[3,2]。
W_reg = W[[0, 2, 4], :]

# X取异常特征列1/3→[3,2]。
X_out = X[:, [1, 3]]
# W取相同特征行1/3→[2,2]，两支路必须覆盖且互斥。
W_out = W[[1, 3], :]

# %% [markdown]
# ## 单元 28（markdown）：普通支路标题
#
# small values是相对异常特征而言；本格8bit只是设计目标，后续还要检查round与astype才能判断真实存储。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# ### regular parts (small values) => 8bit quant

# %% [markdown] 原单元 28
# ### regular parts (small values) => 8bit quant

# %% [markdown]
# ## 单元 29（code）：观察普通输入
#
# `X_reg`是[3,3]整数数组，行依次为[2,-1,-1]、[0,3,2]、[-1,-1,0]。裸变量表达式只展示当前对象，不重新计算或复制变量。

# %% 原单元 29
# 观察普通输入：详见本格中文说明，原执行语句保持不变。
# 普通输入[3,3]，仅查看。
X_reg

# %% [markdown]
# ## 单元 30（code）：先观察列absmax
#
# `np.abs`逐元素绝对值；`np.max(...,axis=0)`沿行归约，返回每列最大幅度[2,3,2]，shape=[3]。这是探索，不是第33格输入行尺度；axis=0与axis=1不能互换。

# %% 原单元 30
# 先观察列absmax：详见本格中文说明，原执行语句保持不变。
# axis=0归约行，输出每列absmax[3]；不是后面输入每行尺度。
np.max(np.abs(X_reg), axis=0)

# %% [markdown]
# ## 单元 31（code）：观察普通权重
#
# W_reg[3,2]依次[-1,0]、[0,-2]、[-1,2]；仍是paper方向输入特征行、输出列。它与X_reg的3个共享普通特征一一对应。

# %% 原单元 31
# 观察普通权重：详见本格中文说明，原执行语句保持不变。
# 普通权重[3,2]，与3个普通特征对齐。
W_reg

# %% [markdown]
# ## 单元 32（markdown）：准备两套向量尺度
#
# 输入每行absmax，权重每列absmax；接下来用新增轴表达可广播的列/行向量。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# ####  vector-wise constants

# %% [markdown] 原单元 32
# ####  vector-wise constants

# %% [markdown]
# ## 单元 33（code）：归约并插入长度1轴
#
# Cx=np.max(abs(X_reg),axis=1)得到[2,3,1]，`[:,np.newaxis]`变[3,1]。Cw沿axis=0得到[1,2]，`[np.newaxis,:]`变[1,2]。np.newaxis即None，只增加长度1维，不增添数值；广播后Cx*Cw为[3,2]输出尺度。

# %% 原单元 33
# 归约并插入长度1轴：详见本格中文说明，原执行语句保持不变。
# 行最大（振幅）
# axis=1→每行absmax[3]，加新轴变Cx[3,1]以便广播。
Cx = np.max(np.abs(X_reg), axis=1)[:, np.newaxis]
# 列最大（振幅）
# axis=0→每列absmax[2]，加新轴变Cw[1,2]。
Cw = np.max(np.abs(W_reg), axis=0)[np.newaxis, :]

# %% [markdown]
# ## 单元 34（code）：检查权重尺度
#
# Cw=[[1,2]]、shape[1,2]，每个输出列一个尺度。仅展示Cw，不会执行量化。

# %% 原单元 34
# 检查权重尺度：详见本格中文说明，原执行语句保持不变。
# 权重两列尺度[[1,2]]。
Cw

# %% [markdown]
# ## 单元 35（markdown）：量化阶段标题
#
# 标题quantize需要与下一格实际语句区分；原下一格只缩放，没有完整整数编码。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# #### quantize

# %% [markdown] 原单元 35
# #### quantize

# %% [markdown]
# ## 单元 36（code）：原程序仅缩放，无舍入或转换
#
# `127/Cx`、`127/Cw`为浮点除法，分别广播到X_reg[3,3]/W_reg[3,2]。实际X_i8/W_i8为float64，不是int8；例如-1/2*127=-63.5被保留，不变成-64。判断位宽时看数组的dtype；整数补充算例另列round与astype步骤，便于比较量化前后的数值。

# %% 原单元 36
# 原程序仅缩放，无舍入或转换：详见本格中文说明，原执行语句保持不变。
# element-wise 乘法
# 127/Cx浮点广播；没有round/astype，X_i8实际float64[3,3]。
X_i8 = X_reg * (127/Cx)
# 127/Cw浮点广播；W_i8实际float64[3,2]，名称不改变dtype。
W_i8 = W_reg * (127/Cw)

# %% [markdown]
# ## 单元 37（markdown）：整数乘法的设计目标
#
# 标题int8 matmul描述论文普通支路；第38格实际仍使用float64 dot，没有整数CUDA内核。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# #### int8 matmul

# %% [markdown] 原单元 37
# #### int8 matmul

# %% [markdown]
# ## 单元 38（code）：缩放后的浮点矩阵乘法
#
# X_i8[3,3].dot(W_i8[3,2])得到Out_i32[3,2]，实际float64。名称i32不等于累加器int32。补充先round→int8存储，再转int32@，防止直接int8累加溢出。

# %% 原单元 38
# 缩放后的浮点矩阵乘法：详见本格中文说明，原执行语句保持不变。
# 原dot输入float64，Out_i32也是float64[3,2]，没有int32累加器。
Out_i32 = X_i8.dot(W_i8)

# %% [markdown]
# ## 单元 39（markdown）：反量化阶段标题
#
# dequant即恢复原数值尺度。已丢掉的舍入细节无法恢复；原程序没有舍入，所以原算例近乎精确恢复并不奇怪。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# #### dequant

# %% [markdown] 原单元 39
# #### dequant

# %% [markdown]
# ## 单元 40（code）：输入尺度shape
#
# ndarray的`.shape`是属性而非函数调用，返回Python tuple(3,1)。长度1列维用于输出[3,2]广播。

# %% 原单元 40
# 输入尺度shape：详见本格中文说明，原执行语句保持不变。
# shape是ndarray属性，返回(3,1)而非Tensor。
Cx.shape

# %% [markdown]
# ## 单元 41（code）：权重尺度shape
#
# Cw.shape返回tuple(1,2)。与Cx(3,1)结合形成(3,2)而非长度2内积。

# %% 原单元 41
# 权重尺度shape：详见本格中文说明，原执行语句保持不变。
# 返回权重尺度形状(1,2)。
Cw.shape

# %% [markdown]
# ## 单元 42（code）：Kronecker积在本形状下等于尺度外积
#
# `np.kron(Cx,Cw)`函数执行Kronecker积，当前列向量[3,1]和行向量[1,2]恰好给[[2,4],[3,6],[1,2]]，shape[3,2]。更直接是广播`Cx*Cw`；一般任意形状kron不是逐元素乘法。

# %% 原单元 42
# Kronecker积在本形状下等于尺度外积：详见本格中文说明，原执行语句保持不变。
# 当前列/行向量kron等于输出尺度外积，shape[3,2]；一般kron不是*。
np.kron(Cx, Cw)

# %% [markdown]
# ## 单元 43（code）：恢复普通贡献
#
# Out_i32逐元素乘输出尺度/127²，得到Out_f16_1[3,2]；原输入float64，结果仍float64，不是FP16。缩放/逆缩放抵消，本例结果接近X_reg@W_reg。没有量化舍入，所以不能由此得出8bit无误差。

# %% 原单元 43
# 恢复普通贡献：详见本格中文说明，原执行语句保持不变。
# element-wise 乘法
# 缩放抵消，恢复普通贡献[3,2]；未显式转FP16，实际float64。
Out_f16_1 = Out_i32 * (np.kron(Cx, Cw)) / (127*127)
# 展示恢复结果；近乎精确因为原程序没有量化舍入。
Out_f16_1

# %% [markdown]
# ## 单元 44（markdown）：异常支路的设计目标
#
# 标题16-bit表示论文高精度支路；第45格原整数ndarray没有FP16转换，实际仍需检查dtype。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：
#
# ### 16-bit

# %% [markdown] 原单元 44
# ### 16-bit

# %% [markdown]
# ## 单元 45（code）：异常贡献的矩阵乘法
#
# X_out[3,2].dot(W_out[2,2])输出[3,2]，本环境实际int32；原Out_f16_2名字不改变dtype。异常支路保留大幅度特征的贡献，不把它们裁剪或丢掉。补充可把它们显式转FP16以验证真实半精度算术，但不能声称运行bitsandbytes。

# %% 原单元 45
# 异常贡献的矩阵乘法：详见本格中文说明，原执行语句保持不变。
# 异常特征矩阵乘法[3,2]；原整数输入仍整数输出，不是真FP16。
Out_f16_2 = X_out.dot(W_out)

# %% [markdown]
# ## 单元 46（code）：两支路相加
#
# 两个[3,2]数组逐元素加，float64普通支路和整数异常支路组合得到float64输出。原未舍入示例接近完整X.dot(W)，是一项代数拆分演示。本次整数量化补充的误差另记，不能覆盖原历史输出。

# %% 原单元 46
# 两支路相加：详见本格中文说明，原执行语句保持不变。
# 同shape两支路相加，不重叠特征；float64+整数→float64。
Out_f16_1 + Out_f16_2

# %% [markdown]
# ## 单元 47（markdown）：Hugging Face集成入口
#
# 前面讲算法，后面通过预训练OPT模型展示集成入口；原notebook没有训练loss/optimizer/dataset，也没有论文C4评测。
#
# 原Markdown（保留原符号/伪代码，可与上面中文说明对照）：

# %% [markdown] 原单元 47
# ## huggingface 中的使用

# %% [markdown]
# ## 单元 48（code）：三个安装提示都是注释
#
# 三行以#开头，Python不执行pip，也不锁版本。bitsandbytes负责相关低精度层，transformers负责模型/tokenizer，accelerate负责设备分配等。真正安装需在终端执行并匹配平台；CPU算例所用的Transformers版本见复现环境记录。

# %% 原单元 48
# 三个安装提示都是注释：详见本格中文说明，原执行语句保持不变。
# pip install bitsandbytes
# pip install transformers
# pip install accelerate

# %% [markdown]
# ## 单元 49（code）：导入Auto加载接口
#
# AutoTokenizer和AutoModelForCausalLM是transformers类，from_pretrained是类加载入口：根据checkpoint的配置选择具体tokenizer/因果语言模型。导入本身不下载6.7B权重；返回对象、网络下载和设备迁移发生在下一几格。

# %% 原单元 49
# 导入Auto加载接口：详见本格中文说明，原执行语句保持不变。
# Auto类根据checkpoint配置选择具体实现，导入不下载OPT。
from transformers import AutoTokenizer, AutoModelForCausalLM

# %% [markdown]
# ## 单元 50（code）：修改进程代理环境
#
# os是标准库，os.environ是当前进程环境的可变映射；两条赋值会影响后续HTTP/HTTPS请求。127.0.0.1:7890必须是本机正在运行的代理，不能假定每人都有。原语句保留，本次跳过此格，不修改真实代理环境。

# %% 原单元 50
# 修改进程代理环境：详见本格中文说明，原执行语句保持不变。
# 标准库环境接口。
import os
# 修改当前进程HTTP代理；读者必须确有此服务，本次跳过。
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# 同样修改HTTPS代理，非算法参数。
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# %% [markdown]
# ## 单元 51（code）：prompt到token ID
#
# MAX_NEW_TOKENS只是int变量128，ckpt是远程模型ID字符串，sample是英文prompt。AutoTokenizer.from_pretrained默认可联网下载/缓存文件；tokenizer(...)是对象可调用接口，return_tensors=pt返回BatchEncoding，`.input_ids`为int64[1,S]，通常默认CPU。只取input_ids丢掉其他字段，原格没显式保留attention_mask。原OPT未下载；教学补充独立使用本地小词表，不伪造此格执行成功。

# %% 原单元 51
# prompt到token ID：详见本格中文说明，原执行语句保持不变。
# 变量名说new，但第53格实际传max_length，限制总长度。
MAX_NEW_TOKENS = 128
# 远程OPT6.7B模型标识；加载可能下载大权重/tokenizer。
ckpt = 'facebook/opt-6.7b'

# 一条英文输入字符串，不是训练dataset。
sample = 'hello, who are you?'
# 类加载入口下载/缓存tokenizer文件，本次原格未执行。
tokenizer = AutoTokenizer.from_pretrained(ckpt)
# 返回BatchEncoding，取int64[1,S]词表索引，默认CPU，不能改成int8。
input_ids = tokenizer(sample, return_tensors='pt').input_ids
# 仅查看输入ID。
input_ids

# %% [markdown]
# ## 单元 52（code）：加载并量化OPT底座
#
# from_pretrained的ckpt指定预训练模型；device_map=auto要求加载机制根据资源分配层；load_in_8bit=True触发历史bitsandbytes集成与线性层替换。返回model并涉及权重、量化状态、设备内存；不是把全部参数/激活/cache改成int8，也不是数据并行训练。6.7B权重未下载，此格未执行。CPU小OPT和参数遍历仅验证共享生成接口，不替代量化加载。

# %% 原单元 52
# 加载并量化OPT底座：详见本格中文说明，原执行语句保持不变。
# 原6.7B量化加载会下载权重并分配内存，本次未执行。
model = AutoModelForCausalLM.from_pretrained(ckpt,
                                             # auto让加载器分配模块设备；8bit替换可量化线性层，非所有张量整数化。
                                             device_map='auto', load_in_8bit=True, )

# %% [markdown]
# ## 单元 53（code）：生成总长度与decode
#
# model.generate读取input_ids[1,S]，返回generated_ids[1,S+T]；原传max_length=128限制总长度，尽管变量名叫MAX_NEW_TOKENS。max_new_tokens才限制新增数量。decode取第一个batch完整序列，包括prompt，skip_special_tokens=True仅移除注册的特殊token。设备需与模型输入层匹配。教学本地小OPT实际执行这两行，但设置总长8并明确不量化。

# %% 原单元 53
# 生成总长度与decode：详见本格中文说明，原执行语句保持不变。
# 生成结果包括prompt，max_length限制总token数，不是最多新增128。
generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
# 取第一个样本解码，跳过特殊token但不会自动移除prompt。
tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# %% [markdown]
# ## 单元 54（code）：枚举参数设备与存储类型
#
# nn.Module方法named_parameters()默认recurse=True，迭代(name,Parameter)二元组；enumerate再加整数序号i。para[0]为名字，para[1]为Tensor参数，.device/.dtype是属性。f字符串把表达式插入文字，\t是制表符，print返回None。这里观察参数，不更新参数；尺度/布局缓存未必在named_parameters里，不能据此推算全部显存或证明int8内核执行。

# %% 原单元 54
# 枚举参数设备与存储类型：详见本格中文说明，原执行语句保持不变。
# 遍历(名字,Parameter)，enumerate另给i；只观察，不更新参数。
for i, para in enumerate(model.named_parameters()):
#     print(f'{i}, {para[0]}\t {para[1].device} \t{para[1].dtype}')
    # f字符串插值设备/dtype；尺度缓存不一定在参数迭代器内。
    print(f'{i}:{para[0]}, \t {para[1].device} \t{para[1].dtype}')

# %% [markdown]
# ## 单元 55（code）：空代码格
#
# 原source为空，保留编号、元数据和历史输出字段。空格没有模型逻辑，不凭空添加训练/评价内容；注释版只加入说明性注释，AST仍为空。

# %% 原单元 55
# 空代码格：详见本格中文说明，原执行语句保持不变。
