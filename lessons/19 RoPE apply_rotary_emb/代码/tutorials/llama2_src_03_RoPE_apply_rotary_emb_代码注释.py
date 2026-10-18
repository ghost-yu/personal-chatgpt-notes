# -*- coding: utf-8 -*-
# 第19讲原代码和详注；普通脚本不自动显示裸表达式，helper缺失保留。

# %% 0
# 位置机制与顺序
# 没有位置机制及其他顺序约束的self-attention对置换等变，不表示Transformer输出是完全相同。causal mask也引入顺序约束；RoPE用乘性旋转显式编码位置。
# - Transformer中Position Embedding的意义：“破坏Transformer结构的完全对称性”。
#     - 即对 sequence 中的 token 顺序敏感

# %% 1
# 导入库
# 原代码导入 NumPy 为 np、matplotlib.pyplot 为 plt、torch、torch.nn 为 nn、typing.Tuple。`import ... as ...` 给模块短名称，不创建模型实例。
#
# np 用于后面的旋转矩阵与复数求和；plt 用于绘图；torch 用于 Tensor 旋转；Tuple 用于返回类型注解。nn 在本 notebook 后续没有使用，可以识别为未使用导入，不能据此猜测这里定义了神经网络类。导入没有 Tensor shape，也没有模型权重或训练动作。
# NumPy别名np，用于实矩阵、复数相位与求和。
import numpy as np
# 绘图接口plt，画数学统计量而非训练指标。
import matplotlib.pyplot as plt
# Torch Tensor、复数视图和自动求导。
import torch
# nn提供神经网络模块接口。
from torch import nn
# 返回类型注解Tuple；不自动验证shape。
from typing import Tuple

# %% 2
# cis编码小节
# cis为cos+i sin，位置对应复数单位因子。
# ## cis 编码

# %% 3
# 复数与二维矩阵
# q=q0+iq1与列向量[q0,q1]对应；乘exp(imθ)等于R(mθ)q。m是位置、θ是频率弧度/位置，不是precompute的基数theta。shape[2]→[2]，原代码度制示例稍后转换。
# $$
# \begin{split}
# f(q,m)&=qe^{im\theta}\\
# f(q,m)&=\begin{pmatrix}
# \cos m\theta & -\sin m\theta\\
# \sin m\theta & \cos m\theta
# \end{pmatrix}\begin{pmatrix}
# q_0\\
# q_1
# \end{pmatrix}\\
# &=R_mq
# \end{split}
# $$

# %% 4
# 旋转合成与转置
# R(α)R(β)=R(α+β)；Rᵀ(α)=R(-α)，因此Rᵀ(α)R(β)=R(β-α)。矩阵乘不是逐元素乘，列向量计算从右向左。
# $$
# \begin{split}
# R(\alpha)R(\beta)&=R(\alpha+\beta)\\
# R^T(\alpha)R(\beta)&=\begin{pmatrix}
# \cos\alpha & \sin\alpha\\
# -\sin\alpha & \cos\alpha
# \end{pmatrix}\begin{pmatrix}
# \cos\beta & -\sin\beta\\
# \sin\beta & \cos\beta
# \end{pmatrix}\\
# &=R(-\alpha)R(\beta)\\
# &=R(\beta-\alpha)
# \end{split}
# $$

# %% 5
# 相对点积
# 固定内容q/k分别在m/n旋转后，点积qᵀR(n-m)k仅显式包含位置差。内容仍决定分数；不能随意将n-m反号。复数形式需内容共轭和取实部。
# $$
# \begin{split}
# f(q,m)^Tf(k,n)&=(R_mq)^T(R_nk)\\
# &=q^T(R^T_mR_n)k\\
# &=q^TR_{n-m}k
# \end{split}
# $$

# %% 6
# 定义 R(theta)
# 这是普通 Python 函数，不属于类，没有 self。输入 theta 在这个函数里是**角度制数值**，第一行 `np.radians(theta)` 将度转成弧度。其后 cos/sin 返回标量，`np.array` 将嵌套列表组合成 float64 `[2,2]` 旋转矩阵。
#
# 返回矩阵的第一行是 `[cos,−sin]`，第二行是 `[sin,cos]`。矩阵乘列向量 `[x,y]` 会得到 `[x cos−y sin,x sin+y cos]`。R(90) 将 `[1,0]` 变为 `[0,1]`。这里 theta 与官方 precompute 的默认 theta=10000 是不同含义：前者为角度，后者为频率基数。
#
# 定义函数只登记函数体，此时没有执行旋转。它不修改全局变量或模型参数。
# 普通函数接收角度制数值，与频率base theta不同。
def R(theta):
    # 局部变量由度转为弧度，不修改调用者或全局配置。
    theta = np.radians(theta)
    # 同一角度的cos/sin标量。
    c, s = np.cos(theta), np.sin(theta)
    # 返回float64[2,2]正角逆时针旋转矩阵。
    return np.array([[c, -s], [s, c]])

# %% 7
# 创建 R1 与 R2
# R1=R(30)、R2=R(60) 实际调用前一单元。二者均为 NumPy float64 `[2,2]`，表示 30° 和 60° 旋转；数字不是 token 位置，也不是弧度。
#
# 变量名 R1/R2 只是分别保存两次返回的数组，并没有共享可训练权重。后续运算需要这两个变量，单独运行后续单元前应先执行 6–7。
# 创建30°矩阵[2,2]。
R1 = R(30)
# 创建60°矩阵[2,2]。
R2 = R(60)

# %% 8
# R1.dot(R2)
# 二维数组的 `.dot` 在此执行矩阵乘法，`[2,2] @ [2,2] → [2,2]`，不是逐元素乘法。将结果作用于一个列向量，先执行右侧 R2 再执行 R1，因此总角度为 60°+30°=90°。
#
# 历史输出对角线约 2.63e-16，是浮点误差，不是新的角度参数。数学上应是 0。本次脚本按 1e-15 绝对容差与 R(90) 比较通过。
# 矩阵乘法，列向量先转60°后30°，合计90°。
R1.dot(R2)

# %% 9
# R1.T.dot(R2)
# `.T` 转置 `[2,2]` 数组。旋转矩阵的转置等于逆矩阵，即 R(−30°)。和 R(60°) 相乘得到 R(30°)，返回 `[2,2]`。
#
# 这是相对位置推导中的关键模式 `R_mᵀ R_n=R_(n−m)`。此处只演示角度差，尚未调用模型注意力。历史结果约为 `[[0.8660254,−0.5],[0.5,0.8660254]]`。
# 转置等于逆30°，再与60°合成正30°，对应相对角度。
R1.T.dot(R2)

# %% 10
# R(30+60)
# 先用 Python 整数加法得到 90，再调用 R。与单元 8 采用不同计算路径验证角度相加性质。返回 `[2,2]`，不更新 R1/R2。浮点结果不要求与单元 8 逐 bit 相同。
# 另一计算路径构造90°，按浮点容差与矩阵乘结果比较。
R(30+60)

# %% 11
# R(60−30)
# 计算 30° 矩阵，与单元 9 对照，验证转置后的角度相减。减法方向不能交换：R(30−60) 会得到相反旋转，通常不同。
# 构造正30°，减法方向不能交换。
R(60-30)

# %% 12
# 核心函数章节
# 接下来只定义旋转函数，不是运行完整attention。
# ## apply_rotary_emb

# %% 13
# 模型维度表
# heads为query头数，dim为D，head_dim=D/Hq，三行均为128。layers不是head数。表没有列K/V头数，不能据此推出xk必须与xq同头数；模型并行时本地头数另计算。
# | **model** | **heads** | **layers** | **dim** | **head_dim** |
# |-----------|-----------|------------|---------|--------------|
# | 7b        | 32        | 32         | 4096    | 4096/32      |
# | 13b       | 40        | 40         | 5120    | 5120/40      |
# | 70b       | 64        | 80         | 8192    | 8192/64      |

# %% 14
# 先投影再旋转
# q=wq(x)、k=wk(x)先完成内容投影，再乘旋转因子；不是将传统sin/cos表加到embedding。没有在此处更新权重或计算V的旋转。
# - position encoding 直接作用于 q(query) 与 k(key)
#     - `q = wq(x)`
#     - `k = wk(x)`
# - 是一种乘性，而非 sin position encoding 的加性；
#     - BERT（Transformer库）的 position encoding 也是加性的；

# %% 15
# 调用签名与shape
# 示例xq/xk=[1,8,32,128]、freq=[8,64]，B/T/H/d分别为batch、当前位置数、头数、每头通道。tuple返回值可解包；这行在Markdown中展示，没有实际调用函数。
# `xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)`
# - input:
#     - xq: [1, 8, 32, 128]
#     - xk: [1, 8, 32, 128]
#     - freqs_cis.shape: [8, 64]

# %% 16
# 定义 apply_rotary_emb
# 这是本讲的核心普通函数，三个输入均为 torch.Tensor，返回 Tuple[Tensor,Tensor]。类型注解说明预期类型，不自动转换输入或执行 shape 检查。源码没有默认参数，没有 self，也没有 nn.Parameter。
#
# ### 输入与用途
#
# 输入 xq 是 query，xk 是 key，通常为浮点 `[B,T,Hq,d]` 和 `[B,T,Hkv,d]`；freqs_cis 为 complex64 `[T,d/2]`。Hq 与 Hkv 可以不同，但 T、d 和频率表必须相容。它将每对通道按对应绝对位置旋转，返回两个实数 Tensor，不返回注意力分数。
#
# ### 两条复数转换语句
#
# 对 xq 和 xk 分别执行以下步骤。`.float()` 把数据转为 float32；即使输入 float64，这里也先降为 float32。`xq.shape[:-1]` 是去掉最后维度的 shape tuple，例子为 `(1,8,32)`；`*` 将 tuple 展开成 reshape 的多个参数。
#
# `reshape(*xq.shape[:-1],−1,2)` 保留前三轴，将 128 配成 64 对：`[1,8,32,128]→[1,8,32,64,2]`。−1 表示由总元素数推断这一维，不表示负的维度长度。最后的 2 是每个复数的实部/虚部，不是新的 token 或 head。
#
# `torch.view_as_complex` 将末轴 `[a,b]` 解释成 a+ib，shape 变为 `[1,8,32,64]`，dtype 为 complex64。它不是丢掉一半通道，存储仍包含两部分。偶数 d、合适 stride 是必要条件；本次还实测了最后 stride=2 的切片输入会报错。
#
# ### 调用 reshape_for_broadcast
#
# 将 `[8,64]` 表变成 `[1,8,1,64]`，位置轴与 T 对齐，batch/head 轴用长度1广播。**本 notebook 没有定义或导入该函数。** Python 允许先定义引用未知名字的函数；只有调用函数体时才报 NameError。单元 16 的定义成功不证明调用成功。
#
# ### 两条复数乘法语句
#
# `xq_ * freqs_cis` 逐元素复数乘，广播后为 `[1,8,32,64]`。乘以 cos+i sin 等于逐对二维旋转，不是 Q 与 K 相乘。`view_as_real` 恢复 `[1,8,32,64,2]` float32，`.flatten(3)` 合并从第3轴到最后轴，回到 `[1,8,32,128]`。这里轴从0计数，flatten(3) 的“3”指第四个轴。
#
# xk 同样处理。不要加入第18讲重建传统 sin/cos 表时使用的 flip，它会交换结果通道。`.type_as(xq/xk)` 各自转回输入 dtype；`return a,b` 创建含两个张量的 tuple，调用方可以用 `xq,xk=...` 解包。
#
# ### 状态、输出与实际补足
#
# 函数不原地修改输入或频率表，不写 cache；输出创建后交给 Attention。单元 16 没有调用示例。本次先实际捕获缺失辅助函数的 NameError，再把固定版本官方 helper 绑定到 notebook 的命名空间，调用它并与官方函数、独立 sin/cos 运算比较。补足仅用于明确记录的后续测试，没有修改原始单元文本。
# 普通前向函数，无参数或缓存状态；本格只定义，未实际调用。
def apply_rotary_emb(
    # query浮点Tensor[B,T,Hq,d]，d必须偶数。
    xq: torch.Tensor,
    # key浮点Tensor[B,T,Hkv,d]，头数可以不同，T/d须相容。
    xk: torch.Tensor,
    # complex64[T,d/2]频率切片，已包含绝对位置，不是原base。
    freqs_cis: torch.Tensor,
# 类型提示返回两个Tensor的tuple，不做运行时shape验证。
) -> Tuple[torch.Tensor, torch.Tensor]:

    # [1, 8, 32, 128] => [1, 8, 32, 64] (*2)
    # FP32→保留B/T/H、末轴配对[...,d/2,2]→complex64[B,T,Hq,d/2]；星号展开shape。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # [1, 8, 32, 128] => [1, 8, 32, 64] (*2)
    # 同理转换key，Hkv可不同；复数view还要求合适stride。
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # [8, 64] => [1, 8, 1, 64]
    # 原notebook未定义此helper，调用时NameError；官方helper补成[1,T,1,d/2]。
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # (xq_ * freqs_cis).shape == [1, 8, 32, 64]
    # torch.view_as_real(xq_ * freqs_cis).shape == [1, 8, 32, 64, 2]
    # flatten(3) == [1, 8, 32, 128]
    # 逐元素复数乘旋转，real恢复末轴2，flatten第3轴起→[B,T,Hq,d]；不加flip。
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)

    # xq_out == [1, 8, 32, 128]
    # key同理，仍不计算QKᵀ、softmax或cache写入。
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # 各自转回输入dtype并组成tuple；局部重绑定不修改原输入。
    return xq_out.type_as(xq), xk_out.type_as(xk)

# %% 17
# 只有注释
# 该单元每行以 `#` 开头，因此没有执行计算。它描述参数 dim=4096、n_heads=32 得 head_dim=128，max_seq_len=1024 得 end=2048，最终表 shape 为 `[2048,64]`。
#
# 注释中的 `2^12/2^5` 是数学说明；在真正 Python 表达式中 `^` 为按位异或，不是乘方，乘方要用 `**`。将这条注释写成计算时，应改为 `2**12/2**5`；实际最大长度由运行配置中的max_seq_len决定。
# 以下全为历史配置注释，不执行；^是说明符号，Python幂要用**。
# self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
# params.dim == 4096, n_heads == 32, 2^12/2^5 == 2^7 == 128
# params.max_seq_len == 1024
# freqs_cis = precompute_freqs_cis(4096//32, 2048)
# freqs_cis.shape

# %% 18
# 加载 SentencePiece 模型
# 从 sentencepiece 导入 SentencePieceProcessor，构造 `sp_model`，用相对路径 `./tokenizer.model` 读取文件。`model_file` 是构造参数名，实例拥有分词模型状态，但不含 Llama Transformer 权重。
#
# 相对路径根据当前工作目录解析，不自动根据 notebook 所在目录解析；缺文件或路径不兼容会失败。原课程 tutorials 下确有同哈希 tokenizer。早期验证复制到临时目录；新验证直接在正式代码/tutorials目录读取同字节文件。两种方式均执行原单元并恢复工作目录，没有改写该单元。
# 加载分词器类，而不是语言模型权重。
from sentencepiece import SentencePieceProcessor
# 按cwd相对路径读原499723byte分词模型；本仓库提供相同字节文件。
sp_model = SentencePieceProcessor(model_file='./tokenizer.model')

# %% 19
# 分词与 BOS
# prompt 是 Python str：`I believe the meaning of life is`。`sp_model.encode(prompt)` 默认得到 token ID 的 list[int]，不是浮点 embedding。`bos_id()` 返回开始标记 ID=1。
#
# 列表表达式 `[bos_id,*encode结果]` 中的 `*` 将一个列表展开成多个元素；得到长度8的列表 `[1,306,4658,278,6593,310,2834,338]`，本次与历史输出一致，没有自动添加 EOS。
#
# shape 可理解为 `[8]` ID 序列，但此时对象仍为 Python list。这里不调用 embedding、Q/K 投影或 apply_rotary_emb；8 个 token 与 markdown 中 seqlen=8 的例子有关，但不能把这个分词输出当成完成模型旋转。
# 待编码Python字符串，未调用embedding。
prompt = 'I believe the meaning of life is'
# BOS ID1前置，星号解包encode列表→8个整数；不添加EOS。
[sp_model.bos_id(), *sp_model.encode(prompt)]

# %% 20
# 远距性质章节
# 原文“远程”在这里指相对距离，不是网络请求；单个实际attention分数不保证单调下降。
# ## 远程（$m-n$）衰减性

# %% 21
# 原作者相关推导
# 链接是RoFormer作者的旋转位置推导资料，作为背景；本讲运行代码不等于复现博客或论文所有结论。
# - Transformer升级之路：2、博采众长的旋转式位置编码
#     - https://spaces.ac.cn/archives/8265/comment-page-1

# %% 22
# 部分和公式
# 用r替代原式重复的i索引以免和虚数单位混淆；S_j=sum前j个复数相位，j0为空和0。真正代码还对各|S_j|求和，不是直接返回某一个S_j或attention概率。
# $$
# \begin{split}
# &S_j=\sum_{i=0}^{j-1}\exp(i(m-n)\theta_i)\\
# &\theta_i=10000^{-2i/d}
# \end{split}
# $$

# %% 23
# 定义 s(m,d=128)
# m 是相对距离，不是下面 arange 的通道编号；d 默认为128。局部 `theta_i=lambda i:10000**(-2*i/d)` 是匿名函数，输入索引数组 i，返回频率数组。
#
# 外层列表推导遍历 j=0,…,63。每次 `np.arange(0,j)` 取前 j 个频率，`np.exp(1j*m*频率)` 返回 `[j]` 复数相位，再由内层 sum 形成一个复数 S_j；abs 取其模长，不是取实部。外层 sum 把64个非负模长相加，返回 NumPy 浮点标量。
#
# j=0 对应空数组，sum 为0；最后 j=63，不包含完整64项的 S_64。函数不接受 q/k 内容，没有 softmax，所以输出不是模型 attention score。记住这里有两层求和，不是仅求 `abs(sum全部相位)`。
# 相对距离m，统计宽度d默认128；不是模型分数函数。
def s(m, d=128):
    # 匿名函数生成频率；i是通道对索引，虚数单位另写1j。
    theta_i = lambda i: 10000**(-2*i/d)
    # 内sum前j个相位→复数，abs取模，外sum各模；j0..63，不包括S64。
    return np.sum([np.abs(np.sum(np.exp(1j*m*theta_i(np.arange(0, j))))) for j in range(0, d//2)])

# %% 24
# 生成曲线数据
# d=128，xs=np.arange(256) 为 `[256]` int64，取距离0到255。`ys=[s(x)/(d//2) for x in xs]` 逐距离计算，返回长度256的 Python浮点列表，归一化除数为64。
#
# 没有 batch/token/head 张量，不执行模型训练。距离0时所有 exp 项为1，S_j=j，因此 ys[0]=(0+…+63)/64=31.5，已经说明它不是0到1的概率。本次256点中有113次相邻上升，不能称为严格单调衰减。
#
# 全局d与函数s的默认参数是两处配置。s(x)未传d，仍用定义时的默认128；单独将全局d改64后，距离0的统计会变成2016/32=63，而显式s(0,d=64)/32为15.5。本例两者原本都是128，因此原曲线正确，但改宽度时须同步传参。
# 这里只赋值全局d；下一行s(x)仍用函数默认128，改变本变量不改变默认值。
d = 128
# int64距离数组[256]，0..255。
xs = np.arange(256)
# 列表推导逐距离计算统计量再除全局d//2；要改变宽度须同步传给s。
ys = [s(x)/(d//2) for x in xs]

# %% 25
# 画曲线
# `plt.plot(xs,ys)` 输入两个等长序列，返回 Line2D 对象列表并绘图。原单元没有标签、legend 或指标说明，阅读时必须结合前一单元确定纵轴为部分和模长之和/64。
#
# 本篇自己重绘并加入坐标标签，同时绘制单频率点积振荡的反例。曲线展示旋转频率聚合后的变化趋势；具体文本的注意力概率还取决于Q、K的内容及softmax。
# 画部分和模长统计量，C0=31.5不是概率；实际曲线有振荡。
plt.plot(xs, ys)

# %% 26
# 空代码单元
# source 为空，无 import、计算、输出或副作用。覆盖统计保留它，避免把“全部代码单元”悄悄改成只统计有内容的单元。
# 原空单元保留，不添加计算。
