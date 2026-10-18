# -*- coding: utf-8 -*-
# 第18讲 overview 原代码与中文注释；错误原语句保留，修正另验证。

# %% 0
# ### 单元 0（code）：导入展示与Torch
#
# Image读取配图，不执行注意力；torch/nn提供位置表计算。
# 导入Image类，展示后面的配图。
from IPython.display import Image
# 导入PyTorch，使用实数、复数张量和自动求导。
import torch
# 导入Module基类，供位置表类继承。
from torch import nn

# %% 1
# ### 单元 1（markdown）：基础章节
#
# 整理位置机制背景。
# ## basics

# %% 2
# ### 单元 2（markdown）：RoFormer来源
#
# 固定v4论文的二维/多维旋转；博客仅辅助，模型依赖不能与bert4keras路线混用。
# - RoPE: rotary position embeddings
#     - https://arxiv.org/pdf/2104.09864v4.pdf
#         - ROFORMER, Jianlin Su
#     - 属于 position embedding 的范畴
#     - 通过旋转矩阵（rotation matrix）编码绝对位置信息
# - 参考
#     - https://blog.eleuther.ai/rotary-embeddings/
#     - https://sh-tsang.medium.com/brief-review-roformer-enhanced-transformer-with-rotary-position-embedding-36f67a619442

# %% 3
# ### 单元 3（markdown）：相对点积公式纠正
#
# 分数仍依赖内容q/k；复数点积必须Re(zq*conj(zk))，原简写省略内容共轭与取实部。m-n和n-m要随约定一致。0<epsilon<2π/N不是Meta多频率函数的限制，不能据此断言可无限外推。
# - position encoding：位置编码核心要做的事情
#     - token embedding $\mathbf x$, position $\ell$，其 position embedding 为
#     $$
#     f(\mathbf{x}, \ell)
#     $$
#     - 如另有两个 token embedding 分别为 $f(\mathbf q, m)$ 与 $f(\mathbf k, n)$，其 inner product 之后的结果应当只与 $m-n$ 有关（relative position）
#
# - RoPE
#     - 序列长度为 $N$，任取 $0\lt \varepsilon \lt \frac{2\pi}N$
#
# $$
# \begin{align}
# \mathrm{RoPE}(x, m)&=xe^{mi\varepsilon}\\
# \langle\mathrm{RoPE}(q_j, m), \mathrm{RoPE}(k_j, n)\rangle &=\langle q_je^{mi\varepsilon},k_je^{ni\varepsilon} \rangle \\
# &=q_jk_j e^{mi\varepsilon} \overline{e^{ni\varepsilon}}\\
# &=q_jk_je^{(m-n)i\varepsilon}\\
# &=\mathrm{RoPE}(q_jk_j, m-n)
# \end{align}
# $$

# %% 4
# ### 单元 4（markdown）：sin位置表章节
#
# 传统表作为比较，不是已应用RoPE。
# ## sin encoding

# %% 5
# ### 单元 5（markdown）：共享频率公式
#
# 偶数通道i用sin、奇数通道i用cos且频率索引i-1；分母指数必须除d_model。
# $$
# \begin{split}
# PE(t,2i)&=\sin(\frac{t}{10000^{\frac{2i}{d_{model}}}})\\
# PE(t,2i+1)&=\cos(\frac{t}{10000^{\frac{2i}{d_{model}}}})\\
# \Downarrow\\
# PE(t,i)&=\sin(\frac{t}{10000^{\frac{i}{d_{model}}}}), \quad \text{i is even}\\
# PE(t,i)&=\cos(\frac{t}{10000^{\frac{i-1}{d_{model}}}}), \quad \text{i is odd}\\
# \end{split}
# $$

# %% 6
# ### 单元 6（code）：配置
#
# 位置10、通道6为小例子，不是7B的配置。
# 位置数量10，对应0..9，端点不包含10。
max_sequence_length = 10
# 教学通道数6；真实模型频率dim应使用head_dim。
d_model = 6

# %% 7
# ### 单元 7（code）：奇偶通道
#
# float32[3]偶数/奇数索引，打印值；后续频率配对。
# 偶数通道0/2/4，float32[3]；右端不含。
even_i = torch.arange(0, d_model, 2).float()
# 展示偶数通道索引，不修改数值。
print(even_i)
# 奇数通道1/3/5，float32[3]。
odd_i = torch.arange(1, d_model, 2).float()
# 展示奇数通道索引。
print(odd_i)

# %% 8
# ### 单元 8（code）：二维位置列
#
# arange→[10]，reshape(-1,1)→float32[10,1]，-1推断第一轴；用于和[3]广播。主notebook改为一维outer，二者不是同一shape。
# 位置列[10,1]，-1推断10，用广播计算[10,3]。
position = torch.arange(max_sequence_length, dtype=torch.float).reshape(-1, 1)
# Notebook展示当前位置tensor；普通Python不自动显示。
position

# %% 9
# ### 单元 9（code）：广播计算sin/cos
#
# 位置[10,1]/频率分母[3]→[10,3]角度，再逐元素三角函数；不需要outer但公式一致。
# [10,1]和[3]广播→[10,3]，先按频率产生角度再sin。
even_pe = torch.sin(position / torch.pow(10000, even_i/d_model))
# 奇数列使用odd_i-1的配对频率，输出[10,3]。
odd_pe = torch.cos(position / torch.pow(10000, (odd_i - 1)/d_model))

# %% 10
# ### 单元 10（code）：查看sin表
#
# even_pe float32[10,3]，不改变对象。
# 展示sin表，不更新对象。
even_pe

# %% 11
# ### 单元 11（code）：查看cos表
#
# odd_pe float32[10,3]；位置0全1。
# 展示cos表，不更新对象。
odd_pe

# %% 12
# ### 单元 12（code）：堆叠shape
#
# stack dim2→[10,3,2]；shape是torch.Size对象，不是数值表。
# 新轴2堆叠→[10,3,2]，末轴[sin,cos]。
stacked = torch.stack([even_pe, odd_pe], dim=2)
# 查看torch.Size([10,3,2])，不是数值归约。
stacked.shape

# %% 13
# ### 单元 13（code）：展平通道
#
# flatten轴1到2→[10,6]按sin/cos交错，不合并位置轴。
# 交错展平→[10,6]，没有求和或丢数据。
torch.flatten(stacked, start_dim=1, end_dim=2)

# %% 14
# ### 单元 14（code）：广播位置类
#
# 无可学习参数，每次forward创建位置[10,1]、broadcast sin/cos、stack/flatten；与主实数类公式一致但实现路径不同。
# 定义位置表模块；同名重定义只改变名称指向，不改变旧对象。
class SinPositionEncoding(nn.Module):
    # 构造配置：位置数、偶数通道数、默认频率基数10000；没有Parameter。
    def __init__(self, max_sequence_length, d_model, base=10000):
        # 初始化Module基类；self表示当前实例。
        super().__init__()
        # 保存位置数量，forward每次重新生成表。
        self.max_sequence_length = max_sequence_length
        # 保存实例的通道配置；与全局d_model是不同变量。
        self.d_model = d_model
        # 保存频率基数，既非逐位置角度也非训练参数。
        self.base = base
    # 无输入张量，返回位置表；不缓存buffer或更新参数。
    def forward(self):
        # 按实例宽度生成偶数通道索引；奇数宽度会多一个通道对。
        even_i = torch.arange(0, self.d_model, 2).float()
        # 按实例宽度生成奇数索引；奇数宽度与even列数不等。
        odd_i = torch.arange(1, self.d_model, 2).float()
        # float32位置列[T,1]，用广播与频率[通道对]组合。
        position = torch.arange(self.max_sequence_length, dtype=torch.float).reshape(-1, 1)
        # 广播位置列与实例频率分母得到sin[T,d/2]。
        even_pe = torch.sin(position / torch.pow(self.base, even_i/self.d_model))
        # 广播cos表；odd_i-1与偶数列频率一致。
        odd_pe = torch.cos(position / torch.pow(self.base, (odd_i-1)/self.d_model))
        # 新轴2堆叠→[10,3,2]，末轴[sin,cos]。
        stacked = torch.stack([even_pe, odd_pe], dim=2)
        # 输出传统交错表[T,d]；奇数维在stack处已失败。
        return torch.flatten(stacked, start_dim=1, end_dim=2)

# %% 15
# ### 单元 15（code）：调用位置类
#
# 创建spe并返回[10,6]，没有传入embedding。
# 创建实例，不自动forward；后面同名类重新创建新对象。
spe = SinPositionEncoding(max_sequence_length=10, d_model=6)
# 通过Module调用forward，返回传统位置表[10,6]。
spe()

# %% 16
# ### 单元 16（markdown）：RoPE章节
#
# 下面定位旋转结构和预计算函数。
# ## RoPE

# %% 17
# ### 单元 17（markdown）：二维旋转矩阵
#
# R(mθ)是[2,2]正交矩阵，负sin在右上，正sin在左下，列向量约定正角逆时针。
# - 2d rotation matrix
#
# $$
# \begin{pmatrix}
# \cos(m\theta), -\sin(m\theta)\\
# \sin(m\theta), \cos(m\theta)
# \end{pmatrix}
# $$

# %% 18
# ### 单元 18（markdown）：先投影再按通道对旋转
#
# x→Wq/Wk投影再R，矩阵乘次序不可任意交换；多维R为d/2个2×2块，[d,d]。每对独立θj。实现使用成对乘加/复数乘法，不必分配稀疏大矩阵。
# - 如下将 2d $\mathbf x$ 旋转 $m\theta$（$\theta$ 表示刻度，）
#     - 这里将 query 和 key 已经 position encoding 组装在一起
#
# $$
# f_{\{q,k\}}(\mathbf x, m)=\begin{pmatrix}
# \cos(m\theta), -\sin(m\theta)\\
# \sin(m\theta), \cos(m\theta)
# \end{pmatrix}\begin{pmatrix}
# W^{(11)}_{q,k},W^{(12)}_{q,k}\\
# W^{(21)}_{q,k},W^{(22)}_{q,k}
# \end{pmatrix}\begin{pmatrix}
# x_{m}^{(1)}\\
# x_{m}^{(2)}
# \end{pmatrix}
# $$
#
# - 一个 general 的版本
#
# $$
# f_{\{q,k\}}(\mathbf x_m, m)=R^{d}_{\Theta, m}W_{q,k}x_m
# $$
#
# - 其中 $R^{d}_{\Theta,m}\in \mathbb R^{d\times d}$ 定义为：
#
# $$
# R^{d}_{\Theta,m}=\begin{pmatrix}
# \cos(m\theta_1)& -\sin(m\theta_1)& 0 & 0 & \cdots & 0 & 0\\
# \sin(m\theta_1)&  \cos(m\theta_1)& 0 & 0 & \cdots & 0 & 0\\
# 0 & 0 & \cos(m\theta_2)& -\sin(m\theta_2)& \cdots & 0 & 0\\
# 0 & 0 & \sin(m\theta_2)&  \cos(m\theta_2)& \cdots & 0 & 0\\
# \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
# 0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2})& -\sin(m\theta_{d/2}) \\
# 0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2})& \cos(m\theta_{d/2})
# \end{pmatrix}
# $$
#
# - $R^{d}_{\Theta,m}\mathbf x$ 的一种高效形式

# %% 19
# ### 单元 19（code）：原配图中的实数实现
#
# Image从../imgs/hf_llama_rope.png读取600像素宽截图。图中rotate_half把末维前后两半配对，cat(-x2,x1)；apply_rotary_pos_emb将cos/sin按position_ids选取后unsqueeze轴1，广播到Q/K。此半区布局与Meta相邻通道复数配对不同，需要一致的通道排列；不能把图当成箭头示意或混用频率布局。转录详注见代码/配图_RoPE源码阅读.py，来源是此固定截图，未找到截图对应的HF版本。
# 读取保留的原配图，宽度600只控制显示；须在tutorials目录运行。
Image('../imgs/hf_llama_rope.png', width=600)

# %% 20
# ### 单元 20（markdown）：频率函数章节
#
# 先生成固定因子，实际旋转调用归第19讲。
# ### precompute_freqs_cis

# %% 21
# ### 单元 21（markdown）：Markdown源码与公式不一致
#
# 六行函数含device=freqs.device与/div dim的正确频率。下方公式省略/dim且把端点写成包含2*max_seq_len，均需纠正；正确索引0..end-1。complex64不减少两实通道的字节数。4096/32=128是head_dim，预留2048×64不是两倍上下文质量保证。源码块在Markdown不执行，仍加逐行注释。
# ```
# # 普通函数：dim为偶数通道，end位置数量，theta频率base默认10000；不旋转Q/K。
# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     # 偶数索引除dim作指数→幂倒数频率[dim/2]；//整数除法，不能漏/div dim。
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     # int64[end]位置，0..end-1；与频率同设备，type ignore不影响运行。
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     # 外积把频率变为相位[end,dim/2]，变量freqs被复用。
#     freqs = torch.outer(t, freqs).float()  # type: ignore
#     # 半径1、角度相位→complex64[end,dim/2]，实部cos/虚部sin。
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     # 返回旋转因子，未应用到任何Q/K。
#     return freqs_cis
# ```
#
# $$
# \begin{split}
# &\text{freqs} = \frac{1}{10000^{2i}} \quad &\in \mathbb{R}^{\frac{\text{dim}}{2} = 64} \\
# &t = 0, 1, 2, \cdots, 2 \times \text{max\_seq\_len} \quad &\in \mathbb{R}^{2 \times \text{max\_seq\_len} = 2048} \\
# &\text{freqs} \in \mathbb{R}^{2048 \times 64}
# \end{split}
# $$
#
# - freqs: 行表示position，列表示 dim
# - `torch.polar(abs, angle)`
#     - $\text{out} = \text{abs} \cdot \cos(\text{angle}) + \text{abs} \cdot \sin(\text{angle}) \cdot j$
#     - 实部是 `cos`，虚部是 `sin`，压缩了一倍的存储
#
#     - 关于复数及复数乘法
#
#     $$
#     \begin{split}
#     &(a+bi)(c+di)=(ac-bd)+(bc+ad)i\\
#     &(\cos\theta_1+\sin\theta_1i)(\cos\theta_2+\sin\theta_2i)=(\cos\theta_1\cos\theta_2-\sin\theta_1\sin\theta_2)+(\sin\theta_1\cos\theta_2+\cos\theta_1\sin\theta_2)i
#     \end{split}
#     $$
#
# - llama2-7b
#     - dim: 4096, head: 32
#     - precompute_freqs_cis:
#         - dim: 4096/32 = 128, 每一个head的dim
#         - end：1024\*2 = 2048，1024: max seq len
#         - freq_cis.shape
#             - (1024\*2) \* (128/2): 2048 \* 64

# %% 22
# ### 单元 22（code）：重复导入
#
# 再次import torch使用已导入模块，不重新建立模型。
# 导入PyTorch，使用实数、复数张量和自动求导。
import torch

# %% 23
# ### 单元 23（code）：CPU示例频率函数
#
# 与正确公式一致，t未显式device，依赖默认设备；本次CPU可运行但不能声称GPU通用。返回[end,dim/2] complex64，无shape打印或状态更新。
# 普通函数：dim为偶数通道，end位置数量，theta频率base默认10000；不旋转Q/K。
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 偶数索引除dim作指数→幂倒数频率[dim/2]；//整数除法，不能漏/div dim。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # CPU例未显式device，不能视为任意默认设备配置下都安全。
    t = torch.arange(end)  # type: ignore
    # 外积把频率变为相位[end,dim/2]，变量freqs被复用。
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 半径1、角度相位→complex64[end,dim/2]，实部cos/虚部sin。
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # 返回旋转因子，未应用到任何Q/K。
    return freqs_cis

# %% 24
# ### 单元 24（code）：生成5行正确表
#
# dim6/end5→[5,3]，与主例end10区别；end是数量，最后位置4。
# 小配置6通道、5位置，与主notebook的end10不同。
dim, end = 6, 5
# 正确函数返回[5,3]，不旋转输入内容。
freqs_cis = precompute_freqs_cis(dim, end)
# 显示复数因子表，不是学习得到的embedding。
freqs_cis

# %% 25
# ### 单元 25（code）：遗漏维度分母反例
#
# 原手写频率1/(10000**[0,2,4])为[1,1e-8,1e-16]，遗漏/dim，不等价于正确函数。theta为int，整数幂后倒数按Torch类型转换；polar产生[5,3]。保留错误原代码用于比较，不偷偷修正。
# 手算例频率基数，整数。
theta = 10000
# 原错误省略/dim，频率变成1/1e8/1e16，不等于正确表；这里变量其实是角度。
my_freqs_cis = torch.outer(torch.arange(end), 1./(theta ** (torch.arange(0, 6, 2))))
# 按错误频率生成[5,3]单位复数；模长1或shape对不代表频率正确。
complex_tensor = torch.polar(torch.ones_like(my_freqs_cis), my_freqs_cis)
# 显示原错误频率例，注释版不偷偷修正。
complex_tensor

# %% 26
# ### 单元 26（code）：查看错误表shape
#
# torch.Size([5,3])与正确表相同不能证明值正确。
# 查看[5,3]，形状与正确表一致但数值不同。
complex_tensor.shape

# %% 27
# ### 单元 27（code）：查看实虚视图
#
# 错误complex_tensor的view_as_real为float32[5,3,2]，末轴实/虚；shape正确不能修复频率。
# 共享存储的float32[5,3,2]实/虚视图，没有修复错误频率。
torch.view_as_real(complex_tensor)

# %% 28
# ### 单元 28（code）：空格
#
# 不包含计算，保留原空单元和编号。
# 原空代码格保留，不添加计算。
