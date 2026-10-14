# -*- coding: utf-8 -*-
"""Complete27-cell notes and comment-only derivatives; legacy43-cell builder is retained."""
import ast
import copy
import hashlib
import json
from pathlib import Path

CODE_DETAILS={1: '【必须会】原代码导入 NumPy 为 np、matplotlib.pyplot 为 plt、torch、torch.nn 为 nn、typing.Tuple。`import ... as ...` '
    '给模块短名称，不创建模型实例。\n'
    '\n'
    'np 用于后面的旋转矩阵与复数求和；plt 用于绘图；torch 用于 Tensor 旋转；Tuple 用于返回类型注解。nn 在本 notebook '
    '后续没有使用，可以识别为未使用导入，不能据此猜测这里定义了神经网络类。导入没有 Tensor shape，也没有模型权重或训练动作。',
 6: '【必须会】这是普通 Python 函数，不属于类，没有 self。输入 theta 在这个函数里是**角度制数值**，第一行 `np.radians(theta)` 将度转成弧度。其后 cos/sin '
    '返回标量，`np.array` 将嵌套列表组合成 float64 `[2,2]` 旋转矩阵。\n'
    '\n'
    '返回矩阵的第一行是 `[cos,−sin]`，第二行是 `[sin,cos]`。矩阵乘列向量 `[x,y]` 会得到 `[x cos−y sin,x sin+y cos]`。R(90) 将 `[1,0]` 变为 '
    '`[0,1]`。这里 theta 与官方 precompute 的默认 theta=10000 是不同含义：前者为角度，后者为频率基数。\n'
    '\n'
    '定义函数只登记函数体，此时没有执行旋转。它不修改全局变量或模型参数。',
 7: '【必须会】R1=R(30)、R2=R(60) 实际调用前一单元。二者均为 NumPy float64 `[2,2]`，表示 30° 和 60° 旋转；数字不是 token 位置，也不是弧度。\n'
    '\n'
    '变量名 R1/R2 只是分别保存两次返回的数组，并没有共享可训练权重。后续运算需要这两个变量，单独运行后续单元前应先执行 6–7。',
 8: '【必须会】二维数组的 `.dot` 在此执行矩阵乘法，`[2,2] @ [2,2] → [2,2]`，不是逐元素乘法。将结果作用于一个列向量，先执行右侧 R2 再执行 R1，因此总角度为 60°+30°=90°。\n'
    '\n'
    '历史输出对角线约 2.63e-16，是浮点误差，不是新的角度参数。数学上应是 0。本次脚本按 1e-15 绝对容差与 R(90) 比较通过。',
 9: '【必须会】`.T` 转置 `[2,2]` 数组。旋转矩阵的转置等于逆矩阵，即 R(−30°)。和 R(60°) 相乘得到 R(30°)，返回 `[2,2]`。\n'
    '\n'
    '这是相对位置推导中的关键模式 `R_mᵀ R_n=R_(n−m)`。此处只演示角度差，尚未调用模型注意力。历史结果约为 `[[0.8660254,−0.5],[0.5,0.8660254]]`。',
 10: '【必须会】先用 Python 整数加法得到 90，再调用 R。与单元 8 采用不同计算路径验证角度相加性质。返回 `[2,2]`，不更新 R1/R2。浮点结果不要求与单元 8 逐 bit 相同。',
 11: '【必须会】计算 30° 矩阵，与单元 9 对照，验证转置后的角度相减。减法方向不能交换：R(30−60) 会得到相反旋转，通常不同。',
 16: '【必须会】这是本讲的核心普通函数，三个输入均为 torch.Tensor，返回 Tuple[Tensor,Tensor]。类型注解说明预期类型，不自动转换输入或执行 shape 检查。源码没有默认参数，没有 self，也没有 '
     'nn.Parameter。\n'
     '\n'
     '### 输入与用途\n'
     '\n'
     '输入 xq 是 query，xk 是 key，通常为浮点 `[B,T,Hq,d]` 和 `[B,T,Hkv,d]`；freqs_cis 为 complex64 `[T,d/2]`。Hq 与 Hkv 可以不同，但 T、d '
     '和频率表必须相容。它将每对通道按对应绝对位置旋转，返回两个实数 Tensor，不返回注意力分数。\n'
     '\n'
     '### 两条复数转换语句\n'
     '\n'
     '对 xq 和 xk 分别执行以下步骤。`.float()` 把数据转为 float32；即使输入 float64，这里也先降为 float32。`xq.shape[:-1]` 是去掉最后维度的 shape tuple，例子为 '
     '`(1,8,32)`；`*` 将 tuple 展开成 reshape 的多个参数。\n'
     '\n'
     '`reshape(*xq.shape[:-1],−1,2)` 保留前三轴，将 128 配成 64 对：`[1,8,32,128]→[1,8,32,64,2]`。−1 表示由总元素数推断这一维，不表示负的维度长度。最后的 2 '
     '是每个复数的实部/虚部，不是新的 token 或 head。\n'
     '\n'
     '`torch.view_as_complex` 将末轴 `[a,b]` 解释成 a+ib，shape 变为 `[1,8,32,64]`，dtype 为 complex64。它不是丢掉一半通道，存储仍包含两部分。偶数 d、合适 '
     'stride 是必要条件；本次还实测了最后 stride=2 的切片输入会报错。\n'
     '\n'
     '### 调用 reshape_for_broadcast\n'
     '\n'
     '将 `[8,64]` 表变成 `[1,8,1,64]`，位置轴与 T 对齐，batch/head 轴用长度1广播。**本 notebook 没有定义或导入该函数。** Python '
     '允许先定义引用未知名字的函数；只有调用函数体时才报 NameError。单元 16 的定义成功不证明调用成功。\n'
     '\n'
     '### 两条复数乘法语句\n'
     '\n'
     '`xq_ * freqs_cis` 逐元素复数乘，广播后为 `[1,8,32,64]`。乘以 cos+i sin 等于逐对二维旋转，不是 Q 与 K 相乘。`view_as_real` 恢复 `[1,8,32,64,2]` '
     'float32，`.flatten(3)` 合并从第3轴到最后轴，回到 `[1,8,32,128]`。这里轴从0计数，flatten(3) 的“3”指第四个轴。\n'
     '\n'
     'xk 同样处理。不要加入第18讲重建传统 sin/cos 表时使用的 flip，它会交换结果通道。`.type_as(xq/xk)` 各自转回输入 dtype；`return a,b` 创建含两个张量的 '
     'tuple，调用方可以用 `xq,xk=...` 解包。\n'
     '\n'
     '### 状态、输出与实际补足\n'
     '\n'
     '函数不原地修改输入或频率表，不写 cache；输出创建后交给 Attention。单元 16 没有调用示例。本次先实际捕获缺失辅助函数的 NameError，再把固定版本官方 helper 绑定到 notebook '
     '的命名空间，调用它并与官方函数、独立 sin/cos 运算比较。补足仅用于明确记录的后续测试，没有修改原始单元文本。',
 17: '【建议会】该单元每行以 `#` 开头，因此没有执行计算。它描述参数 dim=4096、n_heads=32 得 head_dim=128，max_seq_len=1024 得 end=2048，最终表 shape 为 '
     '`[2048,64]`。\n'
     '\n'
     '注释中的 `2^12/2^5` 是数学说明；在真正 Python 表达式中 `^` 为按位异或，不是乘方，乘方要用 `**`。不能把注释复制成可执行计算而保留 `^`。注释也不能证明老师真实运行时的最大长度配置。',
 18: '【必须会】从 sentencepiece 导入 SentencePieceProcessor，构造 `sp_model`，用相对路径 `./tokenizer.model` 读取文件。`model_file` '
     '是构造参数名，实例拥有分词模型状态，但不含 Llama Transformer 权重。\n'
     '\n'
     '相对路径根据当前工作目录解析，不自动根据 notebook 所在目录解析；缺文件或路径不兼容会失败。原课程 tutorials 下确有同哈希 '
     'tokenizer。本次复制到临时工作目录中用同一原始单元加载，运行后恢复工作目录，没有改写这个单元。',
 19: '【必须会】prompt 是 Python str：`I believe the meaning of life is`。`sp_model.encode(prompt)` 默认得到 token ID 的 '
     'list[int]，不是浮点 embedding。`bos_id()` 返回开始标记 ID=1。\n'
     '\n'
     '列表表达式 `[bos_id,*encode结果]` 中的 `*` 将一个列表展开成多个元素；得到长度8的列表 `[1,306,4658,278,6593,310,2834,338]`，本次与历史输出一致，没有自动添加 '
     'EOS。\n'
     '\n'
     'shape 可理解为 `[8]` ID 序列，但此时对象仍为 Python list。这里不调用 embedding、Q/K 投影或 apply_rotary_emb；8 个 token 与 markdown 中 '
     'seqlen=8 的例子有关，但不能把这个分词输出当成完成模型旋转。',
 23: '【建议会】m 是相对距离，不是下面 arange 的通道编号；d 默认为128。局部 `theta_i=lambda i:10000**(-2*i/d)` 是匿名函数，输入索引数组 i，返回频率数组。\n'
     '\n'
     '外层列表推导遍历 j=0,…,63。每次 `np.arange(0,j)` 取前 j 个频率，`np.exp(1j*m*频率)` 返回 `[j]` 复数相位，再由内层 sum 形成一个复数 S_j；abs '
     '取其模长，不是取实部。外层 sum 把64个非负模长相加，返回 NumPy 浮点标量。\n'
     '\n'
     'j=0 对应空数组，sum 为0；最后 j=63，不包含完整64项的 S_64。函数不接受 q/k 内容，没有 softmax，所以输出不是模型 attention score。记住这里有两层求和，不是仅求 '
     '`abs(sum全部相位)`。',
 24: '【建议会】d=128，xs=np.arange(256) 为 `[256]` int64，取距离0到255。`ys=[s(x)/(d//2) for x in xs]` 逐距离计算，返回长度256的 '
     'Python浮点列表，归一化除数为64。\n'
     '\n'
     '没有 batch/token/head 张量，不执行模型训练。距离0时所有 exp 项为1，S_j=j，因此 '
     'ys[0]=(0+…+63)/64=31.5，已经说明它不是0到1的概率。本次256点中有113次相邻上升，不能称为严格单调衰减。',
 25: '【建议会】`plt.plot(xs,ys)` 输入两个等长序列，返回 Line2D 对象列表并绘图。原单元没有标签、legend 或指标说明，阅读时必须结合前一单元确定纵轴为部分和模长之和/64。\n'
     '\n'
     '本篇自己重绘并加入坐标标签，同时绘制单频率点积振荡的反例。曲线能说明这一数学统计量的趋势，不能证明特定文本的注意力概率、模型准确率或长上下文表现。',
 26: '【可以暂时不掌握】source 为空，无 import、计算、输出或副作用。覆盖统计保留它，避免把“全部代码单元”悄悄改成只统计有内容的单元。'}

CODE_DETAILS[18]=CODE_DETAILS[18].replace('本次复制到临时工作目录中用同一原始单元加载，运行后恢复工作目录，没有改写这个单元。','早期验证复制到临时目录；新验证直接在正式代码/tutorials目录读取同字节文件。两种方式均执行原单元并恢复工作目录，没有改写该单元。')
CODE_DETAILS[24]+='\n\n全局d与函数s的默认参数是两处配置。s(x)未传d，仍用定义时的默认128；单独将全局d改64后，距离0的统计会变成2016/32=63，而显式s(0,d=64)/32为15.5。本例两者原本都是128，因此原曲线正确，但改宽度时须同步传参。'

NOTES={
0:('位置机制与顺序','【必须会】没有位置机制及其他顺序约束的self-attention对置换等变，不表示Transformer输出是完全相同。causal mask也引入顺序约束；RoPE用乘性旋转显式编码位置。'),
1:('导入库','【必须会】np为数组/矩阵，plt为图；torch为实/复数Tensor和autograd；nn在原课程中没有使用；Tuple为返回值注解，不是检查或转换数据。'),
2:('cis编码小节','【必须会】cis为cos+i sin，位置对应复数单位因子。'),
3:('复数与二维矩阵','【必须会】q=q0+iq1与列向量[q0,q1]对应；乘exp(imθ)等于R(mθ)q。m是位置、θ是频率弧度/位置，不是precompute的基数theta。shape[2]→[2]，原代码度制示例稍后转换。'),
4:('旋转合成与转置','【必须会】R(α)R(β)=R(α+β)；Rᵀ(α)=R(-α)，因此Rᵀ(α)R(β)=R(β-α)。矩阵乘不是逐元素乘，列向量计算从右向左。'),
5:('相对点积','【必须会】固定内容q/k分别在m/n旋转后，点积qᵀR(n-m)k仅显式包含位置差。内容仍决定分数；不能随意将n-m反号。复数形式需内容共轭和取实部。'),
6:('定义 R(theta)','【必须会】函数接收角度制数值，radians转换弧度，cos/sin标量组合为float64[2,2]；无self、训练参数或状态改变。'),
7:('创建 R1 与 R2','【必须会】30°与60°调用返回两个float64[2,2]，不是token位置或预计算base。'),
8:('R1.dot(R2)','【必须会】[2,2]矩阵乘，先R2再R1，结果90°；浮点对角线约0，不要求bit等于另一计算路径。'),
9:('R1.T.dot(R2)','【必须会】转置等于反向30°，乘60°得到正30°；对应n-m，不可交换差值方向。'),
10:('R(30+60)','【必须会】独立构造90°矩阵，与单元8按数值容差比较。'),
11:('R(60−30)','【必须会】独立构造30°矩阵，与单元9按容差比较。'),
12:('核心函数章节','【必须会】接下来只定义旋转函数，不是运行完整attention。'),
13:('模型维度表','【必须会】heads为query头数，dim为D，head_dim=D/Hq，三行均为128。layers不是head数。表没有列K/V头数，不能据此推出xk必须与xq同头数；模型并行时本地头数另计算。'),
14:('先投影再旋转','【必须会】q=wq(x)、k=wk(x)先完成内容投影，再乘旋转因子；不是将传统sin/cos表加到embedding。没有在此处更新权重或计算V的旋转。'),
15:('调用签名与shape','【必须会】示例xq/xk=[1,8,32,128]、freq=[8,64]，B/T/H/d分别为batch、当前位置数、头数、每头通道。tuple返回值可解包；这行在Markdown中展示，没有实际调用函数。'),
16:('定义 apply_rotary_emb','【必须会】普通函数三个Tensor输入，无默认值；Tuple注解不强制检查。FP32成对reshape、view_as_complex、显式频率广播、复数乘法、view_as_real/flatten3、转回dtype；返回两个同shapeTensor。不计算QKᵀ或softmax，不原地写输入、模型权重或cache。原课程缺reshape_for_broadcast，定义成功并不等于可调用。'),
17:('只有注释','【建议会】dim/head与max_seq_len只是注释配置，没有真正precompute。^在Python是异或，幂为**。1024是示例，不是固定官方ModelArgs默认2048。'),
18:('加载 SentencePiece 模型','【必须会】从./tokenizer.model载入分词器，不是Transformer检查点；路径相对cwd。本仓库提供相同字节原文件，启动内核应在代码/tutorials。'),
19:('分词与 BOS','【必须会】str编码为list[int]，星号展开encode的ID再前置bos1；实际8个ID[1,306,4658,278,6593,310,2834,338]，没有自动EOS或embedding/旋转。'),
20:('远距性质章节','【建议会】原文“远程”在这里指相对距离，不是网络请求；单个实际attention分数不保证单调下降。'),
21:('原作者相关推导','【建议会】链接是RoFormer作者的旋转位置推导资料，作为背景；本讲运行代码不等于复现博客或论文所有结论。'),
22:('部分和公式','【建议会】用r替代原式重复的i索引以免和虚数单位混淆；S_j=sum前j个复数相位，j0为空和0。真正代码还对各|S_j|求和，不是直接返回某一个S_j或attention概率。'),
23:('定义 s(m,d=128)','【建议会】lambda产生频率数组；列表推导遍历j0..d/2-1，内部复数sum再abs，外部再sum。默认d128有64项，不包含S64；返回非负NumPy浮点标量，无q/k输入或参数。'),
24:('生成曲线数据','【建议会】xs int64[256]为距离0..255，ys为256个浮点的list，各s(x)除64；全局d128不传入s，s仍用默认128，此例恰好一致。改全局d不改s的默认值可能使统计宽度与归一化分母不一致。'),
25:('画曲线','【建议会】plt.plot返回Line2D列表，修改当前图；纵轴为部分和模长之和除64，而非概率或loss。曲线有局部上升，不能证实严格单调或模型长文本质量。'),
26:('空代码单元','【可以暂时不掌握】原空格/输出保持，不编造计算。'),
}

COMMENTS={
'import numpy as np':'NumPy别名np，用于实矩阵、复数相位与求和。',
'import matplotlib.pyplot as plt':'绘图接口plt，画数学统计量而非训练指标。',
'import torch':'Torch Tensor、复数视图和自动求导。',
'from torch import nn':'nn在后续原课程未使用，不能据此猜测这里建立了模型。',
'from typing import Tuple':'返回类型注解Tuple；不自动验证shape。',
'def R(theta):':'普通函数接收角度制数值，与频率base theta不同。',
'theta = np.radians(theta)':'局部变量由度转为弧度，不修改调用者或全局配置。',
'c, s = np.cos(theta), np.sin(theta)':'同一角度的cos/sin标量。',
'return np.array([[c, -s], [s, c]])':'返回float64[2,2]正角逆时针旋转矩阵。',
'R1 = R(30)':'创建30°矩阵[2,2]。','R2 = R(60)':'创建60°矩阵[2,2]。',
'R1.dot(R2)':'矩阵乘法，列向量先转60°后30°，合计90°。',
'R1.T.dot(R2)':'转置等于逆30°，再与60°合成正30°，对应相对角度。',
'R(30+60)':'另一计算路径构造90°，按浮点容差与矩阵乘结果比较。',
'R(60-30)':'构造正30°，减法方向不能交换。',
'def apply_rotary_emb(':'普通前向函数，无参数或缓存状态；本格只定义，未实际调用。',
'xq: torch.Tensor,':'query浮点Tensor[B,T,Hq,d]，d必须偶数。',
'xk: torch.Tensor,':'key浮点Tensor[B,T,Hkv,d]，头数可以不同，T/d须相容。',
'freqs_cis: torch.Tensor,':'complex64[T,d/2]频率切片，已包含绝对位置，不是原base。',
') -> Tuple[torch.Tensor, torch.Tensor]:':'类型提示返回两个Tensor的tuple，不做运行时shape验证。',
'xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))':'FP32→保留B/T/H、末轴配对[...,d/2,2]→complex64[B,T,Hq,d/2]；星号展开shape。',
'xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))':'同理转换key，Hkv可不同；复数view还要求合适stride。',
'freqs_cis = reshape_for_broadcast(freqs_cis, xq_)':'原notebook未定义此helper，调用时NameError；官方helper补成[1,T,1,d/2]。',
'xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)':'逐元素复数乘旋转，real恢复末轴2，flatten第3轴起→[B,T,Hq,d]；不加flip。',
'xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)':'key同理，仍不计算QKᵀ、softmax或cache写入。',
'return xq_out.type_as(xq), xk_out.type_as(xk)':'各自转回输入dtype并组成tuple；局部重绑定不修改原输入。',
'from sentencepiece import SentencePieceProcessor':'加载分词器类，而不是语言模型权重。',
"sp_model = SentencePieceProcessor(model_file='./tokenizer.model')":'按cwd相对路径读原499723byte分词模型；本仓库提供相同字节文件。',
"prompt = 'I believe the meaning of life is'":'待编码Python字符串，未调用embedding。',
'[sp_model.bos_id(), *sp_model.encode(prompt)]':'BOS ID1前置，星号解包encode列表→8个整数；不添加EOS。',
'def s(m, d=128):':'相对距离m，统计宽度d默认128；不是模型分数函数。',
'theta_i = lambda i: 10000**(-2*i/d)':'匿名函数生成频率；i是通道对索引，虚数单位另写1j。',
'return np.sum([np.abs(np.sum(np.exp(1j*m*theta_i(np.arange(0, j))))) for j in range(0, d//2)])':'内sum前j个相位→复数，abs取模，外sum各模；j0..63，不包括S64。',
'd = 128':'这里只赋值全局d；下一行s(x)仍用函数默认128，改变本变量不改变默认值。',
'xs = np.arange(256)':'int64距离数组[256]，0..255。',
'ys = [s(x)/(d//2) for x in xs]':'列表推导逐距离计算统计量再除全局d//2；要改变宽度须同步传给s。',
'plt.plot(xs, ys)':'画部分和模长统计量，C0=31.5不是概率；实际曲线有振荡。',
}

def annotate(s,i):
    out=[]
    if i==17:out.append('# 以下全为历史配置注释，不执行；^是说明符号，Python幂要用**。\n')
    if not s.strip():out.append('# 原空单元保留，不添加计算。\n')
    for line in s.splitlines(True):
        key=line.partition('#')[0].strip()
        if key:
            assert key in COMMENTS,key
            out.append(line[:len(line)-len(line.lstrip())]+'# '+COMMENTS[key]+'\n')
        out.append(line)
    return ''.join(out)

def official_reading(root):
    path=Path('G:/zzz/Personal ChatGPT 系列/19 RoPE apply_rotary_emb/资料/llama-model.py')
    data=path.read_bytes();assert hashlib.sha256(data).hexdigest()=='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    text=data.decode('utf-8');original_lines=text.splitlines()
    names=['precompute_freqs_cis','reshape_for_broadcast','apply_rotary_emb']
    details={
        'def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):':'dim=head_dim，end位置数量，theta默认10000为频率基数，不是角度。',
        'freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))':'频率[dim/2]，指数必须除dim；dim偶数由调用方保证。',
        't = torch.arange(end, device=freqs.device)':'位置0..end-1与频率同设备。',
        'freqs = torch.outer(t, freqs).float()':'频率变量改绑为相位[end,dim/2]。',
        'freqs_cis = torch.polar(torch.ones_like(freqs), freqs)':'半径1/弧度相位→complex64[end,dim/2]。',
        'return freqs_cis':'返回单位复数因子，无参数更新。',
        'def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):':'原课程缺少的helper，接收[T,P]表与[B,T,H,P]复数张量。',
        'ndim = x.ndim':'预期x.ndim为4；通用helper并不代替apply的四维输入约束。',
        'assert 0 <= 1 < ndim':'确认位置轴1存在；Python优化模式可能去掉assert。',
        'assert freqs_cis.shape == (x.shape[1], x.shape[-1])':'核验位置T和通道对P，不能让T静默对齐head轴。',
        'shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]':'enumerate遍历轴；保留位置/最后轴，其余设1→[1,T,1,P]。',
        'return freqs_cis.view(*shape)':'星号解包shape，返回广播视图，共享频率存储。',
    }
    details.update(COMMENTS)
    lines=['# -*- coding: utf-8 -*-','# Copyright (c) Meta Platforms, Inc. and affiliates.',
        '# 原源码适用Llama 2 Community License Agreement。',
        '# 固定来源：https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py',
        '# 三个官方函数执行语句保留，英文docstring省去并加入中文注释；不是原notebook单元。',
        '# 实际验证加载完整未修改官方文件，原notebook缺helper的调用错误单独记录。',
        'from typing import Tuple','import torch','']
    for node in ast.parse(text).body:
        if not isinstance(node,ast.FunctionDef) or node.name not in names:continue
        doc=node.body[0] if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str) else None
        for number in range(node.lineno,node.end_lineno+1):
            if doc and doc.lineno<=number<=doc.end_lineno:continue
            line=original_lines[number-1];key=line.partition('#')[0].strip()
            if key in details:lines.append(line[:len(line)-len(line.lstrip())]+'# '+details[key])
            lines.append(line.rstrip())
        lines.append('')
    (root/'代码/RoPE_官方函数阅读.py').write_bytes(('\n'.join(lines).rstrip()+'\n').encode('utf-8'))

def main():
    root=Path(__file__).resolve().parents[2];folder=root/'代码/tutorials'
    raw=json.loads((folder/'llama2_src_03_RoPE_apply_rotary_emb.ipynb').read_bytes())
    assert len(raw['cells'])==27 and set(NOTES)==set(range(27))
    for changed,name in [(False,'原代码_带说明.ipynb'),(True,'代码注释.ipynb')]:
        n=copy.deepcopy(raw);n['cells']=[dict(cell_type='markdown',metadata={},id='lesson19-intro',source=['# 第19讲说明与注释\n','原函数缺helper且未调用；明确补足后的测试另记，不改原语句或历史输出。\n'])]
        for i,c in enumerate(raw['cells']):
            title,detail=NOTES[i];note='## 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
            n['cells'].append(dict(cell_type='markdown',metadata={},id='lesson19-note-'+str(i),source=note.splitlines(True)))
            saved=copy.deepcopy(c)
            if changed and c['cell_type']=='code':saved['source']=annotate(''.join(c['source']),i).splitlines(True)
            n['cells'].append(saved)
        (folder/name).write_bytes((json.dumps(n,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    doc='# 第19讲 notebook 全部原单元说明\n\n[主解释](解释.md) · [阅读入口](README.md)\n\n全部27格、15代码格（含注释17与空26）均有说明；原公式、模型表、分词文件和历史输出保留。原函数缺helper，补足验证另记。\n\n'
    py=['# -*- coding: utf-8 -*-','# 第19讲原代码和详注；普通脚本不自动显示裸表达式，helper缺失保留。','']
    for i,c in enumerate(raw['cells']):
        title,detail=NOTES[i]
        heading='## 单元 {}：{}\n\n'.format(i,title) if c['cell_type']=='code' else '## Markdown 单元 {}：{}\n\n'.format(i,title)
        text=CODE_DETAILS.get(i,detail)+'\n'
        doc+=heading+text+'\n';py+=['# %% '+str(i),'# '+title]+['# '+l for l in text.splitlines()]
        s=''.join(c['source'])
        if c['cell_type']=='code':doc+='```python\n'+annotate(s,i)+'\n```\n\n';py+=annotate(s,i).splitlines()
        else:doc+=s+'\n\n';py+=['# '+l for l in s.splitlines()]
        py+=['']
    doc+='## 覆盖检查\n\n27个原单元/15代码格详注，两派生版各55格。早期只为15代码格插说明的43格版本与覆盖报告保留为历史记录；新完整验证见复现目录。\n'
    (root/'代码单元解释.md').write_bytes(('\n'.join(l.rstrip() for l in doc.splitlines()).rstrip()+'\n').encode('utf-8'))
    (folder/'llama2_src_03_RoPE_apply_rotary_emb_代码注释.py').write_bytes(('\n'.join(l.rstrip() for l in py).rstrip()+'\n').encode('utf-8'))
    official_reading(root)
    print('27 notes/15 code, two55-cell derivatives, Python reading and three official function comments.')

if __name__=='__main__':main()
