# -*- coding: utf-8 -*-
# 原课程代码中文注释阅读版；不是直接运行原GPU/远程权重的承诺。

# %% 原代码单元 1
# 【必须会】Image只负责展示，不是矩阵分解库。
# 从指定包导入本格需要的真实API；导入不等于加载远程权重。
from IPython.display import Image

# %% 原代码单元 2
# 【必须会】远程LoRA图：原URL和HTML引用保留；未下载图片，未验证远端图像内容。新增解释提供可直接读的流程图。
# 返回远程图片HTML引用，width仅为显示宽度。
Image(url='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png', width=500)

# %% 原代码单元 3
# 【必须会】导入torch并固定seed0，影响后续随机矩阵。本次原版/注释版各自从该格重新设seed。
# 导入模块命名空间供后续使用。
import torch
# 固定后续torch随机数序列，不固定所有硬件算法。
torch.random.manual_seed(0)

# %% 原代码单元 4
# 【必须会】d,n分别是输出/输入维度100；low_rank20是构造与截断使用的目标秩，不是batch或token数。
# Define dimensions and low rank
# 定义矩阵输出/输入维度，不是batch或序列长度。
d, n = 100, 100
# 构造低秩矩阵及截断SVD所用目标秩。
low_rank = 20

# %% 原代码单元 5
# 【必须会】两个随机float32矩阵[100,20]@[20,100]得到W[100,100]，数学秩至多20。随机实数下通常20，数值秩还受dtype和容差影响。
# Create a low rank matrix W
# lora_b @ lora_a
# 两个窄矩阵相乘，所得矩阵秩至多low_rank。
W = torch.randn(d, low_rank) @ torch.randn(low_rank, n)

# %% 原代码单元 6
# 【必须会】matrix_rank使用奇异值容差判断数值秩。本次真实得到20，不把历史结果当新结果。
# 按奇异值容差计算数值秩，不是精确符号代数秩。
torch.linalg.matrix_rank(W)

# %% 原代码单元 8
# 【必须会】第二个远程图片展示，width400只影响HTML布局。
# 返回远程图片HTML引用，width仅为显示宽度。
Image(url='https://www.statlect.com/images/matrix-product-and-rank__5.png', width=400)

# %% 原代码单元 10
# 【必须会】torch.linalg.svd(W,full_matrices=True)返回U[100,100]、S[100]、Vh[100,100]，Vh已是V的共轭转置。float32 CPU实际执行。
# 返回奇异向量/奇异值；full=True在大矩形词表上很占内存。
U, S, Vh = torch.linalg.svd(W, full_matrices=True)

# %% 原代码单元 11
# 【必须会】只展示三个shape，方阵容易掩盖矩形full/reduced SVD尺寸差异，解释中另外推导。
U.shape, S.shape, Vh.shape

# %% 原代码单元 12
# 【必须会】diag把奇异值向量变对角矩阵；U@diag(S)@Vh重建W；torch.dist默认Frobenius距离，数值误差不是严格0。
# 默认计算Frobenius距离；浮点重建不要求严格零。
torch.dist(U @ torch.diag(S) @ Vh, W)

# %% 原代码单元 13
# 【必须会】切U前20列/S前20项/Vh前20行，再把S乘进U。变量U此后代表U20Σ20，不再是正交矩阵；重建用U@Vh，不能再乘一次S。
# 保留左奇异向量前r列；切片通常是view。
U = U[:, :low_rank]
# 保留前r个最大奇异值。
S = S[:low_rank]
# 把奇异值吸收进左因子；此后U不再正交。
U = U @ torch.diag(S)
# 保留右奇异向量前r行，对应新A。
Vh = Vh[:low_rank, :]

# %% 原代码单元 14
# 【必须会】比较截断重建误差。对构造的秩20矩阵应很小；不意味着任意100×100矩阵都能由秩20无损表示。
# 默认计算Frobenius距离；浮点重建不要求严格零。
torch.dist(U @ Vh, W)

# %% 原代码单元 16
# 【必须会】randn_like保持W的shape/device/dtype，重新生成独立随机W2，数学上几乎必然满秩。
# 构造相同shape/dtype的独立随机满秩对照矩阵。
W2 = torch.randn_like(W)

# %% 原代码单元 17
# 【必须会】数值秩本次100；不是把low_rank变量改成100。
# 按奇异值容差计算数值秩，不是精确符号代数秩。
torch.linalg.matrix_rank(W2)

# %% 原代码单元 18
# 【必须会】对满秩W2做完整SVD，重新覆盖U/S/Vh三个变量。
# 返回奇异向量/奇异值；full=True在大矩形词表上很占内存。
U, S, Vh = torch.linalg.svd(W2, full_matrices=True)

# %% 原代码单元 19
# 【必须会】完整100项重建W2应只有浮点误差，不能用此误差评价语言模型。
# 默认计算Frobenius距离；浮点重建不要求严格零。
torch.dist(U @ torch.diag(S) @ Vh, W2)

# %% 原代码单元 20
# 【必须会】只保留20项后有明显误差。最优秩20近似仍会丢弃剩余80项；其Frobenius误差平方等于被丢弃奇异值平方和。
# 保留左奇异向量前r列；切片通常是view。
U = U[:, :low_rank]
# 保留前r个最大奇异值。
S = S[:low_rank]
# 把奇异值吸收进左因子；此后U不再正交。
U = U @ torch.diag(S)
# 保留右奇异向量前r行，对应新A。
Vh = Vh[:low_rank, :]

# 默认计算Frobenius距离；浮点重建不要求严格零。
torch.dist(U @ Vh, W2)
