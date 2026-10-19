# CoPE/SelfAttn完整实现，来自原单元39，对照论文2405.18719v2附录B Listing1。
# 作者Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar，FAIR at Meta。
# 独立 Python 阅读版补充了导入和中文注释，方便运行与追踪这段课程实现；来源记录见本讲复现资料。
import math
import torch
from torch import nn

# 中文说明：nn.Module子类，负责位置偏置；类定义本身还没有建立实例。
class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        # 中文说明：先初始化Module，之后赋给属性的Parameter/子Module才能被正常登记。
        super().__init__()
        # 中文说明：保存整数位置表槽数P；合法索引0到P-1，不等于最大token长度。
        self.npos_max = npos_max
        # 中文说明：注册可训练表[1,D,P]，初始全0；不是nn.Embedding对象但同样由optimizer发现。
        self.pos_emb = nn.parameter.Parameter(torch.zeros(1, head_dim, npos_max))

    # 中文说明：query[B,Q,D]和已缩放且已mask的attn_logits[B,Q,K]，返回位置bias同[B,Q,K]。
    def forward(self, query, attn_logits):
        # compute positions
        # 中文说明：sigmoid把每个内容分数映到(0,1)；屏蔽位置logit=-inf时门值为0。
        gates = torch.sigmoid(attn_logits)
        # 中文说明：右到左累积：pos[...,j]=sum_{k=j}^{K-1}gate[...,k]；因果mask使未来贡献0。
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        # 中文说明：上界P-1防止gather越界；超过上界的位置都落到末槽，距离区分会饱和。
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        # 中文说明：向上取整再转int64索引；floor/ceil索引本身不可微。
        pos_ceil = pos.ceil().long()
        # 中文说明：向下取整槽；小数权重仍从浮点pos保留梯度。
        pos_floor = pos.floor().long()
        # 中文说明：[B,Q,D]乘[1,D,P]广播成[B,Q,P]；先点积再插值避免四维位置向量。
        logits_int = torch.matmul(query, self.pos_emb)
        # 中文说明：按[B,Q,K]索引取右槽分数，输出[B,Q,K]。
        logits_ceil = logits_int.gather(-1, pos_ceil)
        # 中文说明：取左槽分数；两个gather共享query和位置表的梯度。
        logits_floor = logits_int.gather(-1, pos_floor)
        # 中文说明：小数部分在[0,1)，pos恰为整数时两个索引相等，此代码选择的门梯度可能为0。
        w = pos - pos_floor
        # 中文说明：线性插值，前向等价于先插值embedding再与query点积。
        return logits_ceil * w + logits_floor * (1 - w)

# 中文说明：单head注意力示例，Q/K/V由调用者传入，不在类中训练投影。
class SelfAttn(nn.Module):
    def __init__(self, npos_max, head_dim):
        # 中文说明：先初始化Module，之后赋给属性的Parameter/子Module才能被正常登记。
        super().__init__()
        # 中文说明：注册CoPE子模块，SelfAttn.parameters会包含其位置表。
        self.cope = CoPE(npos_max, head_dim)
        # 中文说明：保存Q/K的最后维度D，用sqrt(D)缩放内容分数。
        self.head_dim = head_dim

    # 中文说明：query/key/val为三维batch张量；mask须是允许=1/屏蔽=0的可广播张量，不是加性0/-inf mask。
    def forward(self, query, key, val, mask):
        # q, k, v have dimensions batch x seq_len x head_dim
        # 中文说明：[B,Q,D]乘[B,D,K]得[B,Q,K]，bmm不自动广播batch。
        attn_logits = torch.bmm(query, key.transpose(-1, -2))
        # 中文说明：缩放内容分数；CoPE门直接使用这份已缩放分数。
        attn_logits = attn_logits / math.sqrt(self.head_dim)
        # 中文说明：二值mask的log变为0/-inf，先屏蔽再计数；全屏蔽行softmax会NaN。
        attn_logits += mask.log()
        # 中文说明：将位置偏置加回同一logit；输出不是把position加到hidden states。
        attn_logits += self.cope(query, attn_logits)
        # 中文说明：沿最后key维归一化；每条非空有效行权重和为1。
        attn = torch.softmax(attn_logits, dim=-1)
        # 中文说明：[B,Q,K]乘V[B,K,Dv]返回[B,Q,Dv]，没有额外output投影。
        out = torch.bmm(attn, val)
        return out
