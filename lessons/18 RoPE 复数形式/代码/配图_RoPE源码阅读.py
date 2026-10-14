# -*- coding: utf-8 -*-
# 固定来源截图：chunhuizhang/personal_chatgpt 7fbc29824171fd07dc2c30fe922e61ebc018ad1d
# imgs/hf_llama_rope.png；两函数逐语句转录并添加中文注释。
# 未找到截图的HuggingFace确切版本，不把它标为当前HF或Meta官方源码。
# 这是原图代码的阅读补充；原notebook/Image语句和原PNG没有改动。
import torch

# x输入[...,d]，d为偶数，前d/2与后d/2配对，而非相邻通道配对。
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # ...保留全部前置轴；//2整数除法，取前半通道[...,d/2]。
    x1 = x[..., : x.shape[-1] // 2]
    # 同一tensor的后半通道，维度与x1匹配。
    x2 = x[..., x.shape[-1] // 2 :]
    # cat沿最后轴拼成[-x2,x1]，对应每个半区通道对旋转π/2。
    return torch.cat((-x2, x1), dim=-1)


# q/k均[B,H,T,d]布局；H可以不同；cos/sin原表[1,1,L,d]，ids为int64[B,T]。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can squeeze them.
    # squeeze仅移除指定长度1的轴，两次得到[L,d]；其他shape须另行核对。
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # 用二维ids选位置→[B,T,d]，插入head轴→[B,1,T,d]，共享各head旋转规则。
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # 两次逐元素乘与相加完成旋转，不构造[d,d]矩阵；需采用半区配对的Q和频率表。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回两个同输入shape的tensor，不旋转V，不更新权重或修改输入。
    return q_embed, k_embed
