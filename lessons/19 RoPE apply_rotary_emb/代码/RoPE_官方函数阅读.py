# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 原源码适用Llama 2 Community License Agreement。
# 固定来源：https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py
# 三个官方函数执行语句保留，英文docstring省去并加入中文注释；不是原notebook单元。
# 实际验证加载完整未修改官方文件，原notebook缺helper的调用错误单独记录。
from typing import Tuple
import torch

# dim=head_dim，end位置数量，theta默认10000为频率基数，不是角度。
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 频率[dim/2]，指数必须除dim；dim偶数由调用方保证。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 位置0..end-1与频率同设备。
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 频率变量改绑为相位[end,dim/2]。
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 半径1/弧度相位→complex64[end,dim/2]。
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # 返回单位复数因子，无参数更新。
    return freqs_cis

# 原课程缺少的helper，接收[T,P]表与[B,T,H,P]复数张量。
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 预期x.ndim为4；通用helper并不代替apply的四维输入约束。
    ndim = x.ndim
    # 确认位置轴1存在；Python优化模式可能去掉assert。
    assert 0 <= 1 < ndim
    # 核验位置T和通道对P，不能让T静默对齐head轴。
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # enumerate遍历轴；保留位置/最后轴，其余设1→[1,T,1,P]。
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # 星号解包shape，返回广播视图，共享频率存储。
    return freqs_cis.view(*shape)

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
    # FP32→保留B/T/H、末轴配对[...,d/2,2]→complex64[B,T,Hq,d/2]；星号展开shape。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # 同理转换key，Hkv可不同；复数view还要求合适stride。
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 原notebook未定义此helper，调用时NameError；官方helper补成[1,T,1,d/2]。
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 逐元素复数乘旋转，real恢复末轴2，flatten第3轴起→[B,T,Hq,d]；不加flip。
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # key同理，仍不计算QKᵀ、softmax或cache写入。
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # 各自转回输入dtype并组成tuple；局部重绑定不修改原输入。
    return xq_out.type_as(xq), xk_out.type_as(xk)
