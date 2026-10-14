# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Llama 2 Community License Agreement applies to the original source excerpt.
# 固定来源：https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py
# 官方类的原语句和docstring保留，只添加中文注释；这是源码阅读补充，不是原notebook单元。
# 实际验证加载完整未修改model.py，使用已初始化的单进程Gloo/FairScale与显式随机权重。
from typing import Optional
from torch import nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear

# 完整SwiGLU前馈模块；不是单独SiLU。继承Module注册三投影参数。
class FeedForward(nn.Module):
    # 构造签名四配置都需传入；Optional只表示允许None，签名没有默认值。
    def __init__(
        self,
        # 输入/输出通道d；Llama2 7B为4096，小验证为4。
        dim: int,
        # 传入候选宽度4d，下面还会缩小与向上对齐。
        hidden_dim: int,
        # 对齐倍数，如256；小验证设4。
        multiple_of: int,
        # None不再缩放；ModelArgs默认None，但本构造调用必须显式提供。
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        # 初始化基类以注册后续模块。
        super().__init__()
        # 从候选4d变为约8d/3，使三矩阵参数接近传统两矩阵FFN。
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        # is not None检查是否显式指定额外宽度比例。
        if ffn_dim_multiplier is not None:
            # 应用额外乘数并取整数，配置须与checkpoint匹配。
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # 整数向上对齐；4096→10922→11008，不是直接四舍五入。
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 门控分支投影，逻辑权重[m,d]；world1小验证[12,4]。
        self.w1 = ColumnParallelLinear(
            # 无bias/不收集分支输出；lambda原样返回权重，必须加载checkpoint或另行明确初始化。
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        # 最终输出投影，权重[d,m]；小验证[4,12]。
        self.w2 = RowParallelLinear(
            # 输入已按中间通道划分；world1无需跨rank归约，不能据此验证多卡。
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        # 内容分支独立参数[m,d]，与w1没有共享weight。
        self.w3 = ColumnParallelLinear(
            # 无bias/不收集分支输出；lambda原样返回权重，必须加载checkpoint或另行明确初始化。
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    # 输入[B,T,d]，同位置独立计算，不更新参数，返回[B,T,d]。
    def forward(self, x):
        # 双投影[B,T,m]→SiLU门控逐元素乘→w2恢复d；不是w1→w2→w3顺序。
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
