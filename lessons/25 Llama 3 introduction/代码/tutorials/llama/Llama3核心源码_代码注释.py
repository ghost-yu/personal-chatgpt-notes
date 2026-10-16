# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# 完整官方配置/FeedForward类摘录，仅注释；依赖原模块的imports和FairScale初始化。

@dataclass
# dataclass声明配置字段，实际checkpoint JSON覆盖默认值。
class ModelArgs:
    # 模型隐藏宽度D，输入嵌入/残差都保留此维度。
    dim: int = 4096
    # TransformerBlock数量L，不包含最终norm和输出头。
    n_layers: int = 32
    # 查询Q头数H，单头宽度为dim//n_heads。
    n_heads: int = 32
    # KV头数G，None时Attention采用与Q相同的头数。
    n_kv_heads: Optional[int] = None
    # 词表总行数，默认-1需被实际词表大小覆盖。
    vocab_size: int = -1
    # SwiGLU中间宽上取整倍数；8B JSON实际覆盖为1024。
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    # SwiGLU宽度额外倍率，8B实际为1.3，None则不乘。
    ffn_dim_multiplier: Optional[float] = None
    # RMSNorm分母稳定项1e-5，不是可训练参数。
    norm_eps: float = 1e-5
    # 位置旋转频率底数500000，不是上下文长度。
    rope_theta: float = 500000

    # 运行缓存batch容量，默认32，可被CLI覆盖。
    max_batch_size: int = 32
    # 运行缓存序列容量默认2048，不是8k训练上下文声明。
    max_seq_len: int = 2048


# 完整官方SwiGLU层，仅加注释，依赖真实FairScale初始化。
class FeedForward(nn.Module):
    # 实例初始化；self是该对象，不是输入张量。
    def __init__(
        self,
        # 模型隐藏宽度D，输入嵌入/残差都保留此维度。
        dim: int,
        hidden_dim: int,
        # SwiGLU中间宽上取整倍数；8B JSON实际覆盖为1024。
        multiple_of: int,
        # SwiGLU宽度额外倍率，8B实际为1.3，None则不乘。
        ffn_dim_multiplier: Optional[float],
    ):
        # 初始化nn.Module注册机制，随后赋值的子层才可被遍历。
        super().__init__()
        # 先乘2/3并向零截断到10922。
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        # 可选倍率不为None时才应用。
        if ffn_dim_multiplier is not None:
            # 乘配置倍率1.3后再次截断。
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # 将中间宽上取整为multiple_of的整数倍。
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 门控分支权重[M,D]；world1输出[B,T,M]。
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        # 输出投影权重[D,M]；world1恢复[B,T,D]。
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        # 内容分支权重[M,D]，独立于w1。
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    # nn.Module.__call__会调用forward；x为浮点[B,T,D]。
    def forward(self, x):
        # silu门控逐元素乘内容，最后投影回D，无bias。
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
