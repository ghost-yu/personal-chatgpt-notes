# -*- coding: utf-8 -*-
# 固定Meta提交7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e的完整Attention类。
# 只增加中文注释；依赖完整model.py的导入、ModelArgs、RoPE/repeat_kv等，不能单独运行。
# nn.Module 子类的摘录；缺 super 初始化及部分属性，不能单独运行。
class Attention(nn.Module):
    """Multi-head attention module."""
    # 构造方法：args提供D、head数和最大batch/长度等配置。
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        # 调用nn.Module父类构造，注册子模块与参数机制。
        super().__init__()
        # 未指定KV heads时默认与Q heads相同；否则使用GQA配置。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 从FairScale取得模型并行组大小，不是数据并行batch数。
        model_parallel_size = fs_init.get_model_parallel_world_size()
        # 全局head数按模型并行组切分；本次真实运行world_size=1。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 全局head数按模型并行组切分；本次真实运行world_size=1。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 每个KV head服务多少Q heads；完整forward据此repeat_kv。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每head通道d=D/Hq；配置需要可整除。
        self.head_dim = args.dim // args.n_heads

        # Q投影：实际本rank输出[B,S,Hq_local*d]。
        self.wq = ColumnParallelLinear(
            args.dim,
            # 轴3为每head通道d=D/Hq。
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # K投影：输出[B,S,Hkv_local*d]。
        self.wk = ColumnParallelLinear(
            args.dim,
            # 轴3为每head通道d=D/Hq。
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # V投影：输出[B,S,Hkv_local*d]。
        self.wv = ColumnParallelLinear(
            args.dim,
            # 轴3为每head通道d=D/Hq。
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # 合并heads后投影回D通道；完整类在末尾调用它。
        self.wo = RowParallelLinear(
            # 轴3为每head通道d=D/Hq。
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # 预分配固定容量的普通Tensor属性；不是可训练权重。
        self.cache_k = torch.zeros(
            (
                # 轴0为最大batch容量，实际只使用前B行。
                args.max_batch_size,
                # 轴1为最大token容量；切片写入，不动态扩容。
                args.max_seq_len,
                # 轴2为当前rank的KV heads，不一定等于Q heads。
                self.n_local_kv_heads,
                # 轴3为每head通道d=D/Hq。
                self.head_dim,
            )
        # 原版强制CUDA；CPU教学模块仅显式修改这两处设备调用。
        ).cuda()
        # 预分配固定容量的普通Tensor属性；不是可训练权重。
        self.cache_v = torch.zeros(
            (
                # 轴0为最大batch容量，实际只使用前B行。
                args.max_batch_size,
                # 轴1为最大token容量；切片写入，不动态扩容。
                args.max_seq_len,
                # 轴2为当前rank的KV heads，不一定等于Q heads。
                self.n_local_kv_heads,
                # 轴3为每head通道d=D/Hq。
                self.head_dim,
            )
        # 原版强制CUDA；CPU教学模块仅显式修改这两处设备调用。
        ).cuda()

    # 前向方法在本实例上写cache；摘录缺输入shape解包和最终return。
    def forward(
        self,
        # 输入浮点隐藏状态[B,S,D]，不是token整数ID。
        x: torch.Tensor,
        # P为本次首token绝对位置，也是写缓存切片的起点。
        start_pos: int,
        # 本次位置P:P+S的复数频率[S,d/2]，由上层切片。
        freqs_cis: torch.Tensor,
        # None或可广播的因果mask，完整得分shape[B,Hq_local,S,P+S]。
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        # 从输入[B,S,D]解包batch与本次token长度，补齐摘录缺失的变量。
        bsz, seqlen, _ = x.shape
        # 三次投影只计算本次S个token，旧KV不重新投影。
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 把投影通道拆为Q heads，得到[B,S,Hq_local,d]。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # 轴2为当前rank的KV heads，不一定等于Q heads。
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # 轴2为当前rank的KV heads，不一定等于Q heads。
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Q/K按绝对位置旋转，V不旋转；缓存K是此后的结果。
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 匹配Q的设备和dtype，不是保存Q；转换可能重分配。
        self.cache_k = self.cache_k.to(xq)
        # 匹配Q的设备和dtype，不是保存Q；转换可能重分配。
        self.cache_v = self.cache_v.to(xq)

        # 原地覆写当前[B,P:P+S]区域；此前KV保持不变。
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # 原地覆写当前[B,P:P+S]区域；此前KV保持不变。
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 读取已用前缀P+S，包括当前新KV，排除未写入的容量尾部。
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        # 读取已用前缀P+S，包括当前新KV，排除未写入的容量尾部。
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # 将Hkv_local按组复制到Hq_local以对齐Q，缓存自身仍保存较少KV heads。
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # 将Hkv_local按组复制到Hq_local以对齐Q，缓存自身仍保存较少KV heads。
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # 交换token与head轴，使后续每个head独立矩阵乘法。
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # 交换token与head轴，使后续每个head独立矩阵乘法。
        keys = keys.transpose(1, 2)
        # 交换token与head轴，使后续每个head独立矩阵乘法。
        values = values.transpose(1, 2)
        # [B,Hq,S,d]乘[B,Hq,d,P+S]，除√d得attention scores。
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 多token需防止新chunk内部读取未来位置；单token decode无未来新行。
        if mask is not None:
            # mask以0放行、负无穷阻断，需要可广播到[B,Hq,S,P+S]。
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # 先用FP32沿key轴归一化，再恢复Q dtype。
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 注意力权重乘V得[B,Hq,S,d]；V本身不是加权和。
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # 交换token与head轴，使后续每个head独立矩阵乘法。
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 输出投影回D，返回[B,S,D]给残差分支。
        return self.wo(output)
