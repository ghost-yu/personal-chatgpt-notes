# -*- coding: utf-8 -*-
# 固定Meta官方repeat_kv与完整Attention类，只新增中文注释。
# 依赖原模块的torch/nn/F、FairScale、ModelArgs与RoPE helper，不能单文件直接运行。

# 普通函数，不是模型方法；输入[B,T,G,d]，返回[B,T,G*R,d]，无可训练参数。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 解包四轴：batch、已用序列长度、KV groups、每head通道。
    bs, slen, n_kv_heads, head_dim = x.shape
    # R=1不需重复；MHA情况直接复用原Tensor对象。
    if n_rep == 1:
        # 没有复制；返回对象与输入是同一个。
        return x
    return (
        # None插入新轴，[B,T,G,d]变为[B,T,G,1,d]。
        x[:, :, :, None, :]
        # 广播重复轴到R；通常是stride=0的共享存储view，此步不复制元素。
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # 合并G/R轴，相邻Q heads拿到同一组KV；必要时reshape会分配新存储。
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# 完整固定官方Attention类；依赖原model.py的其他导入和helper。
class Attention(nn.Module):
    """Multi-head attention module."""
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
        # 初始化nn.Module，随后赋给self的线性层会注册为子模块。
        super().__init__()
        # None默认退化为MHA；否则使用指定的KV groups。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 取得真实FairScale模型并行组大小；本次只执行单rank。
        model_parallel_size = fs_init.get_model_parallel_world_size()
        # 按rank拆分Q heads，要求配置适配该并行度。
        self.n_local_heads = args.n_heads // model_parallel_size
        # KV heads也按rank拆分，不能在这里直接让每rank少于一个head。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 每KV组服务的Q heads个数R=Hq_local/Hkv_local。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每head通道d，标准配置中D/Hq应为整数。
        self.head_dim = args.dim // args.n_heads

        # Q投影仍输出全部Q heads，GQA不会把Q数一起减少。
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # K/V投影只输出Hkv*d通道，因此参数和缓存较小。
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # K/V投影只输出Hkv*d通道，因此参数和缓存较小。
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # 拼回所有Q head输出后投影回隐藏通道D。
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # 持久cache按Hkv_local分配，不按repeat后的Hq_local分配。
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                # 缓存轴2是较少的KV groups；这决定持久内存节省。
                self.n_local_kv_heads,
                self.head_dim,
            )
        # 原版强制CUDA；教学执行只显式修改这两处cache设备调用。
        ).cuda()
        # 持久cache按Hkv_local分配，不按repeat后的Hq_local分配。
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                # 缓存轴2是较少的KV groups；这决定持久内存节省。
                self.n_local_kv_heads,
                self.head_dim,
            )
        # 原版强制CUDA；教学执行只显式修改这两处cache设备调用。
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
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
        # x为[B,S,D]隐藏状态，读出batch和本次token数。
        bsz, seqlen, _ = x.shape
        # 三组不同权重投影；K/V输出宽度可能比Q小。
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Q拆为[B,S,Hq_local,d]。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # K/V拆为[B,S,Hkv_local,d]；仅head轴比Q少。
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # K/V拆为[B,S,Hkv_local,d]；仅head轴比Q少。
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 给Q/K同位置RoPE；同组Q仍是各自不同的query，V不旋转。
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # cache匹配Q设备与dtype，不是保存Q。
        self.cache_k = self.cache_k.to(xq)
        # cache匹配Q设备与dtype，不是保存Q。
        self.cache_v = self.cache_v.to(xq)

        # 只将本次少量KV写入对应token槽，不在cache中重复成Q head数。
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # 只将本次少量KV写入对应token槽，不在cache中重复成Q head数。
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 读取有效历史与当前token，shape[B,P+S,Hkv_local,d]。
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        # 读取有效历史与当前token，shape[B,P+S,Hkv_local,d]。
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # 为通用attention矩阵乘法把每组KV连续重复R次，对齐Hq_local。
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # 为通用attention矩阵乘法把每组KV连续重复R次，对齐Hq_local。
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # 交换token/head轴以按head独立计算。
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # 交换token/head轴以按head独立计算。
        keys = keys.transpose(1, 2)
        # 交换token/head轴以按head独立计算。
        values = values.transpose(1, 2)
        # QKᵀ/√d得[B,Hq_local,S,P+S]，分数head数仍是Hq。
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # 因果mask约束可见位置；GQA本身不改变因果规则。
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # 每个Q head独立沿key位置softmax，先FP32再转回Q dtype。
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 各Q head使用自己的权重聚合共享V；输出仍有全部Q heads。
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # 交换token/head轴以按head独立计算。
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 最终输出[B,S,D]并交给上层残差；cache已被原地更新。
        return self.wo(output)
