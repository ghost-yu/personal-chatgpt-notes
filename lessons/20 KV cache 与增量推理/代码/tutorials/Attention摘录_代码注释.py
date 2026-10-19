# -*- coding: utf-8 -*-
# 原Markdown18的不完整摘录，供阅读，不可独立实例化运行。
# nn.Module 子类的摘录；缺 super 初始化及部分属性，不能单独运行。
class Attention(nn.Module):
    """Multi-head attention module."""
    # 构造方法：args提供D、head数和最大batch/长度等配置。
    def __init__(self, args: ModelArgs):

        # Q投影：实际本rank输出[B,S,Hq_local*d]。
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim)
        # K投影：输出[B,S,Hkv_local*d]。
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim)
        # V投影：输出[B,S,Hkv_local*d]。
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim)
        # 合并heads后投影回D通道；完整类在末尾调用它。
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim)


        # [8, 1024, 32, 128]
        # 预先分配固定容量的缓存张量，作为普通属性保存各位置的 K/V。
        self.cache_k = torch.zeros(
            (
                # 轴0为最大batch容量，实际只使用前B行。
                args.max_batch_size,    # 8
                # 轴1为最大token容量；切片写入，不动态扩容。
                args.max_seq_len,       # 1024, 不断地在这个维度上append keys
                # 轴2为当前rank的KV heads，不一定等于Q heads。
                self.n_local_kv_heads,  # 32
                # 轴3为每head通道d=D/Hq。
                self.head_dim,          # 128
            )
        # 原版强制CUDA；CPU教学模块仅显式修改这两处设备调用。
        ).cuda()

        #  [8, 1024, 32, 128]
        # 预先分配固定容量的缓存张量，作为普通属性保存各位置的 K/V。
        self.cache_v = torch.zeros(
            (
                # 轴0为最大batch容量，实际只使用前B行。
                args.max_batch_size,    # 8
                # 轴1为最大token容量；切片写入，不动态扩容。
                args.max_seq_len,       # 1024, 不断地在这个维度上append values
                # 轴2为当前rank的KV heads，不一定等于Q heads。
                self.n_local_kv_heads,  # 32
                # 轴3为每head通道d=D/Hq。
                self.head_dim,          # 128
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

        # 这里在复用之前的计算, all_past
        # 读取已用前缀P+S，包括当前新KV，排除未写入的容量尾部。
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        # 读取已用前缀P+S，包括当前新KV，排除未写入的容量尾部。
        values = self.cache_v[:bsz, : start_pos + seqlen]
