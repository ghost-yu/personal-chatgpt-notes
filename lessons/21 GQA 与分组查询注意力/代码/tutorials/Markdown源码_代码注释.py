# -*- coding: utf-8 -*-
# 原Markdown4的六条语句；仅供阅读，变量需由完整Attention上下文提供。

# Q拆成[B,S,Hq_local,d]，query head数保持不变。
xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)

# K拆成[B,S,Hkv_local,d]，仅投影较少的KV通道。
xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

# V与K共用分组数量，但投影权重和内容不同。
xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

# 每组有R个Q heads，//不自动检查是否整除。
self.n_rep = self.n_local_heads // self.n_local_kv_heads

# 当前rank的Q head数；world_size=1时等于全局数。
self.n_local_heads = args.n_heads // model_parallel_size

# 当前rank的KV head数，不是query数量，也不是batch。
self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
