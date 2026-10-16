# Transformers 4.44.2 官方实现摘录，只增中文注释。
# 版权所有 HuggingFace Inc.；Apache-2.0：https://github.com/huggingface/transformers/blob/v4.44.2/LICENSE
# 导入原模块依赖，再定义完整原函数/类；该文件可导入，但不是完整LlamaForCausalLM。
from transformers.models.llama import modeling_llama as _official
globals().update({name: value for name, value in vars(_official).items() if not name.startswith("__")})

# ===== rotate_half：完整对象 =====
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 沿指定维连接分片；本验证pretraining_tp=1不走此分支。
    return torch.cat((-x2, x1), dim=-1)

# ===== apply_rotary_pos_emb：完整对象 =====
# 对Q/K应用同位置RoPE；不旋转V。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # 为head轴插入可广播维，cos/sin原为[B,T,dh]。
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # cos*q加sin*半区90度旋转，shape与Q相同。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 对K做相同位置旋转，KV头数可与Q不同。
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ===== repeat_kv：完整对象 =====
# 沿head轴把每份KV连续重复R次，匹配H个query头。
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 插入重复轴，用expand共享原KV数据。
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 把KV头和重复轴合并；不据此声称整个算子零拷贝。
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ===== LlamaRMSNorm：完整对象 =====
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # RMSNorm内部FP32统计，随后恢复输入dtype。
        hidden_states = hidden_states.to(torch.float32)
        # RMSNorm按最后一维算平方均值，不减去均值。
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 乘1/sqrt(均方+eps)，维持[B,T,D]。
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# ===== LlamaMLP：完整对象 =====
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # gate分支D→F；前向中施加SiLU激活。
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # up分支D→F，随后与激活后的gate逐元素相乘。
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # down分支F→D，恢复残差流宽度。
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 旧张量切片计算分支，不等同device_map=auto。
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            # gate分支D→F；前向中施加SiLU激活。
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            # up分支D→F，随后与激活后的gate逐元素相乘。
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            # down分支F→D，恢复残差流宽度。
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            # 沿指定维连接分片；本验证pretraining_tp=1不走此分支。
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            # 沿指定维连接分片；本验证pretraining_tp=1不走此分支。
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # gate分支D→F；前向中施加SiLU激活。
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

# ===== LlamaAttention：完整对象 =====
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        # R=H//G，每份KV服务R个连续query头。
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        # 无显式mask且T>1时才让SDPA自行施加因果约束。
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Q投影：D→H×dh；Linear权重存储为[out,in]。
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # K投影：D→G×dh；GQA只保留G个KV头。
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # V投影：D→G×dh；与K共用头数，不共享参数矩阵。
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # 拼接H个query头输出后映射回D。
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # 旧张量切片计算分支，不等同device_map=auto。
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            # Q投影：D→H×dh；Linear权重存储为[out,in]。
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            # K投影：D→G×dh；GQA只保留G个KV头。
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            # V投影：D→G×dh；与K共用头数，不共享参数矩阵。
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            # 沿指定维连接分片；本验证pretraining_tp=1不走此分支。
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            # 沿指定维连接分片；本验证pretraining_tp=1不走此分支。
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            # 沿指定维连接分片；本验证pretraining_tp=1不走此分支。
            value_states = torch.cat(value_states, dim=-1)

        else:
            # Q投影：D→H×dh；Linear权重存储为[out,in]。
            query_states = self.q_proj(hidden_states)
            # K投影：D→G×dh；GQA只保留G个KV头。
            key_states = self.k_proj(hidden_states)
            # V投影：D→G×dh；与K共用头数，不共享参数矩阵。
            value_states = self.v_proj(hidden_states)

        # [B,T,H*dh]拆头并换轴为[B,H,T,dh]。
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B,T,G*dh]拆为[B,G,T,dh]；cache保留G头。
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # V同样拆成G个KV头，V不施加RoPE。
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # 对Q/K应用同位置RoPE；不旋转V。
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # 真实Cache按layer_idx追加KV，发生状态修改。
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 沿head轴把每份KV连续重复R次，匹配H个query头。
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # 沿head轴把每份KV连续重复R次，匹配H个query头。
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 逐query头计算QK转置，除sqrt(dh)，得[B,H,T,S]。
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # 加性mask：有效处0、不可见处负无穷或极小值。
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        # 沿key位置归一化，内部升FP32再转回query dtype。
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 训练时按概率丢弃注意力权重；eval时关闭。
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # [B,H,T,S]乘V[B,H,S,dh]得到每头输出。
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 从[B,H,T,dh]换回[B,T,H,dh]，contiguous使后续view合法。
        attn_output = attn_output.transpose(1, 2).contiguous()

        # 拼接head，恢复[B,T,D]。
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # 旧张量切片计算分支，不等同device_map=auto。
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            # 拼接H个query头输出后映射回D。
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            # 输出投影；不会减少query头数或序列长度。
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# ===== LlamaSdpaAttention：完整对象 =====
class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # SDPA不直接返回权重，要求输出时回退eager分支。
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        # Q投影：D→H×dh；Linear权重存储为[out,in]。
        query_states = self.q_proj(hidden_states)
        # K投影：D→G×dh；GQA只保留G个KV头。
        key_states = self.k_proj(hidden_states)
        # V投影：D→G×dh；与K共用头数，不共享参数矩阵。
        value_states = self.v_proj(hidden_states)

        # [B,T,H*dh]拆头并换轴为[B,H,T,dh]。
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B,T,G*dh]拆为[B,G,T,dh]；cache保留G头。
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # V同样拆成G个KV头，V不施加RoPE。
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # 对Q/K应用同位置RoPE；不旋转V。
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # 真实Cache按layer_idx追加KV，发生状态修改。
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 沿head轴把每份KV连续重复R次，匹配H个query头。
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # 沿head轴把每份KV连续重复R次，匹配H个query头。
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # 无显式mask且T>1时才让SDPA自行施加因果约束。
        is_causal = True if causal_mask is None and q_len > 1 else False

        # 调用PyTorch真实SDPA内核，默认含1/sqrt(dh)缩放。
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # 从[B,H,T,dh]换回[B,T,H,dh]，contiguous使后续view合法。
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 拼接head，恢复[B,T,D]。
        attn_output = attn_output.view(bsz, q_len, -1)

        # 输出投影；不会减少query头数或序列长度。
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
