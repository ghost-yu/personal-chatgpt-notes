# -*- coding: utf-8 -*-
# 原Markdown片段需要各自上下文；供阅读，不能直接当完整generate运行。

# %% 原Markdown 4
# 调用generate并解包两个返回项。
generation_tokens, generation_logprobs = self.generate(
    prompt_tokens=prompt_tokens,
    max_gen_len=max_gen_len,
    temperature=temperature,
    top_p=top_p,
    logprobs=logprobs,
    echo=echo,
)


# %% 原Markdown 10
# 从最短prompt末尾开始，右端total_len不包含。
for cur_pos in range(min_prompt_len, total_len):
    # logits.shape: [bsz, slice_seq_len, vocab_size]
    # tokens: [0, 8), prompt 部分,
    # 下一次 tokens: [8, 9)
    # 下一次 tokens: [9, 10)
    # 仅传尚未处理的[prev_pos:cur_pos]，起始绝对位置为prev_pos。
    logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)


# %% 原Markdown 13
# 截止到当前（包含历史）！！！
keys = self.cache_k[:bsz, : start_pos + seqlen]
values = self.cache_v[:bsz, : start_pos + seqlen]



# %% 原Markdown 16
# 正温度才softmax采样；0或负值走greedy。
if temperature > 0:
    # temperature 与 top_p 组合使用
    # 取最后位置词表分布[B,V]，温度缩放后归一化。
    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    # 真实核采样，返回词表ID[B,1]。
    next_token = sample_top_p(probs, top_p)
else:
    # greedy直接选最大logit，[B]；不使用top_p。
    next_token = torch.argmax(logits[:, -1], dim=-1)
