# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# 固定Meta完整generate、text_completion和sample_top_p，只增加中文注释。
# 仅选取Llama的相关方法，非完整Llama类；依赖原模块的typing/torch/F/Transformer/Tokenizer等。

class Llama:
    # 保存真实Transformer与Tokenizer实例，generate依赖这两个状态对象。
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # 装饰器关闭梯度记录，不自动切换eval模式。
    @torch.inference_mode()
    # 实例方法输入二维token列表，返回每样本token列表与可选logprobs。
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # 读取模型容量配置，不在此加载权重。
        params = self.model.params
        # 批次数是外层列表长度，单条list[int]需先组成批次。
        bsz = len(prompt_tokens)
        # 断言不能超过最大batch容量。
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # 取最短提示长度作为首次prefill终点。
        min_prompt_len = min(len(t) for t in prompt_tokens)
        # 最长提示决定批次统一容量。
        max_prompt_len = max(len(t) for t in prompt_tokens)
        # 检查最长prompt不超过模型容量。
        assert max_prompt_len <= params.max_seq_len
        # 全批次分配长度=min(模型上限，最长prompt+新增上限)。
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # 真实SentencePiece此处PAD=-1，仅占位，不得传入embedding。
        pad_id = self.tokenizer.pad_id
        # 分配int64[B,total_len]，原版强制CUDA，初始全为PAD。
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        # enumerate同时返回样本索引与该条token列表。
        for k, t in enumerate(prompt_tokens):
            # 将原prompt拷到各行前缀；不等长空位暂时保留PAD。
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        # 控制是否记录token对数概率；原版循环内CE发生在填新增token之前。
        if logprobs:
            # 分配FP32[B,total_len]，第一个token没有此前logit而保持0。
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        # 首轮从token0开始；后续更新才避免重算。
        prev_pos = 0
        # 每样本布尔停止标记；batch要全部True才整体退出。
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        # 固定布尔mask保存哪些位置原先属于提示词，不能由采样覆盖。
        input_text_mask = tokens != pad_id
        # 从最短prompt末尾开始，右端total_len不包含。
        for cur_pos in range(min_prompt_len, total_len):
            # 仅传尚未处理的[prev_pos:cur_pos]，起始绝对位置为prev_pos。
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # 控制是否记录token对数概率；原版循环内CE发生在填新增token之前。
            if logprobs:
                # 原版此时tokens当前新增槽仍为PAD；ignore_index令新增概率被错误记成0。
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    # CE需要[B,V,S]，交换位置轴与词表轴。
                    input=logits.transpose(1, 2),
                    # 目标相对输入右移一位；新增最后目标尚未填。
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    # PAD=-1不参与CE，解释固定旧版新增logprobs为-0的原因。
                    ignore_index=pad_id,
                )
            # 正温度才softmax采样；0或负值走greedy。
            if temperature > 0:
                # 取最后位置词表分布[B,V]，温度缩放后归一化。
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # 真实核采样，返回词表ID[B,1]。
                next_token = sample_top_p(probs, top_p)
            else:
                # greedy直接选最大logit，[B]；不使用top_p。
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # 把采样[B,1]或greedy[B]统一成[B]。
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            # 原prompt位置选原ID，空位才选新token。
            next_token = torch.where(
                # True表示原prompt，保留tokens中对应原值。
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # 将所选token真正写回当前列；已EOS的行也未被冻结。
            tokens[:, cur_pos] = next_token
            # 只把新生成EOS置True，原prompt内EOS不在此触发。
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            # 下一轮只处理刚写入的token；cache保存之前的位置。
            prev_pos = cur_pos
            # 全部样本都遇到生成EOS才提前退出批次。
            if all(eos_reached):
                break

        # 控制是否记录token对数概率；原版循环内CE发生在填新增token之前。
        if logprobs:
            # 张量转普通嵌套浮点列表供返回。
            token_logprobs = token_logprobs.tolist()
        # 准备逐样本输出列表，不是另一次生成。
        out_tokens, out_logprobs = [], []
        # 按样本取整行token，进行返回裁剪。
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            # echo控制返回起点：0或本条prompt长度。
            start = 0 if echo else len(prompt_tokens[i])
            # 每条最多保留自身prompt长度+max_gen_len以内的位置。
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            # 控制是否记录token对数概率；原版循环内CE发生在填新增token之前。
            if logprobs:
                # 概率使用与token同样的起止切片。
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            # 在所返回片段中找首个EOS，echo时也可能找到prompt里的EOS。
            if self.tokenizer.eos_id in toks:
                # list.index返回第一个EOS位置。
                eos_idx = toks.index(self.tokenizer.eos_id)
                # 不返回EOS本身及之后token。
                toks = toks[:eos_idx]
                # 概率与token一起截断，维持对齐。
                probs = probs[:eos_idx] if logprobs else None
            # 收集当前样本token列表。
            out_tokens.append(toks)
            # logprobs=False时只临时收None，最终第二返回值为None。
            out_logprobs.append(probs)
        # 返回二元tuple；不在此decode成字符串。
        return (out_tokens, out_logprobs if logprobs else None)

    # 外层封装把List[str]变二维IDs并decode结果。
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        # 每条文本编码，通常加BOS、不加EOS；形成batch。
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # 调用generate并解包两个返回项。
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        # 控制是否记录token对数概率；原版循环内CE发生在填新增token之前。
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]


# 普通函数接收[B,V]概率和阈值，返回[B,1]整数token ID。
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    # 降序排序，索引表用于回到原词表ID。
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 沿词表累计排序概率。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 用当前项之前的累计质量判断，保留越过阈值的一项。
    mask = probs_sum - probs_sort > p
    # 被排除候选概率置0，不修改原probs输入。
    probs_sort[mask] = 0.0
    # 原地除保留质量，重新归一化；非法p可能导致除0。
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 按概率抽样一项，这是真实随机步骤，不是argmax。
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # 把排序后位置映射回原词表ID。
    next_token = torch.gather(probs_idx, -1, next_token)
    # 返回采样ID，[B,1]。
    return next_token
