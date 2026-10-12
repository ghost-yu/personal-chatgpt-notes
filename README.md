# Personal ChatGPT 论文与代码阅读笔记

按讲次解释公式、张量 shape、Python/PyTorch 代码与官方实现。每次完成一篇并提交；原课程为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt)。

## 已完成

| 讲次 | 文档 | 状态 |
|---|---|---|
| 01 | [LLaMA introduction](lessons/01%20LLaMA%20introduction/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 02 | [LoRA - Low-Rank Adaptation of Large Language Models](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 03 | [LoRA 微调 BLOOM](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 04 | [PEFT LoRA 源码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 05 | [FP16 与 BF16](lessons/05%20FP16%20%E4%B8%8E%20BF16/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 06 | [Alpaca LoRA 推理](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 07 | [自动混合精度训练](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 08 | [LLM.int8() - 8-bit Matrix Multiplication for Transformers at Scale](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/%E8%A7%A3%E9%87%8A.md) | 解释完成；部分复现 |
| 09 | [BPE 与 GPT-2 tokenizer](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/%E8%A7%A3%E9%87%8A.md) | 解释完成；小规模真实 tokenizer 训练成功，原配置部分复现 |
| 10 | [预训练语料、memory mapping 与 streaming](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/%E8%A7%A3%E9%87%8A.md) | 解释完成；本地 Datasets/Arrow 验证成功，原 benchmark 部分复现 |
| 11 | [梯度累积 gradient accumulation](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/%E8%A7%A3%E9%87%8A.md) | 解释完成；数学/静态验证通过，未运行 BERT |

## 待完成

- 12：LLM SFT 与 SFTTrainer
- 13：gradient checkpointing
- 14：Llama 2 introduction 与 Guanaco
- 15：Llama 2 text/chat completion
- 16：TRL 基础与 reward/PPOTrainer
- 17：RMSNorm 与 SwiGLU
- 18：RoPE 复数形式
- 19：apply_rotary_emb
- 20：KV cache
- 21：GQA
- 22：generate 完整过程
- 23：RewardTrainer
- 24：LoRA merge/inject/SVD
- 25：Llama 3 introduction
- 26：GPT-4o o200k_base
- 27：RLHF PPOTrainer / OpenRLHF
- 28：RoPE 到 CoPE
- 29：Llama 2 7B vs Llama 3 8B
- 30：InstructGPT reward modeling

课程标题中的“221”按顺序记为第 22 讲；第 16 讲对应未编号的 TRL 基础。资料缺失与版本不确定会在各讲明确记录。

## 验证边界

当前结果主要为静态与小型教学验证，未完成大模型全量训练或论文全量评测。请阅读各篇复现状态。第 08 讲脚本可直接用 Python 标准库运行。第三方资料与本地配套文件范围见 [发布说明](发布说明.md)。
