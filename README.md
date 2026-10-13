# Personal ChatGPT 论文与代码阅读笔记

按讲次解释公式、张量 shape、Python/PyTorch 代码与官方实现。每次完成一篇并提交；原课程为 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt)。

## 原 notebook 与详细中文注释

按用户最新要求，从第01讲开始逐讲将原notebook、配套图片、中文注释notebook和Python阅读版发布到本仓库。已存在的主文列表不代表所有讲次都已按新格式交付。

| 讲次 | 阅读入口 | 验证范围 |
|---|---|---|
| 01 | [本讲目录](lessons/01%20LLaMA%20introduction/README.md) · [原 notebook](lessons/01%20LLaMA%20introduction/代码/tutorials/01_llama_intro.ipynb) · [中文注释 notebook](lessons/01%20LLaMA%20introduction/代码/tutorials/代码注释.ipynb) · [Python 阅读版](lessons/01%20LLaMA%20introduction/代码/tutorials/01_llama_intro_代码注释.py) | 全部9格说明，2格Python与6条命令详细注释，实际展示通过 |
| 02 | [本讲目录](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/README.md) · [原 notebook](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/代码/tutorials/02_lora_baics.ipynb) · [中文注释 notebook](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/代码/tutorials/代码注释.ipynb) · [Python 阅读版](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/代码/tutorials/02_lora_baics_代码注释.py) | 全部16格说明，7格Python与Markdown13伪代码详细注释，展示及CPU教学LoRA验证通过 |
| 03 | [本讲目录](lessons/03%20LoRA%20微调%20BLOOM/README.md) · [原 notebook](lessons/03%20LoRA%20微调%20BLOOM/代码/tutorials/03_lora_practice.ipynb) · [中文注释 notebook](lessons/03%20LoRA%20微调%20BLOOM/代码/tutorials/代码注释.ipynb) · [Python 阅读版](lessons/03%20LoRA%20微调%20BLOOM/代码/tutorials/03_lora_practice_代码注释.py) | 全部47格说明/37格详注；原版与注释版各26格在CPU教学上下文执行，并分别完成2步Trainer训练 |

以上均未运行原大模型推理或微调。下一讲按该格式交付第04讲。

## 主解释已完成

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
| 12 | [LLM SFT 与 SFTTrainer](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E8%A7%A3%E9%87%8A.md) | 解释完成；packing/数学/源码 fixture 验证通过，未运行 Xgen |
| 13 | [Gradient checkpointing](lessons/13%20Gradient%20checkpointing/%E8%A7%A3%E9%87%8A.md) | 解释完成；梯度/重算/静态验证通过，未运行 BERT 显存实验 |
| 14 | [Llama 2 introduction 与 Guanaco](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/%E8%A7%A3%E9%87%8A.md) | 解释完成；真实 tokenizer/样本/配置验证通过，未运行7B生成或训练 |
| 15 | [Llama 2 text 与 chat completion](lessons/15%20Llama%202%20text%20%E4%B8%8E%20chat%20completion/%E8%A7%A3%E9%87%8A.md) | 解释完成；官方包装方法与真实 tokenizer 验证通过，生成器为 fixture |
| 16 | [TRL 基础与 reward/PPOTrainer](lessons/16%20TRL%20%E5%9F%BA%E7%A1%80%E4%B8%8E%20reward%20PPOTrainer/%E8%A7%A3%E9%87%8A.md) | 解释完成；真实 tokenizer/边界/PPO 数学验证通过，未运行 PPO 更新 |

| 17 | [RMSNorm 与 SwiGLU](lessons/17%20RMSNorm%20%E4%B8%8E%20SwiGLU/%E8%A7%A3%E9%87%8A.md) | 解释完成；数学/梯度/官方 forward fixture 验证通过，未运行 PyTorch 模型 |

| 18 | [RoPE 复数形式](lessons/18%20RoPE%20%E5%A4%8D%E6%95%B0%E5%BD%A2%E5%BC%8F/%E8%A7%A3%E9%87%8A.md) | 解释完成；主 notebook/官方频率函数真实 CPU PyTorch 验证成功，未运行 Llama 模型 |

| 19 | [RoPE apply_rotary_emb](lessons/19%20RoPE%20apply_rotary_emb/%E8%A7%A3%E9%87%8A.md) | 主文与全部15个代码单元说明完成；真实 CPU PyTorch 验证通过 |

## 待完成

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

## 全部 notebook 代码解释

按用户补充要求，每讲交付全部单元说明、本地保留原代码的说明版及中文代码注释版。第01讲全部9格与2/2代码注释、第02讲全部16格与7/7代码注释及Markdown伪代码注释已验证；第19讲15/15说明已完成，代码内中文注释待补；第03讲全部47格和37/37代码注释、真实CPU PEFT/Trainer调用链已核对；第04讲18格、8/8代码格及6/6 Markdown源码块注释与固定官方v0.3.0 CPU注入验证完成；第05–18讲待逐篇核验补齐。下一篇补第05讲，详见 [代码覆盖进度](代码覆盖进度.md)。

## 验证边界

当前结果主要为静态与小型教学验证，未完成大模型全量训练或论文全量评测。请阅读各篇复现状态。第 08 讲脚本可直接用 Python 标准库运行。第三方资料与本地配套文件范围见 [发布说明](发布说明.md)。
