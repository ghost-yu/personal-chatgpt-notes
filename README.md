# Personal ChatGPT 中文讲解与代码注释

**阅读入口就是本文件。先选讲次，不必逐个翻项目文件。**

第一次阅读：[从第 01 讲开始](lessons/01%20LLaMA%20introduction/README.md)。在本地请打开 `G:/zzz/personal-chatgpt-notes`，各讲放在 `lessons`。

每讲按这个顺序阅读：

1. **解释.md**：先理解概念、公式、张量 shape 和代码流程。
2. **代码注释.ipynb**：对照原课程代码，阅读详细中文注释与逐格说明。GitHub 和 Jupyter 可按 notebook 展示。
3. **代码注释.py**：喜欢纯文本代码时看这一份，避免 notebook 的 JSON；主要用于阅读，执行方法见本讲复现文档。

原课程来自 [chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt)。当前第 **01–11** 讲的原 notebook、详细代码注释和 Python 阅读版已公开；第 **12–19** 讲已有主解释，完整注释文件仍在逐讲补齐；第 **20–30** 讲待完成。下一讲是第 **12** 讲。

## 已有完整代码注释：01–11

| 讲次 | 中文讲解 | 带注释 notebook | 纯文本代码 |
|---|---|---|---|
| 01 | [LLaMA introduction](lessons/01%20LLaMA%20introduction/解释.md) | [看代码与注释](lessons/01%20LLaMA%20introduction/代码/tutorials/代码注释.ipynb) | [Python版](lessons/01%20LLaMA%20introduction/%E4%BB%A3%E7%A0%81/tutorials/01_llama_intro_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| 02 | [LoRA - Low-Rank Adaptation of Large Language Models](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/解释.md) | [看代码与注释](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/代码/tutorials/代码注释.ipynb) | [Python版](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E4%BB%A3%E7%A0%81/tutorials/02_lora_baics_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| 03 | [LoRA 微调 BLOOM](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/解释.md) | [看代码与注释](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/代码/tutorials/代码注释.ipynb) | [Python版](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E4%BB%A3%E7%A0%81/tutorials/03_lora_practice_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| 04 | [PEFT LoRA 源码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/解释.md) | [看代码与注释](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/代码/tutorials/代码注释.ipynb) | [Python版](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E4%BB%A3%E7%A0%81/tutorials/04_PEFT_lora_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| 05 | [FP16 与 BF16](lessons/05%20FP16%20与%20BF16/解释.md) | [看代码与注释](lessons/05%20FP16%20与%20BF16/代码/tutorials/代码注释.ipynb) | [Python版](lessons/05%20FP16%20与%20BF16/代码/tutorials/05_float16_bf16_代码注释.py) |
| 06 | [Alpaca LoRA 推理](lessons/06%20Alpaca%20LoRA%20推理/解释.md) | [看代码与注释](lessons/06%20Alpaca%20LoRA%20推理/代码/tutorials/代码注释.ipynb) | [Python版](lessons/06%20Alpaca%20LoRA%20推理/代码/tutorials/06_alpaca_llama_7b_lora_代码注释.py) |
| 07 | [自动混合精度训练](lessons/07%20自动混合精度训练/解释.md) | [看代码与注释](lessons/07%20自动混合精度训练/代码/tutorials/代码注释.ipynb) | [Python版](lessons/07%20自动混合精度训练/代码/tutorials/07_fp16_mixed_precision_training_代码注释.py) |
| 08 | [LLM.int8量化与bitsandbytes](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/解释.md) | [看代码与注释](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/代码/tutorials/代码注释.ipynb) | [Python版](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/代码/tutorials/LLM.int8_代码注释.py) |
| 09 | [BPE与GPT-2 tokenizer](lessons/09%20BPE%20与%20GPT-2%20tokenizer/解释.md) | [基础与训练阅读入口](lessons/09%20BPE%20与%20GPT-2%20tokenizer/README.md) | [Python版](lessons/09%20BPE%20与%20GPT-2%20tokenizer/代码/tutorials/tokenizer_basics_代码注释.py) |
| 10 | [预训练语料、memory mapping 与 streaming](lessons/10%20预训练语料、memory%20mapping%20与%20streaming/解释.md) | [原图与数据演示阅读入口](lessons/10%20预训练语料、memory%20mapping%20与%20streaming/README.md) | [Python版](lessons/10%20预训练语料、memory%20mapping%20与%20streaming/代码/tutorials/pretrain_corpus_大模型预训练语料_代码注释.py) |
| 11 | [梯度累积](lessons/11%20梯度累积%20gradient%20accumulation/解释.md) | [训练循环与窗口修正入口](lessons/11%20梯度累积%20gradient%20accumulation/README.md) | [Python版](lessons/11%20梯度累积%20gradient%20accumulation/代码/tutorials/gradient_accumulation_代码注释.py) |

<details>
<summary>第 12–19 讲：已有讲解，完整代码注释待补（展开查看）</summary>

| 讲次 | 中文讲解 |
|---|---|
| 12 | [LLM SFT 与 SFTTrainer](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E8%A7%A3%E9%87%8A.md) |
| 13 | [Gradient checkpointing](lessons/13%20Gradient%20checkpointing/%E8%A7%A3%E9%87%8A.md) |
| 14 | [Llama 2 introduction 与 Guanaco](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/%E8%A7%A3%E9%87%8A.md) |
| 15 | [Llama 2 text 与 chat completion](lessons/15%20Llama%202%20text%20%E4%B8%8E%20chat%20completion/%E8%A7%A3%E9%87%8A.md) |
| 16 | [TRL 基础与 reward PPOTrainer](lessons/16%20TRL%20%E5%9F%BA%E7%A1%80%E4%B8%8E%20reward%20PPOTrainer/%E8%A7%A3%E9%87%8A.md) |
| 17 | [RMSNorm 与 SwiGLU](lessons/17%20RMSNorm%20%E4%B8%8E%20SwiGLU/%E8%A7%A3%E9%87%8A.md) |
| 18 | [RoPE 复数形式](lessons/18%20RoPE%20%E5%A4%8D%E6%95%B0%E5%BD%A2%E5%BC%8F/%E8%A7%A3%E9%87%8A.md) |
| 19 | [RoPE apply_rotary_emb](lessons/19%20RoPE%20apply_rotary_emb/%E8%A7%A3%E9%87%8A.md) |

</details>

## 文件放在哪里

```text
README.md                  ← 总阅读入口，先打开这个
lessons/                   ← 各讲讲解与代码
  01 …/
    README.md              ← 本讲阅读入口，注明交付状态
    解释.md                 ← 概念、公式与实现讲解
    代码/tutorials/
      代码注释.ipynb        ← 首选的代码阅读文件
      …_代码注释.py         ← 纯文本代码阅读版
      原文件名.ipynb        ← 下载的上游原版，用于对照
      原代码_带说明.ipynb   ← 保持原代码不动的说明版
    代码单元解释.md         ← 按原单元编号查具体说明
    复现/                  ← 验证方法、脚本与结果
资料/                      ← 全系列进度、字幕校正、发布说明
工具/                      ← 维护用检查脚本
AGENTS.md                  ← 项目维护约定
```

日常阅读只需要讲解和两种代码阅读文件。`代码单元解释.md` 是查某一格的补充说明；原版和带说明版用于对照，不必重复读所有版本。`来源清单.json`、`复现/scripts` 和 `复现/outputs` 用于追溯来源及验证结果，需要查证时再打开。不同讲次的配套文件以实际目录为准。

<details>
<summary>进度、来源及验证说明</summary>

- [全部代码覆盖进度](资料/代码覆盖进度.md)
- [字幕术语校正](资料/字幕校正.md)
- [发布说明](资料/发布说明.md)
- 每讲的 `复现/README.md` 记录实际运行范围。当前主要是静态与CPU小型教学验证，未完成大模型全量训练或论文全量评测。
- 后续讲次：20 KV cache；21 GQA；22 generate；23 RewardTrainer；24 LoRA merge/inject/SVD；25 Llama 3；26 GPT-4o o200k_base；27 RLHF PPOTrainer/OpenRLHF；28 RoPE 到 CoPE；29 Llama 2 vs Llama 3；30 InstructGPT reward modeling。

</details>
