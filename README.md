# Personal ChatGPT 中文讲解与代码注释

**下表可以直接打开每讲的讲解和注释代码；点课程名查看完整阅读顺序。**

**不知道先打开什么：先看 `解释.md`，再看 Python 阅读版。** 想逐格运行时再打开注释 notebook。每讲同名文件的用途相同，按下表选讲次即可。

[直接进入第 01 讲](lessons/01%20LLaMA%20introduction/README.md)

每讲主要看这三个文件：

| 想看什么 | 打开哪个文件 |
|---|---|
| 概念、公式、张量 shape 与实现流程 | `解释.md` |
| 原课程代码和详细中文注释 | `代码/tutorials/代码注释.ipynb` |
| 方便阅读的纯文本代码 | `代码/tutorials/…_代码注释.py` |

Notebook 请用 GitHub 或 Jupyter 的 notebook 视图打开；想直接看文本代码就打开 `.py`。

`原代码_带说明.ipynb` 是对照版本，`来源清单.json` 和 `复现/outputs` 是核验记录；日常学习直接使用上面三个入口。

**01–19 讲完整注释已发布；20–30 讲待完成。** 下一讲是第 20 讲。

## 选一讲开始阅读

| 讲次 / 课程目录 | 中文讲解 | 注释 notebook | Python 阅读版 |
|---|---|---|---|
| [01 LLaMA introduction](lessons/01%20LLaMA%20introduction/README.md) | [讲解](lessons/01%20LLaMA%20introduction/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/01%20LLaMA%20introduction/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/01%20LLaMA%20introduction/%E4%BB%A3%E7%A0%81/tutorials/01_llama_intro_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [02 LoRA - Low-Rank Adaptation of Large Language Models](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/README.md) | [讲解](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E4%BB%A3%E7%A0%81/tutorials/02_lora_baics_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [03 LoRA 微调 BLOOM](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/README.md) | [讲解](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E4%BB%A3%E7%A0%81/tutorials/03_lora_practice_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [04 PEFT LoRA 源码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/README.md) | [讲解](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E4%BB%A3%E7%A0%81/tutorials/04_PEFT_lora_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [05 FP16 与 BF16](lessons/05%20FP16%20%E4%B8%8E%20BF16/README.md) | [讲解](lessons/05%20FP16%20%E4%B8%8E%20BF16/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/05%20FP16%20%E4%B8%8E%20BF16/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/05%20FP16%20%E4%B8%8E%20BF16/%E4%BB%A3%E7%A0%81/tutorials/05_float16_bf16_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [06 Alpaca LoRA 推理](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/README.md) | [讲解](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/%E4%BB%A3%E7%A0%81/tutorials/06_alpaca_llama_7b_lora_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [07 自动混合精度训练](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/README.md) | [讲解](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/%E4%BB%A3%E7%A0%81/tutorials/07_fp16_mixed_precision_training_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [08 LLM.int8() - 8-bit Matrix Multiplication for Transformers at Scale](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/README.md) | [讲解](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/%E4%BB%A3%E7%A0%81/tutorials/LLM.int8_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [09 BPE 与 GPT-2 tokenizer](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/README.md) | [讲解](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/%E4%BB%A3%E7%A0%81/tutorials/tokenizer_basics_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [10 预训练语料、memory mapping 与 streaming](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/README.md) | [讲解](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/%E4%BB%A3%E7%A0%81/tutorials/pretrain_corpus_%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [11 梯度累积 gradient accumulation](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/README.md) | [讲解](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/%E4%BB%A3%E7%A0%81/tutorials/gradient_accumulation_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [12 LLM SFT 与 SFTTrainer](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/README.md) | [讲解](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E4%BB%A3%E7%A0%81/tutorials/llm_training_sft_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [13 Gradient checkpointing](lessons/13%20Gradient%20checkpointing/README.md) | [讲解](lessons/13%20Gradient%20checkpointing/%E8%A7%A3%E9%87%8A.md) | [带注释代码](lessons/13%20Gradient%20checkpointing/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) | [文本代码](lessons/13%20Gradient%20checkpointing/%E4%BB%A3%E7%A0%81/tutorials/gradient_checkpointing_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) |
| [14 Llama 2 introduction 与 Guanaco](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/README.md) | [讲解](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/解释.md) | [带注释代码](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/代码/tutorials/代码注释.ipynb) | [文本代码](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/代码/tutorials/llama2_introduction_代码注释.py) |
| [15 Llama 2 text 与 chat completion](lessons/15%20Llama%202%20text%20与%20chat%20completion/README.md) | [讲解](lessons/15%20Llama%202%20text%20与%20chat%20completion/解释.md) | [带注释代码](lessons/15%20Llama%202%20text%20与%20chat%20completion/代码/tutorials/代码注释.ipynb) | [文本代码](lessons/15%20Llama%202%20text%20与%20chat%20completion/代码/tutorials/llama2_src_01_代码注释.py) |
| [16 TRL 基础与 reward PPOTrainer](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/README.md) | [讲解](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/解释.md) | [带注释代码](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/代码/tutorials/trl_hf/代码注释.ipynb) | [文本代码](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/代码/tutorials/trl_hf/trl_basics_代码注释.py) |
| [17 RMSNorm 与 SwiGLU](lessons/17%20RMSNorm%20与%20SwiGLU/README.md) | [讲解](lessons/17%20RMSNorm%20与%20SwiGLU/解释.md) | [带注释代码](lessons/17%20RMSNorm%20与%20SwiGLU/代码/tutorials/代码注释.ipynb) | [文本代码](lessons/17%20RMSNorm%20与%20SwiGLU/代码/tutorials/llama2_src_02_RMSNorm_SwiGLU_代码注释.py) |
| [18 RoPE 复数形式](lessons/18%20RoPE%20复数形式/README.md) | [讲解](lessons/18%20RoPE%20复数形式/解释.md) | [带注释代码](lessons/18%20RoPE%20复数形式/代码/tutorials/代码注释.ipynb) | [文本代码](lessons/18%20RoPE%20复数形式/代码/tutorials/llama2_src_03_RoPE_01_complex_position_encoding_代码注释.py) |
| [19 RoPE apply_rotary_emb](lessons/19%20RoPE%20apply_rotary_emb/README.md) | [讲解](lessons/19%20RoPE%20apply_rotary_emb/解释.md) | [带注释代码](lessons/19%20RoPE%20apply_rotary_emb/代码/tutorials/代码注释.ipynb) | [文本代码](lessons/19%20RoPE%20apply_rotary_emb/代码/tutorials/llama2_src_03_RoPE_apply_rotary_emb_代码注释.py) |

第 08、09、15、18 讲包含补充 notebook；第 09 讲还包含训练 tokenizer 的代码。点课程名进入该讲目录，可以看到完整阅读顺序。

<details>
<summary>需要对照原版、查证验证结果或维护项目时，再展开</summary>

- 各讲 `代码/tutorials/原文件名.ipynb`：上游原版。
- 各讲 `原代码_带说明.ipynb`：原代码不改，插入单元说明。
- 各讲 `代码单元解释.md`：按原单元编号查说明。
- 各讲 `复现/README.md`：实际运行范围、环境和重做命令。
- `来源清单.json`、`复现/outputs`：来源及验证证据，通常不用阅读。
- [全部代码覆盖进度](资料/代码覆盖进度.md)、[字幕术语校正](资料/字幕校正.md)、[发布说明](资料/发布说明.md)。
- `工具/` 与 `AGENTS.md`：维护检查及工作约定。

原教程：[chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt)。教学验证与原大模型训练分开记录，实际完成范围见各讲复现文档。

</details>
