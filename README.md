# Personal ChatGPT 中文讲解与代码注释

从下面的课程表选择讲次，按这个顺序阅读：

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `解释.md` | 概念、手算例子、论文原图、公式和代码流程 |
| 2 | `…_代码注释.py` | 原代码及详细中文注释，适合直接阅读 |
| 3 | `代码注释.ipynb` | 在GitHub或Jupyter中逐格查看代码与输出 |

可以先打开[第01讲](lessons/01%20LLaMA%20introduction/解释.md)。第01–30讲的讲解正文已统一调整为直接讲解的写法。第01–18讲已补充基础讲解和内嵌原图；第19–30讲继续补充。[各讲进度](资料/基础讲解与内嵌原图进度.md)

在VS Code中打开[课程阅读.code-workspace](课程阅读.code-workspace)，侧栏会列出总目录和各讲。维护资料收在阅读视图之外，需要时可通过“打开文件夹”查看完整项目。

```text
lessons/
└── 讲次 名称/
    ├── 解释.md                     ← 先读中文讲解
    ├── README.md                   ← 本讲阅读链接
    └── 代码/tutorials/…/
        ├── 原文件名_代码注释.py     ← 再看代码
        └── 代码注释.ipynb           ← 逐格查看输出
```

原notebook、来源和运行记录放在各讲入口的补充栏中。Notebook用GitHub或Jupyter打开；直接阅读代码时，选表中的“文本代码”。

## 选一讲开始阅读

| 讲次 / 课程目录 | ① 中文讲解 | ② Python 注释代码 | ③ 注释 notebook |
|---|---|---|---|
| [01 LLaMA introduction](lessons/01%20LLaMA%20introduction/README.md) | [讲解](lessons/01%20LLaMA%20introduction/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/01%20LLaMA%20introduction/%E4%BB%A3%E7%A0%81/tutorials/01_llama_intro_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/01%20LLaMA%20introduction/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [02 LoRA - Low-Rank Adaptation of Large Language Models](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/README.md) | [讲解](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E4%BB%A3%E7%A0%81/tutorials/02_lora_baics_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/02%20LoRA%20-%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [03 LoRA 微调 BLOOM](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/README.md) | [讲解](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E4%BB%A3%E7%A0%81/tutorials/03_lora_practice_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/03%20LoRA%20%E5%BE%AE%E8%B0%83%20BLOOM/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [04 PEFT LoRA 源码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/README.md) | [讲解](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E4%BB%A3%E7%A0%81/tutorials/04_PEFT_lora_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/04%20PEFT%20LoRA%20%E6%BA%90%E7%A0%81/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [05 FP16 与 BF16](lessons/05%20FP16%20%E4%B8%8E%20BF16/README.md) | [讲解](lessons/05%20FP16%20%E4%B8%8E%20BF16/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/05%20FP16%20%E4%B8%8E%20BF16/%E4%BB%A3%E7%A0%81/tutorials/05_float16_bf16_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/05%20FP16%20%E4%B8%8E%20BF16/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [06 Alpaca LoRA 推理](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/README.md) | [讲解](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/%E4%BB%A3%E7%A0%81/tutorials/06_alpaca_llama_7b_lora_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/06%20Alpaca%20LoRA%20%E6%8E%A8%E7%90%86/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [07 自动混合精度训练](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/README.md) | [讲解](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/%E4%BB%A3%E7%A0%81/tutorials/07_fp16_mixed_precision_training_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/07%20%E8%87%AA%E5%8A%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [08 LLM.int8() - 8-bit Matrix Multiplication for Transformers at Scale](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/README.md) | [讲解](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/%E4%BB%A3%E7%A0%81/tutorials/LLM.int8_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/08%20LLM.int8%28%29%20-%208-bit%20Matrix%20Multiplication%20for%20Transformers%20at%20Scale/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [09 BPE 与 GPT-2 tokenizer](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/README.md) | [讲解](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/%E4%BB%A3%E7%A0%81/tutorials/tokenizer_basics_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/09%20BPE%20%E4%B8%8E%20GPT-2%20tokenizer/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [10 预训练语料、memory mapping 与 streaming](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/README.md) | [讲解](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/%E4%BB%A3%E7%A0%81/tutorials/pretrain_corpus_%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/10%20%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E6%96%99%E3%80%81memory%20mapping%20%E4%B8%8E%20streaming/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [11 梯度累积 gradient accumulation](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/README.md) | [讲解](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/%E4%BB%A3%E7%A0%81/tutorials/gradient_accumulation_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/11%20%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%20gradient%20accumulation/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [12 LLM SFT 与 SFTTrainer](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/README.md) | [讲解](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E4%BB%A3%E7%A0%81/tutorials/llm_training_sft_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/12%20LLM%20SFT%20%E4%B8%8E%20SFTTrainer/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [13 Gradient checkpointing](lessons/13%20Gradient%20checkpointing/README.md) | [讲解](lessons/13%20Gradient%20checkpointing/%E8%A7%A3%E9%87%8A.md) | [文本代码](lessons/13%20Gradient%20checkpointing/%E4%BB%A3%E7%A0%81/tutorials/gradient_checkpointing_%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.py) | [带注释代码](lessons/13%20Gradient%20checkpointing/%E4%BB%A3%E7%A0%81/tutorials/%E4%BB%A3%E7%A0%81%E6%B3%A8%E9%87%8A.ipynb) |
| [14 Llama 2 introduction 与 Guanaco](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/README.md) | [讲解](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/解释.md) | [文本代码](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/代码/tutorials/llama2_introduction_代码注释.py) | [带注释代码](lessons/14%20Llama%202%20introduction%20%E4%B8%8E%20Guanaco/代码/tutorials/代码注释.ipynb) |
| [15 Llama 2 text 与 chat completion](lessons/15%20Llama%202%20text%20与%20chat%20completion/README.md) | [讲解](lessons/15%20Llama%202%20text%20与%20chat%20completion/解释.md) | [文本代码](lessons/15%20Llama%202%20text%20与%20chat%20completion/代码/tutorials/llama2_src_01_代码注释.py) | [带注释代码](lessons/15%20Llama%202%20text%20与%20chat%20completion/代码/tutorials/代码注释.ipynb) |
| [16 TRL 基础与 reward PPOTrainer](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/README.md) | [讲解](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/解释.md) | [文本代码](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/代码/tutorials/trl_hf/trl_basics_代码注释.py) | [带注释代码](lessons/16%20TRL%20基础与%20reward%20PPOTrainer/代码/tutorials/trl_hf/代码注释.ipynb) |
| [17 RMSNorm 与 SwiGLU](lessons/17%20RMSNorm%20与%20SwiGLU/README.md) | [讲解](lessons/17%20RMSNorm%20与%20SwiGLU/解释.md) | [文本代码](lessons/17%20RMSNorm%20与%20SwiGLU/代码/tutorials/llama2_src_02_RMSNorm_SwiGLU_代码注释.py) | [带注释代码](lessons/17%20RMSNorm%20与%20SwiGLU/代码/tutorials/代码注释.ipynb) |
| [18 RoPE 复数形式](lessons/18%20RoPE%20复数形式/README.md) | [讲解](lessons/18%20RoPE%20复数形式/解释.md) | [文本代码](lessons/18%20RoPE%20复数形式/代码/tutorials/llama2_src_03_RoPE_01_complex_position_encoding_代码注释.py) | [带注释代码](lessons/18%20RoPE%20复数形式/代码/tutorials/代码注释.ipynb) |
| [19 RoPE apply_rotary_emb](lessons/19%20RoPE%20apply_rotary_emb/README.md) | [讲解](lessons/19%20RoPE%20apply_rotary_emb/解释.md) | [文本代码](lessons/19%20RoPE%20apply_rotary_emb/代码/tutorials/llama2_src_03_RoPE_apply_rotary_emb_代码注释.py) | [带注释代码](lessons/19%20RoPE%20apply_rotary_emb/代码/tutorials/代码注释.ipynb) |
| [20 KV cache 与增量推理](lessons/20%20KV%20cache%20与增量推理/README.md) | [讲解](lessons/20%20KV%20cache%20与增量推理/解释.md) | [文本代码](lessons/20%20KV%20cache%20与增量推理/代码/tutorials/llama2_src_cache_kv_代码注释.py) | [带注释代码](lessons/20%20KV%20cache%20与增量推理/代码/tutorials/代码注释.ipynb) |
| [21 GQA 与分组查询注意力](lessons/21%20GQA%20与分组查询注意力/README.md) | [讲解](lessons/21%20GQA%20与分组查询注意力/解释.md) | [核心文本代码](lessons/21%20GQA%20与分组查询注意力/代码/tutorials/GQA核心源码_代码注释.py) | [带注释代码](lessons/21%20GQA%20与分组查询注意力/代码/tutorials/代码注释.ipynb) |
| [22 Llama generate 完整流程](lessons/22%20Llama%20generate%20完整流程/README.md) | [讲解](lessons/22%20Llama%20generate%20完整流程/解释.md) | [核心文本代码](lessons/22%20Llama%20generate%20完整流程/代码/tutorials/generate核心源码_代码注释.py) | [带注释代码](lessons/22%20Llama%20generate%20完整流程/代码/tutorials/代码注释.ipynb) |
| [23 TRL Reward Model 与 RewardTrainer](lessons/23%20TRL%20Reward%20Model%20与%20RewardTrainer/README.md) | [讲解](lessons/23%20TRL%20Reward%20Model%20与%20RewardTrainer/解释.md) | [文本代码](lessons/23%20TRL%20Reward%20Model%20与%20RewardTrainer/代码/tutorials/reward_model/trl_reward_trainer_代码注释.py) | [带注释代码](lessons/23%20TRL%20Reward%20Model%20与%20RewardTrainer/代码/tutorials/reward_model/代码注释.ipynb) |
| [24 PEFT LoRA merge 与 SVD](lessons/24%20PEFT%20LoRA%20merge%20与%20SVD/README.md) | [讲解](lessons/24%20PEFT%20LoRA%20merge%20与%20SVD/解释.md) | [文本代码](lessons/24%20PEFT%20LoRA%20merge%20与%20SVD/代码/tutorials/peft/lora_merge_代码注释.py) | [带注释代码](lessons/24%20PEFT%20LoRA%20merge%20与%20SVD/代码/tutorials/peft/代码注释.ipynb) |
| [25 Llama 3 introduction](lessons/25%20Llama%203%20introduction/README.md) | [讲解](lessons/25%20Llama%203%20introduction/解释.md) | [文本代码](lessons/25%20Llama%203%20introduction/代码/tutorials/llama/llama3_0419_代码注释.py) | [带注释代码](lessons/25%20Llama%203%20introduction/代码/tutorials/llama/代码注释.ipynb) |
| [26 GPT-4o tokenizer 与 o200k_base](lessons/26%20GPT-4o%20tokenizer%20与%20o200k_base/README.md) | [讲解](lessons/26%20GPT-4o%20tokenizer%20与%20o200k_base/解释.md) | [文本代码](lessons/26%20GPT-4o%20tokenizer%20与%20o200k_base/代码/tutorials/tokenizer/tokenizer_gpt4o_代码注释.py) | [带注释代码](lessons/26%20GPT-4o%20tokenizer%20与%20o200k_base/代码/tutorials/tokenizer/代码注释.ipynb) |
| [27 TRL RLHF PPOTrainer 与 OpenRLHF](lessons/27%20TRL%20RLHF%20PPOTrainer%20%E4%B8%8E%20OpenRLHF/README.md) | [讲解](lessons/27%20TRL%20RLHF%20PPOTrainer%20%E4%B8%8E%20OpenRLHF/解释.md) | [文本代码](lessons/27%20TRL%20RLHF%20PPOTrainer%20%E4%B8%8E%20OpenRLHF/代码/tutorials/trl_hf/trl_ppotrainer_helloworld_代码注释.py) | [带注释代码](lessons/27%20TRL%20RLHF%20PPOTrainer%20%E4%B8%8E%20OpenRLHF/代码/tutorials/trl_hf/代码注释.ipynb) |
| [28 从 RoPE 到 CoPE](lessons/28%20%E4%BB%8E%20RoPE%20%E5%88%B0%20CoPE/README.md) | [讲解](lessons/28%20%E4%BB%8E%20RoPE%20%E5%88%B0%20CoPE/解释.md) | [文本代码](lessons/28%20%E4%BB%8E%20RoPE%20%E5%88%B0%20CoPE/代码/tutorials/position_encoding/rope_cope_代码注释.py) | [带注释代码](lessons/28%20%E4%BB%8E%20RoPE%20%E5%88%B0%20CoPE/代码/tutorials/position_encoding/代码注释.ipynb) |
| [29 Llama2 7B vs Llama3 8B](lessons/29%20Llama2%207B%20vs%20Llama3%208B/README.md) | [讲解](lessons/29%20Llama2%207B%20vs%20Llama3%208B/解释.md) | [文本代码](lessons/29%20Llama2%207B%20vs%20Llama3%208B/代码/tutorials/llama/llama2_7b_llama3_8b_代码注释.py) | [带注释代码](lessons/29%20Llama2%207B%20vs%20Llama3%208B/代码/tutorials/llama/代码注释.ipynb) |
| [30 InstructGPT reward modeling](lessons/30%20InstructGPT%20reward%20modeling/README.md) | [讲解](lessons/30%20InstructGPT%20reward%20modeling/解释.md) | [文本代码](lessons/30%20InstructGPT%20reward%20modeling/代码/tutorials/reward_model/reward_model_Bradley_Terry_model_代码注释.py) | [带注释代码](lessons/30%20InstructGPT%20reward%20modeling/代码/tutorials/reward_model/代码注释.ipynb) |

第 08、09、15、18、24、25、26、30 讲包含补充 notebook；第 09 讲还包含训练 tokenizer 的代码。点课程名进入该讲目录，可以看到完整阅读顺序。

<details>
<summary>需要对照原版、查证验证结果或维护项目时，再展开</summary>

整套文件验收见 [01–30讲交付核查](资料/课程交付核查.md)：40份原notebook、988个原单元、623个代码格、80份派生notebook及30个独立课程提交。

目录用途：`lessons/` 放正式课程；`资料/` 放进度与来源说明；`工具/` 放检查脚本。每讲的 `代码/` 放 notebook 和 Python 阅读版，`复现/` 放验证脚本与结果，`图片/` 放讲解配图。原版与派生版是不同用途的文件，保留它们便于对照。

| 看到的文件或目录 | 什么时候需要 |
|---|---|
| `解释.md`、`*_代码注释.py` | 日常学习 |
| `代码注释.ipynb` | 查看单元输出，在 notebook 中运行 |
| 上游原 `*.ipynb`、`原代码_带说明.ipynb` | 对照原课，核实注释是否改变代码 |
| `代码单元解释.md` | 按原单元编号定位说明 |
| `来源清单.json`、`复现/`、`资料/`、`工具/`、`AGENTS.md` | 来源查证、复现与项目维护 |
| 图片、tokenizer 文件、配套源码 | 代码或讲解依赖，不能只因不直接阅读就删除 |

- 各讲 `代码/tutorials/原文件名.ipynb`：上游原版。
- 各讲 `原代码_带说明.ipynb`：原代码不改，插入单元说明。
- 各讲 `代码单元解释.md`：按原单元编号查说明。
- 各讲 `复现/README.md`：实际运行范围、环境和重做命令。
- `来源清单.json`、`复现/outputs`：来源及验证证据，通常不用阅读。
- [全部代码覆盖进度](资料/代码覆盖进度.md)、[字幕术语校正](资料/字幕校正.md)、[发布说明](资料/发布说明.md)。
- `工具/` 与 `AGENTS.md`：维护检查及工作约定。

原教程：[chunhuizhang/personal_chatgpt](https://github.com/chunhuizhang/personal_chatgpt)。教学验证与原大模型训练分开记录，实际完成范围见各讲复现文档。

</details>
