# -*- coding: utf-8 -*-
"""真实旧版 TRL/Torch 标签演示；独立教学补充，不是原课程单元。"""
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from trl.trainer.utils import ConstantLengthDataset


def make_teaching_dataset():
    """自建六行 instruction/input/output/text；不冒充 Alpaca 数据下载。"""
    rows = []
    for i in range(6):
        instruction = "Explain addition in one short sentence."
        context = "Two apples and three apples." if i == 5 else ""
        answer = "Addition combines quantities into a total."
        text = "Below is an instruction. Write a response.\n\n### Instruction:\n" + instruction
        if context:
            text += "\n\n### Input:\n" + context
        text += "\n\n### Response:\n" + answer
        rows.append(dict(instruction=instruction, input=context, output=answer, text=text))
    return Dataset.from_list(rows)


def audit_labels(tokenizer, dataset, seq_length=32):
    """用真实分词器、ConstantLengthDataset 和两个 collator 追踪标签。"""
    # shuffle=False 仅供核对顺序；原 SFTTrainer 的 packing 默认会打乱块。
    packed = ConstantLengthDataset(
        tokenizer, dataset, dataset_text_field="text", seq_length=seq_length,
        infinite=False, shuffle=False,
    )
    blocks = list(packed)
    # 每个完整文本后追加一次 EOS，然后跨样本切块；最后不足32个token的尾块丢弃。
    expected = []
    for row in dataset:
        expected.extend(tokenizer(row["text"])["input_ids"] + [tokenizer.eos_token_id])
    kept = len(expected) // seq_length * seq_length
    assert torch.cat([b["input_ids"] for b in blocks]).tolist() == expected[:kept]
    assert all(torch.equal(b["labels"], b["input_ids"]) for b in blocks)
    assert len(packed) == len(dataset)  # 旧类报告原始行数，实际产出的块数另数。

    # packed路径的默认padding collator不重写labels，因此真实EOS仍是学习目标。
    batch = DataCollatorWithPadding(tokenizer)(blocks)
    eos = batch["input_ids"].eq(tokenizer.eos_token_id)
    assert eos.any() and torch.equal(batch["labels"][eos], batch["input_ids"][eos])
    assert batch["input_ids"].dtype == batch["labels"].dtype == torch.int64

    # 为了同时观察真实EOS和padding，明确在两条不同长度文本末尾追加EOS。
    features = []
    for text in ("Hello", "Hello, this is a longer response."):
        ids = tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
        features.append(dict(input_ids=ids, attention_mask=[1] * len(ids)))
    lm_batch = DataCollatorForLanguageModeling(tokenizer, mlm=False)(features)
    all_eos = lm_batch["input_ids"].eq(tokenizer.eos_token_id)
    real_eos = all_eos & lm_batch["attention_mask"].bool()
    padding = ~lm_batch["attention_mask"].bool()
    assert real_eos.sum().item() == 2 and padding.any()
    # pad=eos时按ID而非attention_mask遮标签，真实结束符也被设为-100。
    assert lm_batch["labels"][all_eos].eq(-100).all()
    assert lm_batch["labels"][padding].eq(-100).all()
    report = dict(
        raw_rows=len(dataset), reported_length=len(packed), actual_blocks=len(blocks),
        concatenated_tokens=len(expected), kept_tokens=kept, dropped_tail=len(expected)-kept,
        packed_shape=list(batch["input_ids"].shape), packed_dtype=str(batch["input_ids"].dtype),
        packed_supervised_eos=int(eos.sum()), nonpacked_real_eos_masked=int(real_eos.sum()),
        nonpacked_padding_masked=int(padding.sum()),
        sample_ids=batch["input_ids"][0].tolist(), sample_labels=batch["labels"][0].tolist(),
        nonpacked_ids=lm_batch["input_ids"].tolist(), nonpacked_labels=lm_batch["labels"].tolist(),
        attention_masks=lm_batch["attention_mask"].tolist(),
    )
    return report, blocks[0]


def shifted_cross_entropy(logits, labels):
    """logits[B,T,V]的前T-1步预测labels[B,T]的后T-1个ID，模型内也只shift一次。"""
    scores = logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        scores.view(-1, scores.shape[-1]), targets.view(-1), ignore_index=-100,
    )
