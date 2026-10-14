# -*- coding: utf-8 -*-
"""真实HF tokenizer/collator/CLM loss追踪；独立补充，不改原课程代码。"""
import torch
from transformers import DataCollatorForLanguageModeling


def parameter_formula(vocab,hidden,intermediate,layers):
    """无bias、输入/输出不绑权重的标准LLaMA逻辑参数量。"""
    # 两个词表矩阵；每层四个attention投影、三个MLP投影、两个RMSNorm；末尾Norm。
    return 2*vocab*hidden+layers*(4*hidden**2+3*hidden*intermediate+2*hidden)+hidden


def inspect_supervision(trainer,raw_dataset):
    """直接检查真实SFTTrainer0.5的非packed数据和默认collator。"""
    tokenizer=trainer.tokenizer
    assert type(trainer.data_collator).__name__=='DataCollatorForLanguageModeling'
    assert tokenizer.pad_token_id==tokenizer.eos_token_id==2
    original_lengths=[len(tokenizer(row['text'])['input_ids']) for row in raw_dataset]
    processed_lengths=[len(row['input_ids']) for row in trainer.train_dataset]
    assert processed_lengths==[min(512,n) for n in original_lengths]
    assert all(row['input_ids'][0]==tokenizer.bos_token_id==1 for row in trainer.train_dataset)
    # 两条实际Guanaco文本的不同长度用于展示padding；正常token均被复制为监督。
    batch=trainer.data_collator([trainer.train_dataset[i] for i in (0,1)])
    valid=batch['attention_mask'].bool()
    assert torch.equal(batch['labels'][valid],batch['input_ids'][valid])
    assert batch['labels'][~valid].eq(-100).all()
    assert not batch['input_ids'][valid].eq(tokenizer.eos_token_id).any()

    # 角色标记是多token字符串；没有automatic assistant-only masking。
    human=tokenizer('### Human:',add_special_tokens=False)['input_ids']
    assistant=tokenizer('### Assistant:',add_special_tokens=False)['input_ids']
    # 去掉BOS后匹配首个Human标记，真实labels没有将它设为-100。
    ids=batch['input_ids'][0].tolist();start=next(i for i in range(len(ids)-len(human)+1) if ids[i:i+len(human)]==human)
    assert batch['labels'][0,start:start+len(human)].tolist()==human

    # 单独追加真实EOS，核对pad=eos按ID屏蔽不仅遮padding也遮结束符。
    features=[]
    for sample in ('Hello','A much longer response to this question.'):
        tokens=tokenizer(sample)['input_ids']+[tokenizer.eos_token_id]
        features.append(dict(input_ids=tokens,attention_mask=[1]*len(tokens)))
    lm=DataCollatorForLanguageModeling(tokenizer,mlm=False)(features)
    real_eos=lm['input_ids'].eq(2)&lm['attention_mask'].bool()
    assert real_eos.sum().item()==2 and lm['labels'][real_eos].eq(-100).all()
    # 使用真实模型验证一次shift：logits前T-1预测labels后T-1。
    model=trainer.model;model.eval()
    # Fast tokenizer默认左padding：机械取前32位可能全部是-100，CE会得到NaN。
    # 先记录这个实际边界，再从第一条的有效token区取32位验证shift，不改训练数据。
    prefix=batch['labels'][:1,:32]
    prefix_all_ignored=bool(prefix.eq(-100).all())
    if prefix_all_ignored:
        ignored_loss=torch.nn.functional.cross_entropy(torch.zeros(31,len(tokenizer)),prefix[:,1:].reshape(-1),ignore_index=-100)
        assert torch.isnan(ignored_loss)
    ids=batch['input_ids'][0,valid[0]][:32].unsqueeze(0)
    labels=batch['labels'][0,valid[0]][:32].unsqueeze(0)
    one=dict(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids))
    with torch.no_grad():
        result=model(**one)
        scores=result.logits[:,:-1,:].contiguous()
        targets=one['labels'][:,1:].contiguous()
        manual=torch.nn.functional.cross_entropy(scores.view(-1,scores.shape[-1]),targets.view(-1),ignore_index=-100)
    torch.testing.assert_close(result.loss,manual,rtol=0,atol=0)
    return dict(tokenizer_class=type(tokenizer).__name__,vocab=len(tokenizer),bos_id=1,eos_id=2,pad_id=2,
                original_lengths=original_lengths,processed_lengths=processed_lengths,
                truncated_tokens=[a-b for a,b in zip(original_lengths,processed_lengths)],
                batch_shape=list(batch['input_ids'].shape),padding_masked=int((~valid).sum()),
                whole_text_supervised=True,human_ids=human,assistant_ids=assistant,
                human_labels=batch['labels'][0,start:start+len(human)].tolist(),real_eos_masked=2,
                padding_side=tokenizer.padding_side,left_prefix32_all_ignored=prefix_all_ignored,
                all_ignored_CE='NaN' if prefix_all_ignored else 'not tested',shift_audit_uses_first32_valid_tokens=True,
                shift_loss=float(result.loss),manual_shift_loss=float(manual),logits_shape=list(result.logits.shape))
