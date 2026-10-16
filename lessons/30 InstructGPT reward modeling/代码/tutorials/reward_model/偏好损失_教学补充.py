"""本讲新增的真实小Llama reward模型实验，不是原InstructGPT训练工程。

输入是自建token IDs，只用于验证数据流/分组loss/梯度，不代表语言质量。
"""
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForSequenceClassification


def strict_pairs(ranks):
    """ranks是一维候选等级列表，数值越大越好；相等丢弃。

    返回(chosen下标,rejected下标)列表，每个无序对只输出一次。
    """
    pairs=[]
    for i in range(len(ranks)):
        for j in range(i+1,len(ranks)):
            if ranks[i]==ranks[j]:
                continue
            pairs.append((i,j) if ranks[i]>ranks[j] else (j,i))
    return pairs


def grouped_pair_loss(rewards,groups,ranks):
    """rewards浮点[N]；groups为(start,stop)，每个slice对应同prompt候选。

    ranks为逐prompt的等级列表。这里先对每prompt保留的严格pair取mean，
    再对prompt取mean；这是明确教学选择，不冒称原OpenAI内部ties权重实现。
    返回标量loss。无严格比较的prompt报错，避免空mean变NaN。
    """
    per_prompt=[]
    for (start,stop),rank in zip(groups,ranks,strict=True):
        if stop-start!=len(rank):
            raise ValueError('候选数量与等级数量不匹配')
        pairs=strict_pairs(rank)
        if not pairs:
            raise ValueError('该prompt没有严格偏好对')
        # 同一个reward向量复用到多个pair，不重复运行骨干。
        delta=torch.stack([rewards[start+w]-rewards[start+l] for w,l in pairs])
        # -logsigmoid(delta)=softplus(-delta)，稳定且可反向传播。
        per_prompt.append(F.softplus(-delta).mean())
    if not per_prompt:
        raise ValueError('没有prompt')
    return torch.stack(per_prompt).mean()


def make_experiment(seed=30):
    """返回真实随机HF模型、右pad ID/mask、prompt切片和人工教学等级。"""
    torch.manual_seed(seed)
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
        num_labels=1,pad_token_id=0,bos_token_id=1,eos_token_id=2,
        attention_dropout=0.0,max_position_embeddings=32)
    # num_labels=1只让score成为标量；不使用其默认MSE labels分支。
    model=LlamaForSequenceClassification(config)
    samples=[[1,3,7,2],[1,3,8,2],[1,3,9,10,2],[1,3,11,12,13,2],
             [1,4,14,15,2],[1,4,16,2],[1,4,17,18,2]]
    # 7个不同(prompt,response)序列；第一组4候选，第二组3候选。
    ids=torch.zeros(7,6,dtype=torch.long)
    for row,tokens in enumerate(samples):
        ids[row,:len(tokens)]=torch.tensor(tokens)
    mask=ids.ne(0).long()
    # 第一个prompt D>C>A=B，第二个prompt A>B>C，分别5和3个严格pair。
    return model,ids,mask,[(0,4),(4,7)],[[0,0,1,2],[2,1,0]]


def train_demo(steps=8):
    """完整最小训练：实际Llama→末token标量→分组BT loss→AdamW。"""
    model,ids,mask,groups,ranks=make_experiment()
    # eval关闭训练态随机性，但不关闭autograd；用来稳定比较不同打包方式。
    model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.005)
    trace=[]
    for step in range(steps):
        optimizer.zero_grad()
        rewards=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.squeeze(-1)
        loss=grouped_pair_loss(rewards,groups,ranks)
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('出现缺失或非有限梯度')
        optimizer.step()
        trace.append({'step':step+1,'loss_before_update':loss.item()})
    return model,trace,(ids,mask,groups,ranks)


if __name__=='__main__':
    torch.set_num_threads(1)
    _,trace,_=train_demo()
    for row in trace:
        print(row)
