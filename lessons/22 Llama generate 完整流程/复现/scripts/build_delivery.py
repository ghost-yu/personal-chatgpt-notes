# -*- coding: utf-8 -*-
"""Exact upstream notebook/tokenizer and comment-only full generate reading material."""
import argparse
import ast
import copy
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
SHAS={'llama-generation.py':'acc739aeda4aa222305ca5e7ae020df38d0fa841530d966f041c13e065db05c4',
      'llama-model.py':'56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754',
      'llama-tokenizer.py':'a675762526cd5f6e9d742a642e33a3560ea825522b8a9efb531d36b7c7a69e6c'}
NOTES={
0:'【必须会】text是Python str，内容为I believe the meaning of life is。它还不是Tensor，也没有batch轴；模型稍后接收的是整数token ID。赋值本身不生成文本。',
1:'【必须会】导入SentencePieceProcessor并从./tokenizer.model建立实例。相对路径取决于当前工作目录，应从代码/tutorials启动内核。这里加载的是分词模型，不是Llama神经网络权重。',
2:'【必须会】四次print读取真实特殊ID/词表：BOS=1、EOS=2、PAD=-1、词表32000。PAD=-1是占位哨兵，不是合法embedding索引；完整循环必须在负ID被送入embedding之前填好有效前缀。这里的ID不能直接套到其他tokenizer。',
3:'【必须会】encode(text)返回7个普通整数；列表中的*把它们展开，前置BOS后得到[1,306,4658,278,6593,310,2834,338]。末表达式在notebook显示列表。它是List[int]单条提示；generate要求List[List[int]]批次，text_completion会负责把每条文本编码后组成batch，直接把本变量传给generate会报类型错误。',
4:'''【必须会】这是self.generate的调用片段，generation_tokens/generation_logprobs用二元解包接收返回值。默认echo=False，仅返回生成部分；True从开头保留prompt，不过末尾仍会按EOS截断。返回token IDs，不是已decode的str。logprobs=False时第二项是None。

max_gen_len是每条提示词最多保留的新增数量，total_len=min(max_seq_len,max_gen_len+max_prompt_len)是整个批次统一容量。max_prompt_len取最长提示，而循环从最短提示长度开始。8+64=72只是示例，实际还受容量限制；参数max_seq_len不是Llama2全部型号永远固定1024或2048的定义。''',
5:'【建议会】basics把前几讲的模型模块串联起来。本讲重点为生成控制流，不额外发明训练loss、optimizer或数据集。',
6:'【建议会】列出RMSNorm/SwiGLU、两段RoPE、KV cache、GQA与采样的相关视频。来源链接保留；nuclear sampling应校正为nucleus sampling，中文核采样。这个generate实现有temperature和top_p，没有实现top_k参数，不能把视频标题当成本函数签名。',
7:'【必须会】Llama.generate是Llama实例的方法，使用self.model和self.tokenizer。完整方法有@torch.inference_mode()，关闭梯度记录，但不会自动调用model.eval()。构造/加载模型和设置推理状态是其他环节；本次不执行需要NCCL/checkpoint的build。',
8:'【必须会】表中heads是Q heads；D/Hq都得到128。模型层数/通道影响一次forward，生成循环的起止位置由token长度决定，不由head数决定。70B的KV heads不能从heads列直接读成64，分组见第21讲。',
9:'【必须会】标题省略了切片细节，真正调用为model.forward(tokens[:,prev_pos:cur_pos],prev_pos)。模型返回浮点[B,S,V]，其中S是本次送入长度，不是累计输出长度；循环用logits[:,-1]的词表向量预测当前位置cur_pos的token。',
10:'''【必须会】range(min_prompt_len,total_len)右端不包含。prev_pos初始0，第一次送[0:min_prompt_len]，随后完整方法在写token后令prev_pos=cur_pos，下一次才成为单token切片。单条8-token prompt的调用是[0:8]、[8:9]、[9:10]。

本片段缺prev_pos更新、写token、采样和EOS逻辑，按字面独立执行会一直用prev_pos=0重算完整前缀，不能冒称它自己已实现增量。原句“第一次prompt整体进来”只对单条/等长batch成立，不等长batch首次只处理最短长度，长提示的剩余token由input_text_mask逐位置保护。多token首次prefill需因果上三角mask；后续单token没有chunk内部未来位置。''',
11:'【建议会】这是手写的累计文本示意，原文meanning拼写与text格不同；保留原文，并不认为这些续写是本次或原7B实际输出。真实过程先产生token ID，token边界不一定等于一个英文单词，最后decode才得到文本。',
12:'【必须会】进入缓存环节。KV cache属于每层Attention，不是generate在Python列表中拼接的隐藏状态。tokens保存输入/输出整数，cache保存带位置的K及未旋转V；两者内容、dtype、shape都不同。',
13:'''【必须会】完整接口接受[B,S]，S不总等于1：首次prefill通常S>1，随后decode常为1。原“每次只接受一个token”的说法需要限定后续decode。切片读取cache[:B,:P+S]包含此前历史与本次新KV，不包含未用容量尾部。cache本体仍是[Bmax,Tmax,Hkv_local,d]。

两条keys/values语句属于Attention.forward，其中self是Attention；调用generate时self却是Llama。相同self名字并不代表同一个对象。摘录需要提供bsz/start_pos/seqlen等上下文，不能在Llama实例上直接取cache_k。''',
14:'【建议会】区分返回控制（logprobs/echo）和选词控制（temperature/top_p）。改变echo不改变同一随机种子下的已生成ID，只改变最后切片；logprobs理论上是概率报告，但固定旧版存在记录顺序缺陷，见本讲实测。',
15:'''【必须会】温度公式为p_i(T)=exp(z_i/T)/sum_j exp(z_j/T)，z_i是当前最后位置第i个词表logit，T>0，沿词表归一化。正温度改变分布尖锐度，理论上不改变最大logit索引；熵在非退化例子中随T增加。T=0不能直接代入除法，代码走argmax；负温度也落入else，不做负T softmax。

top_p是在温度softmax之后按概率降序保留动态候选集合、重归一化再采样，不是固定top_k。输出token_logprobs的目标是原模型的log概率，不能直接等同于温度/核截断后采样分布概率；旧版把尚为PAD的新增target先送入CE，实际新增返回-0，不能当真实概率。''',
16:'''【必须会】if temperature>0时，logits[:,-1]从[B,S,V]取最后位置得到[B,V]；除标量温度，softmax(dim=-1)沿词表得到概率，sample_top_p返回int64[B,1]。else下torch.argmax(...,dim=-1)返回[B]，所以完整generate随后reshape(-1)统一为[B]。

sample_top_p先sort得到概率与原词表索引，cumsum得到前缀和，用(cumsum-current_prob)>p遮掉候选，原地重归一化，multinomial实际采样，再gather映射回原token ID。使用的是“当前候选之前的累计质量”，因此保留跨过阈值的那一项，恰好等于阈值时也继续保留下一项。它未主动校验p范围；负p可能导致全零/NaN错误。原片段不包含提示词保护、cache更新或停止逻辑，完整流程见官方方法注释。'''
}

def note(s):
    rules=[('def __init__','保存真实Transformer与Tokenizer实例，generate依赖这两个状态对象。'),
    ('@torch.inference_mode','装饰器关闭梯度记录，不自动切换eval模式。'),('def generate','实例方法输入二维token列表，返回每样本token列表与可选logprobs。'),
    ('params = self.model.params','读取模型容量配置，不在此加载权重。'),('bsz = len','批次数是外层列表长度，单条list[int]需先组成批次。'),
    ('assert bsz','断言不能超过最大batch容量。'),('min_prompt_len =','取最短提示长度作为首次prefill终点。'),('max_prompt_len =','最长提示决定批次统一容量。'),
    ('assert max_prompt_len','检查最长prompt不超过模型容量。'),('total_len =','全批次分配长度=min(模型上限，最长prompt+新增上限)。'),
    ('pad_id =','真实SentencePiece此处PAD=-1，仅占位，不得传入embedding。'),('tokens = torch.full','分配int64[B,total_len]，原版强制CUDA，初始全为PAD。'),
    ('for k, t in enumerate','enumerate同时返回样本索引与该条token列表。'),('tokens[k,','将原prompt拷到各行前缀；不等长空位暂时保留PAD。'),
    ('if logprobs:','控制是否记录token对数概率；原版循环内CE发生在填新增token之前。'),('token_logprobs = torch.zeros','分配FP32[B,total_len]，第一个token没有此前logit而保持0。'),
    ('prev_pos = 0','首轮从token0开始；后续更新才避免重算。'),('eos_reached =','每样本布尔停止标记；batch要全部True才整体退出。'),
    ('input_text_mask =','固定布尔mask保存哪些位置原先属于提示词，不能由采样覆盖。'),('for cur_pos in range','从最短prompt末尾开始，右端total_len不包含。'),
    ('logits = self.model.forward','仅传尚未处理的[prev_pos:cur_pos]，起始绝对位置为prev_pos。'),
    ('token_logprobs[:,','原版此时tokens当前新增槽仍为PAD；ignore_index令新增概率被错误记成0。'),
    ('input=logits.transpose','CE需要[B,V,S]，交换位置轴与词表轴。'),('target=tokens','目标相对输入右移一位；新增最后目标尚未填。'),
    ('ignore_index=pad_id','PAD=-1不参与CE，解释固定旧版新增logprobs为-0的原因。'),
    ('if temperature >','正温度才softmax采样；0或负值走greedy。'),('probs = torch.softmax','取最后位置词表分布[B,V]，温度缩放后归一化。'),
    ('next_token = sample_top_p','真实核采样，返回词表ID[B,1]。'),('next_token = torch.argmax','greedy直接选最大logit，[B]；不使用top_p。'),
    ('next_token = next_token.reshape','把采样[B,1]或greedy[B]统一成[B]。'),('next_token = torch.where','原prompt位置选原ID，空位才选新token。'),
    ('input_text_mask[:,','True表示原prompt，保留tokens中对应原值。'),('tokens[:, cur_pos] =','将所选token真正写回当前列；已EOS的行也未被冻结。'),
    ('eos_reached |=','只把新生成EOS置True，原prompt内EOS不在此触发。'),('prev_pos = cur_pos','下一轮只处理刚写入的token；cache保存之前的位置。'),
    ('if all(eos_reached)','全部样本都遇到生成EOS才提前退出批次。'),('token_logprobs = token_logprobs.tolist','张量转普通嵌套浮点列表供返回。'),
    ('out_tokens, out_logprobs','准备逐样本输出列表，不是另一次生成。'),('for i, toks in enumerate','按样本取整行token，进行返回裁剪。'),
    ('start = 0 if echo','echo控制返回起点：0或本条prompt长度。'),('toks = toks[start','每条最多保留自身prompt长度+max_gen_len以内的位置。'),
    ('probs = token_logprobs[i]','概率使用与token同样的起止切片。'),('if self.tokenizer.eos_id in toks','在所返回片段中找首个EOS，echo时也可能找到prompt里的EOS。'),
    ('eos_idx =','list.index返回第一个EOS位置。'),('toks = toks[:eos_idx]','不返回EOS本身及之后token。'),('probs = probs[:eos_idx]','概率与token一起截断，维持对齐。'),
    ('out_tokens.append','收集当前样本token列表。'),('out_logprobs.append','logprobs=False时只临时收None，最终第二返回值为None。'),
    ('return (out_tokens','返回二元tuple；不在此decode成字符串。'),('def text_completion','外层封装把List[str]变二维IDs并decode结果。'),
    ('prompt_tokens = [','每条文本编码，通常加BOS、不加EOS；形成batch。'),('generation_tokens, generation_logprobs','调用generate并解包两个返回项。'),
    ('def sample_top_p','普通函数接收[B,V]概率和阈值，返回[B,1]整数token ID。'),('probs_sort, probs_idx','降序排序，索引表用于回到原词表ID。'),
    ('probs_sum = torch.cumsum','沿词表累计排序概率。'),('mask = probs_sum','用当前项之前的累计质量判断，保留越过阈值的一项。'),
    ('probs_sort[mask]','被排除候选概率置0，不修改原probs输入。'),('probs_sort.div_','原地除保留质量，重新归一化；非法p可能导致除0。'),
    ('next_token = torch.multinomial','按概率抽样一项，这是真实随机步骤，不是argmax。'),('next_token = torch.gather','把排序后位置映射回原词表ID。'),('return next_token','返回采样ID，[B,1]。')]
    for prefix,text in rules:
        if s.startswith(prefix):return text
    return None

def annotate(src):
    result=[];doc=False
    for line in src.splitlines(keepends=True):
        if '"""' in line:
            if line.count('"""')%2:doc=not doc
            result.append(line);continue
        n=note(line.strip())
        if n and not doc:result.append(line[:len(line)-len(line.lstrip())]+'# '+n+'\n')
        result.append(line)
    return ''.join(result)

def extract(src,node):
    start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return textwrap.dedent(''.join(src.splitlines(keepends=True)[start-1:node.end_lineno]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--sources',required=True);a=p.parse_args()
    code=ROOT/'代码/tutorials';code.mkdir(parents=True,exist_ok=True);entries=[]
    for name in ('tutorials/llama2_src_generate.ipynb','tutorials/tokenizer.model'):
        b=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=a.upstream);target=ROOT/'代码'/name;target.write_bytes(b)
        entries.append({'upstream_path':name,'published_path':'代码/'+name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    for name,h in SHAS.items():assert hashlib.sha256((Path(a.sources)/name).read_bytes()).hexdigest()==h
    manifest={'repository':'https://github.com/chunhuizhang/personal_chatgpt','commit':COMMIT,'files':entries,
              'official_meta_commit':'7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e','official_source_sha256':SHAS}
    (ROOT/'来源清单.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    original=json.loads((code/'llama2_src_generate.ipynb').read_bytes());assert len(original['cells'])==17
    intro={'cell_type':'markdown','metadata':{},'source':['# 第22讲：Llama generate完整流程\n\n原单元从0编号；原代码、Markdown与历史输出保留。完整generate与CPU教学验证是明确补充，未加载原7B权重。\n']}
    explained=copy.deepcopy(original);annotated=copy.deepcopy(original);explained['cells']=[copy.deepcopy(intro)];annotated['cells']=[copy.deepcopy(intro)]
    md=['# 第22讲全部单元说明\n\n17格：4代码、13Markdown；重要Markdown4/10/13/16源码完整注释。\n'];py='# -*- coding: utf-8 -*-\n# 原课程四格分词示例；完整生成方法见generate核心源码_代码注释.py。\n'
    snippets='# -*- coding: utf-8 -*-\n# 原Markdown片段需要各自上下文；供阅读，不能直接当完整generate运行。\n'
    for i,c in enumerate(original['cells']):
        extra=''
        if i in (4,10,13,16):
            raw=''.join(c['source']).split('```')[1].strip('\n')+'\n';body=annotate(textwrap.dedent(raw))
            display_body='\n'.join(line.rstrip() for line in body.splitlines())
            extra='\n\n```python\n'+display_body.rstrip()+'\n```\n';snippets+=f'\n# %% 原Markdown {i}\n'+body+'\n'
        text=f'## 原单元 {i}（{c["cell_type"]}）\n\n'+NOTES[i]+extra+'\n';n={'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
        explained['cells']+=[copy.deepcopy(n),copy.deepcopy(c)];new=copy.deepcopy(c)
        if c['cell_type']=='code':
            body='# '+{0:'输入文本是str，还不是token Tensor。',1:'读取真实SentencePiece模型，相对路径取决于工作目录。',2:'打印BOS/EOS/PAD及词表大小，PAD=-1只作占位。',3:'*展开编码ID；这是单条列表，generate需要再包成二维batch。'}[i]+'\n'+''.join(c['source'])
            new['source']=body.splitlines(keepends=True);py+=f'\n# %% 原代码单元 {i}\n'+body+'\n'
        annotated['cells']+=[copy.deepcopy(n),new];md.append(text)
    for name,nb in [('原代码_带说明.ipynb',explained),('代码注释.ipynb',annotated)]:
        (code/name).write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    (ROOT/'代码单元解释.md').write_bytes(('\n'.join(md).rstrip()+'\n').encode('utf-8'))
    (code/'llama2_src_generate_代码注释.py').write_bytes((py.rstrip()+'\n').encode('utf-8'))
    (code/'Markdown片段_代码注释.py').write_bytes(('\n'.join(s.rstrip() for s in snippets.splitlines()).rstrip()+'\n').encode('utf-8'))
    src=(Path(a.sources)/'llama-generation.py').read_text(encoding='utf-8');tree=ast.parse(src);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Llama')
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','generate','text_completion')]
    reading='# -*- coding: utf-8 -*-\n# Copyright (c) Meta Platforms, Inc. and affiliates.\n# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.\n# 固定Meta完整generate、text_completion和sample_top_p，只增加中文注释。\n# 仅选取Llama的相关方法，非完整Llama类；依赖原模块的typing/torch/F/Transformer/Tokenizer等。\n\nclass Llama:\n'
    for n in selected:reading+=textwrap.indent(annotate(extract(src,n)),'    ')+'\n'
    sample=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_top_p')
    reading+='\n'+annotate(extract(src,sample))
    (code/'generate核心源码_代码注释.py').write_bytes(reading.encode('utf-8'))
    print('Built17-cell coverage, two35-cell notebooks, three Python reading files and exact originals.')

if __name__=='__main__':main()
