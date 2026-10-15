# -*- coding: utf-8 -*-
"""Extract exact original blobs and build comment-only, fully explained GQA reading files."""
import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COMMIT='7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
META_SHA='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
NOTES={
0: '【建议会】from IPython.display import Image 导入展示图片的类。导入不建立Attention，不加载模型参数，没有Tensor输入输出。后面的Image构造对象时读取图片字节，Jupyter将末尾表达式显示为图片；普通.py不会自动显示。',
1: '''【必须会】表中heads是query head数量Hq，不是KV heads。7B/13B/70B的D分别4096/5120/8192，Hq分别32/40/64，所以每head通道d均为128。layers是block数量，影响共有多少份KV cache。

原句n_kv_heads=8不能套用到全部三种型号。课程列出的已发布Llama2中，7B/13B采用MHA，Hkv分别32/40；70B采用8个KV heads，Hq/Hkv=8。Meta源码用args.n_kv_heads=None作为默认值，再令Hkv=Hq，不等于“代码中没有该配置字段”。Llama2论文Table1还列出研究用34B采用GQA，因此“只有70B”需要限定为本课程列出的已发布型号。真实源码约束是分组与并行分片的整除，不是所有GQA都必须取8。''',
2: '【必须会】Image("../imgs/gqa.png",width=600)读取原图，宽600仅为展示设置。请从代码/tutorials启动内核。按图从下往上看：蓝色是Q，红色K，黄色V。左MHA有8Q/8KV；中GQA是8Q/4KV，每两个相邻Q共享一组K/V；右MQA是8Q/1KV。虚线表达共享关系，不是说Q数跟着KV一起减少。图只示意head，未画batch/token/d、RoPE、mask或最终输出投影。',
3: '''【必须会】MHA=multi-head attention，Hkv=Hq；MQA=multi-query attention，Hkv=1；GQA的中间情形1<Hkv<Hq。Hq一般保持不变，所以每个Q head仍有自己的注意力权重和输出。减的是KV投影通道及持久缓存。

原文给MHA写computationally low容易误导：在相同Hq/d/token长度下，MHA不具备MQA的KV缓存节省。不能把MHA/MQA简单标为固定的高/低“质量”；论文比较取决于具体训练、模型和指标。原GQA论文Table1是T5实验，不能直接当Llama2原模型性能，也不能把共享当作无损地改任意checkpoint。

计算注意力分数的Q head数仍为Hq，naive repeat实现还可能创建临时展开副本。持久KV容量降低不意味着所有计算、总模型参数、延迟都按相同比例降低。''',
4: '''【必须会】原段中的“multiple key-value heads share the same query”把共享方向说反了：多个Q heads共享一个K head和对应V head，Q彼此不同。下面给出三条view、n_rep和两条local-head计算的逐语句注释，原Markdown保留。

xq/xk/xv来自不同可训练投影，是浮点Tensor，不是token整数。D为输入隐藏通道，B=bsz，S=seqlen，d=head_dim。Q投影输出[B,S,Hq_local*d]，view拆成[B,S,Hq_local,d]；K/V是[B,S,Hkv_local*d]，view拆成[B,S,Hkv_local,d]。view不把所有head平均，也不改变元素数量；原片段未给出三投影、bsz/seqlen赋值、cache或最终输出，不能单独运行。

self.n_rep=Hq_local//Hkv_local是每组有多少Q heads。//是整数向下取整，不是合法性检查；配置需保证整除，否则可能在后续matmul报head维不匹配。n_rep不是KV组数，组数是Hkv。70B例子64/8=8，图中则8/4=2，二者都成立。

原文行内n_kv_heads == 8使用双等号，是比较表达式，返回bool，并不会把变量设成8；赋值使用单等号。这里它描述课程70B配置，不是通用模型要求。

model_parallel_size是模型并行rank数量，不是batch。两条local赋值将全局heads按rank切分；该固定简单实现要求head能均匀分片，Hkv过小的MQA不能直接任意增加rank数。实际验证只运行真实FairScale的world_size=1；没有通过算整除关系冒称多rank通信成功。

真正对齐heads的repeat_kv没有出现在原notebook。已另补固定官方函数和完整Attention类注释：输入[B,T,G,d]，新插入重复轴后expand为[B,T,G,R,d]，再reshape为[B,T,G*R,d]。这里G=Hkv_local,R=n_rep，顺序是每组连续重复，不是把整套head循环复制。

完整Attention先把少量KV写入缓存，再对有效前缀repeat_kv，最终QKᵀ→mask→softmax→乘V→合并heads→wo。输出[B,S,D]；每层cache被更新。原两代码格只导入和展示图片，所以官方前向、梯度及转换验证是明确补充，不能说它们原本就在notebook里。'''
}

SNIPPETS=[
('xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)','Q拆成[B,S,Hq_local,d]，query head数保持不变。'),
('xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)','K拆成[B,S,Hkv_local,d]，仅投影较少的KV通道。'),
('xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)','V与K共用分组数量，但投影权重和内容不同。'),
('self.n_rep = self.n_local_heads // self.n_local_kv_heads','每组有R个Q heads，//不自动检查是否整除。'),
('self.n_local_heads = args.n_heads // model_parallel_size','当前rank的Q head数；world_size=1时等于全局数。'),
('self.n_local_kv_heads = self.n_kv_heads // model_parallel_size','当前rank的KV head数，不是query数量，也不是batch。')]

def line_comment(s):
    if s.startswith('def repeat_kv'):return '普通函数，不是模型方法；输入[B,T,G,d]，返回[B,T,G*R,d]，无可训练参数。'
    if s.startswith('bs, slen, n_kv_heads, head_dim ='):return '解包四轴：batch、已用序列长度、KV groups、每head通道。'
    if s.startswith('if n_rep == 1'):return 'R=1不需重复；MHA情况直接复用原Tensor对象。'
    if s=='return x':return '没有复制；返回对象与输入是同一个。'
    if 'x[:, :, :, None, :]' in s:return 'None插入新轴，[B,T,G,d]变为[B,T,G,1,d]。'
    if '.expand(bs,' in s:return '广播重复轴到R；通常是stride=0的共享存储view，此步不复制元素。'
    if '.reshape(bs,' in s:return '合并G/R轴，相邻Q heads拿到同一组KV；必要时reshape会分配新存储。'
    if s.startswith('class Attention'):return '完整固定官方Attention类；依赖原model.py的其他导入和helper。'
    if s.startswith('super().__init__'):return '初始化nn.Module，随后赋给self的线性层会注册为子模块。'
    if s.startswith('self.n_kv_heads ='):return 'None默认退化为MHA；否则使用指定的KV groups。'
    if s.startswith('model_parallel_size ='):return '取得真实FairScale模型并行组大小；本次只执行单rank。'
    if s.startswith('self.n_local_heads ='):return '按rank拆分Q heads，要求配置适配该并行度。'
    if s.startswith('self.n_local_kv_heads ='):return 'KV heads也按rank拆分，不能在这里直接让每rank少于一个head。'
    if s.startswith('self.n_rep ='):return '每KV组服务的Q heads个数R=Hq_local/Hkv_local。'
    if s.startswith('self.head_dim ='):return '每head通道d，标准配置中D/Hq应为整数。'
    if s.startswith('self.wq ='):return 'Q投影仍输出全部Q heads，GQA不会把Q数一起减少。'
    if s.startswith(('self.wk =','self.wv =')):return 'K/V投影只输出Hkv*d通道，因此参数和缓存较小。'
    if s.startswith('self.wo ='):return '拼回所有Q head输出后投影回隐藏通道D。'
    if s.startswith(('self.cache_k = torch.zeros','self.cache_v = torch.zeros')):return '持久cache按Hkv_local分配，不按repeat后的Hq_local分配。'
    if s=='self.n_local_kv_heads,':return '缓存轴2是较少的KV groups；这决定持久内存节省。'
    if ').cuda()' in s:return '原版强制CUDA；教学执行只显式修改这两处cache设备调用。'
    if s.startswith('bsz, seqlen, _ ='):return 'x为[B,S,D]隐藏状态，读出batch和本次token数。'
    if s.startswith('xq, xk, xv ='):return '三组不同权重投影；K/V输出宽度可能比Q小。'
    if s.startswith('xq = xq.view'):return 'Q拆为[B,S,Hq_local,d]。'
    if s.startswith(('xk = xk.view','xv = xv.view')):return 'K/V拆为[B,S,Hkv_local,d]；仅head轴比Q少。'
    if s.startswith('xq, xk = apply_rotary_emb'):return '给Q/K同位置RoPE；同组Q仍是各自不同的query，V不旋转。'
    if '.to(xq)' in s:return 'cache匹配Q设备与dtype，不是保存Q。'
    if 'start_pos : start_pos + seqlen] =' in s:return '只将本次少量KV写入对应token槽，不在cache中重复成Q head数。'
    if s.startswith(('keys = self.cache','values = self.cache')):return '读取有效历史与当前token，shape[B,P+S,Hkv_local,d]。'
    if s.startswith(('keys = repeat_kv','values = repeat_kv')):return '为通用attention矩阵乘法把每组KV连续重复R次，对齐Hq_local。'
    if '.transpose(1, 2)' in s:return '交换token/head轴以按head独立计算。'
    if s.startswith('scores = torch.matmul'):return 'QKᵀ/√d得[B,Hq_local,S,P+S]，分数head数仍是Hq。'
    if s.startswith('scores = scores +'):return '因果mask约束可见位置；GQA本身不改变因果规则。'
    if s.startswith('scores = F.softmax'):return '每个Q head独立沿key位置softmax，先FP32再转回Q dtype。'
    if s.startswith('output = torch.matmul'):return '各Q head使用自己的权重聚合共享V；输出仍有全部Q heads。'
    if s.startswith('output = output.transpose'):return '整理内存并合并Q heads，得到[B,S,Hq_local*d]。'
    if s.startswith('return self.wo'):return '最终输出[B,S,D]并交给上层残差；cache已被原地更新。'
    return None

def annotate_official(source):
    result=[];in_doc=False
    for line in source.splitlines(keepends=True):
        s=line.strip()
        if '"""' in line:
            if line.count('"""')%2:in_doc=not in_doc
            result.append(line);continue
        note=line_comment(s)
        if note and not in_doc:result.append(line[:len(line)-len(line.lstrip())]+'# '+note+'\n')
        result.append(line)
    return ''.join(result)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--upstream',required=True);parser.add_argument('--meta-source',required=True)
    args=parser.parse_args();code=ROOT/'代码/tutorials';entries=[]
    for name in ('tutorials/llama2_src_grouped_query_attention.ipynb','imgs/gqa.png'):
        data=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=args.upstream)
        p=ROOT/'代码'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        entries.append({'upstream_path':name,'published_path':'代码/'+name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    meta=Path(args.meta_source).read_bytes();assert hashlib.sha256(meta).hexdigest()==META_SHA
    manifest={'repository':'https://github.com/chunhuizhang/personal_chatgpt','commit':COMMIT,'files':entries,
              'official_meta':{'repository':'https://github.com/meta-llama/llama','commit':'7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e','path':'llama/model.py','sha256':META_SHA}}
    (ROOT/'来源清单.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    original=json.loads((code/'llama2_src_grouped_query_attention.ipynb').read_bytes());assert len(original['cells'])==5
    src=''.join(original['cells'][4]['source'])
    for statement,_ in SNIPPETS:assert statement in src
    mdpy='# -*- coding: utf-8 -*-\n# 原Markdown4的六条语句；仅供阅读，变量需由完整Attention上下文提供。\n\n'
    for statement,note in SNIPPETS:mdpy+='# '+note+'\n'+statement+'\n\n'
    mdpy=mdpy.rstrip()+'\n';(code/'Markdown源码_代码注释.py').write_bytes(mdpy.encode('utf-8'))
    intro={'cell_type':'markdown','metadata':{},'source':['# 第21讲：GQA 与分组查询注意力\n\n原单元从0编号。原文/历史输出保留，新增说明中校正共享方向和配置范围。原notebook只有两格展示代码；官方repeat_kv/Attention验证为本项目补充。\n']}
    explained=copy.deepcopy(original);annotated=copy.deepcopy(original)
    explained['cells']=[copy.deepcopy(intro)];annotated['cells']=[copy.deepcopy(intro)]
    notes=['# 第21讲全部单元说明\n\n共5格：2代码、3Markdown；全部原内容/历史输出保留。\n']
    py='# -*- coding: utf-8 -*-\n# 原课程只导入Image并展示配图；核心官方代码见GQA核心源码_代码注释.py。\n'
    for i,c in enumerate(original['cells']):
        extra='\n\n```python\n'+mdpy.split('\n\n',1)[1]+'```\n' if i==4 else ''
        text=f'## 原单元 {i}（{c["cell_type"]}）\n\n'+NOTES[i]+extra+'\n'
        n={'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
        explained['cells']+=[copy.deepcopy(n),copy.deepcopy(c)];new=copy.deepcopy(c)
        if c['cell_type']=='code':
            comment='导入展示类，不加载模型或计算GQA。' if i==0 else '展示原图：多个Q heads共享K/V，width只控制显示宽度。'
            new['source']=('# '+comment+'\n'+''.join(c['source'])).splitlines(keepends=True)
            py+=f'\n# %% 原代码单元 {i}\n'+''.join(new['source'])+'\n'
        annotated['cells']+=[copy.deepcopy(n),new];notes.append(text)
    for name,nb in [('原代码_带说明.ipynb',explained),('代码注释.ipynb',annotated)]:
        (code/name).write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    (code/'llama2_src_grouped_query_attention_代码注释.py').write_bytes((py.rstrip()+'\n').encode('utf-8'))
    (ROOT/'代码单元解释.md').write_bytes(('\n'.join(notes).rstrip()+'\n').encode('utf-8'))
    meta_text=meta.decode('utf-8');tree=ast.parse(meta_text)
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('repeat_kv','Attention')]
    reading='# -*- coding: utf-8 -*-\n# 固定Meta官方repeat_kv与完整Attention类，只新增中文注释。\n# 依赖原模块的torch/nn/F、FairScale、ModelArgs与RoPE helper，不能单文件直接运行。\n\n'
    reading+='\n\n'.join(annotate_official(ast.get_source_segment(meta_text,n)) for n in selected)+'\n'
    (code/'GQA核心源码_代码注释.py').write_bytes(reading.encode('utf-8'))
    print('Built all5-cell explanations, two11-cell notebooks, three reading Python files and two exact originals.')

if __name__=='__main__':main()
