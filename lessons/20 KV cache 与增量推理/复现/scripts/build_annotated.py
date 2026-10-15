# -*- coding: utf-8 -*-
"""25 original cells, 25 explanations, one introduction; code changes are comments only."""
import copy
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / '代码/tutorials'
NOTES = {
0: '【建议会】从 IPython.display 导入 Image 类。后续 Image(path,width=...) 构造展示对象；它读取图片字节，不计算 Attention。最后一个表达式在 Jupyter 中自动显示，普通 .py 中不一定显示。相对路径按当前工作目录解析，请从代码/tutorials启动内核。',
1: '【必须会】basics 是概念入口：先理解每步预测一个新 token，再理解旧 token 的 K/V 为什么可复用。下面的 GIF、价格截图和示意图是课程材料，不是本次性能实验。',
2: r'''【必须会】公式来自 scaled dot-product attention。每个 head 的 Q=[B,S,d]、K/V=[B,P+S,d]，QKᵀ=[B,S,P+S]，除以 √d 后沿 key 轴 softmax，再乘 V 得 [B,S,d]。B=batch，S=本次输入长度，P=已缓存长度，d=head_dim。这里还需要因果 mask，原公式没有写出 mask 项。

KV cache 保存每层旧 token 的 key/value。decoder-only 自回归推理通常使用它；T5 decoder 可缓存 self-attention KV，也可复用 encoder cross-attention KV。原文的 BERT 说法应理解为常规双向 encoder 用法不采用这种逐 token 自回归缓存，不能推出所有配置永远不支持缓存。未来 token 会改变双向旧位置的隐藏状态，不能直接套用因果复用条件。''',
3: '【必须会】CLM=causal language modeling，causal 表示位置 i 只能读取不晚于 i 的 token。推理使用最后位置的 logits 预测下一 token；训练可一次并行计算所有位置，但用三角 mask 防止看到未来答案。',
4: '【必须会】NTP=next token prediction。训练输入与目标错开一位：输入 SOS,Love,that 的目标是 Love,that,can。模型输出每位置词表 logits，交叉熵对齐下一位置。表格的 Output 是累计生成文本，而非一次 forward 产生整串。无 cache 每步重算完整前缀；有 cache 首次 prefill 处理提示词，再把上一步生成的 token 单独送入模型。EOS 是结束 token；实际 tokenizer 特殊标记与这张英文示意表不一定同名。',
5: '【建议会】展示自回归 GIF，width=700 仅控制显示宽度。图中的序列越来越长，强调每个下一 token 依赖此前上下文；缓存减少重复计算，不让所有未来 token 一次独立生成。文件在代码/imgs，原历史 GIF 输出保留。',
6: '【可以暂时不掌握】展示课程当时的 GPT-4 价格截图，width=600。这是历史资料，不代表当前价格，也不能用计费输入/输出差异证明某一服务的 KV cache 实现或测量本地加速比。',
7: '【必须会】Q/K/V 是 query、key、value。它们是不同投影得到的张量，不是三个 token ID 列表。当前查询用 K 计算相关程度，再根据权重聚合 V。',
8: '【必须会】展示 qkv.png：从左侧 x 出发，分别经过可训练 Wq/Wk/Wv 得 Q/K/V。三条分支共享输入位置，投影参数不同。之后 QKᵀ 得分、softmax 得权重、权重乘 V 得输出；不要把 V 本身叫做最终加权和。',
9: '【必须会】原文 x:token embedding 是简写。第一个 block 的输入来自 embedding，深层 Attention 实际收到上一层输出经 attention_norm 的隐藏状态 x=[B,S,D]。Wq/Wk/Wv 是训练好的参数；cache 是推理产生的中间状态，不是新的可训练参数。',
10: '【必须会】展示 decoding-qkv.png。单 token decode 的 S=1，每个 head 只新增一行 q/k/v；旧 k/v 从缓存读取。多 token prefill 或分块输入 S>1 时 Q 也是矩阵，不能把“Q 总是一条 vector”当一般规则。',
11: '【必须会】按每 head 单 token 情形，q=[1,d]，K/V=[P+1,d]，qKᵀ=[1,P+1]，softmax 后乘 V 得 [1,d]。K/V 包含当前 token 行，不能漏掉它。原文把 V 写作 weighted sum 不准确：V 是未聚合的 value 行，最终 Attention 输出才是加权和。旧 Q 不被未来 query 使用，因此常规实现只缓存 K/V。',
12: '【必须会】展示 kvmatrices.png，按 token 方向查看 K/V 增加的行。缓存长度从 P 变为 P+S。它不是存储整张历史 attention score 矩阵；score 每步根据新 Q 计算，后续长度变化时新 query 的权重也会变化。',
13: '【必须会】展示 kv-cache.gif：第一次建立提示词缓存；后续每次仅投影新 token 的 Q/K/V，读取历史 K/V 并写入新增切片。缓存仍占内存，新 Q 仍须关注整个可见历史，计算并非恒定时间。GIF 是原课程动画，非本机计时记录。',
14: '【建议会】进入 Llama 2 实现。先用 SentencePiece 查看 token ID，再读 Attention 摘录；摘录在 Markdown 中，不是可直接运行的完整类。完整固定官方模型用于独立教学验证，明确记录 CPU 适配。',
15: '【必须会】表格 heads 是 query head 数 Hq。D/Hq 均为128：4096/32、5120/40、8192/64。缓存大小使用 Hkv，不一定等于 Hq；70B 的 KV 数不能仅从本表推断。模型并行时本 rank 的 Hkv_local=Hkv/world_size。layers 决定缓存要保存多少份，不能只算一个 Attention。',
16: '【必须会】SentencePieceProcessor(model_file=...) 读取真实 tokenizer.model；sp_model 是拥有分词状态的实例，不包含 Llama 权重。./tokenizer.model 相对于工作目录。输入文本经 encode 产生 list[int]，不是 embedding。文件来自同一固定上游提交，按字节复制并核对 SHA256。',
17: '【必须会】prompt 是 str；bos_id() 得整数1，encode(prompt) 得7个 ID。列表内 * 展开 encode 结果，得到长度8：[1,306,4658,278,6593,310,2834,338]。普通列表没有 Tensor dtype；送模型前应构造成 int64[B,S]。BOS 是开始 token，不是缓存的特殊槽。',
18: '''【必须会】这是 Attention(nn.Module) 的不完整摘录，下面保留每条语句并增加注释。class 表示类，nn.Module 是父类，self 指实例；__init__ 是构造方法，forward 是调用模型时的前向方法。完整源码先调用 super().__init__() 注册模块，并定义 head 数等属性；摘录缺这些步骤，还缺 bsz,seqlen=x.shape[:2]、辅助函数、最终注意力运算和返回，因此不能直接实例化运行。

args 是 ModelArgs，dim=D；start_pos=P 是本次首 token 的绝对位置；x 是每层归一化后隐藏状态浮点[B,S,D]。freqs_cis 为本次 P:P+S 的 complex64[S,d/2]。mask 允许 None；完整 Attention 接受可广播到[B,Hq_local,S,P+S]的浮点因果 mask。

ColumnParallelLinear 分片输出通道，RowParallelLinear 分片输入通道；本次验证 world_size=1，不能据此证明多 rank 通信。Wq 输出Hq_local*d通道，Wk/Wv 输出Hkv_local*d通道。view 拆 head；apply_rotary_emb 仅旋转 Q/K。cache_k 保存旋转后的 K，cache_v 保存未旋转 V。

torch.zeros 预分配固定容量[Bmax,Tmax,Hkv_local,d]；.cuda() 强制上 GPU；.to(xq) 匹配 xq 的设备及 dtype，不是把 Q 存入缓存。cache 是普通 Tensor 属性，不是 nn.Parameter 或 register_buffer，state_dict 不保存它，模型.to(...)也不自动迁移这些普通属性，所以 forward 显式处理。

切片赋值只覆写[0:B,P:P+S]，容量不增长。读取[:B,:P+S]将旧缓存与新增区域一起暴露，未来预分配槽不参与计算。切片通常为共享存储 view，并不是复制整个缓存。原注释“append”是逻辑追加，不能理解为 torch.cat 重新分配。

完整源码还执行 repeat_kv（GQA 展开）、transpose、QKᵀ/√d、mask、softmax、乘 V、合并 heads、wo，返回[B,S,D]。这里只摘录到 keys/values；逐层 cache 更新属于有状态推理，需要正确连续位置、相同前缀和 batch 顺序。新会话从0重新写有效前缀，读范围必须不包含旧尾部。

固定旧版 Transformer.forward 的 mask 在 P>0 且 S>1 时 shape 错误。本讲验证分别记录原版报错与一条 mask 赋值的教学修正，不将修正写进原 notebook。详见解释及复现报告。''',
19: '【建议会】转入 Transformers 示例：公开 notebook 加载 gpt2，用 generate(use_cache=True/False) 比较计时。GPT-2 使用自己的绝对位置与缓存布局；前面的 Llama RoPE 缓存不可互换。',
20: '【建议会】import os 导入进程接口；os.environ 是环境变量映射，两条赋值把本进程 HTTP/HTTPS 代理设为 localhost:7890。它不是 KV cache 必需项，端口需已有代理服务。本次不执行此格，避免覆盖当前进程网络设置；原代码完整保留。',
21: '【必须会】导入 NumPy、time、torch 和两个 Auto 类。条件表达式依据 torch.cuda.is_available() 选择设备。from_pretrained("gpt2") 分别加载 tokenizer 与预训练权重，可能访问网络；.to(device) 把模型参数放到设备。真实 tokenization 结果为 BatchEncoding，内含 int64 input_ids/attention_mask[B,S]。本次在明确教学变体中只替换两处加载字符串，使用真实 GPT2Tokenizer/Fast 和随机两层H16小GPT2本地 checkpoint，绝不冒称原 GPT-2 权重。',
22: '【建议会】最后表达式 tokenizer 展示对象 repr，查看 vocab_size、特殊 token、fast 状态等。历史输出是 GPT2TokenizerFast、词表50257、最大长度1024；本次本地词表相同，但路径/配置/模型上下文长度另行记录，repr 不代表已经生成或训练。',
23: '''【必须会】外层遍历 (True,False)，每种模式清空 times；内层 range(10) 做10次。_ 表示不使用循环索引。time.time() 获取秒数，两次相减包含分词、输入迁移与 generate 调用；times.append 存浮点秒数。** 将 BatchEncoding 的字段作为关键字参数展开。use_cache 控制重复使用 past_key_values；max_new_tokens=1000 是新增长度上限，不是输入+输出总长。

generate 返回 int64[B,prompt_length+实际新增长度]，原代码没保存返回值；EOS可能提前结束。np.mean 求均值，np.std 默认总体标准差ddof=0，round(...,3)只是格式化；f-string中的条件表达式选择with/without。+- 是标准差的展示，不是置信区间。

历史计时不是本次测量，原环境与真实生成长度未完整保存。缺 warmup、固定输出长度、CUDA synchronize、顺序随机化；不能仅凭打印断言可迁移的加速倍数。本次只将内层10→2、新增1000→16做可控CPU教学运行，保留原23源码和历史输出。两版比较生成ID及投影输入长度；墙钟数值不要求一致，不作为性能结论。''',
24: '【可以暂时不掌握】原空代码单元，无语句、无模型运算。保留 cell metadata、execution_count 和 outputs；阅读版有单元注释但没有新执行动作。'
}

INLINE = {
0: {0:'Image 是 notebook 展示对象，后面用它读取配套图。'},
5: {0:'仅展示原自回归动画；width 是显示像素宽度。'},
6: {0:'这是历史价格截图，不是当前报价或本次性能实验。'},
8: {0:'图中 x 经三组可训练投影分别得到 Q、K、V。'},
10: {0:'S=1 的 decode 每 head 新增一行 Q/K/V，旧 KV 可以复用。'},
12: {0:'KV 的 token 轴随有效长度增加，缓存不保存整张 attention score。'},
13: {0:'首次 prefill 建缓存，后续 decode 只新增当前 token 的 K/V。'},
16: {0:'导入真实 SentencePiece 分词器类。',1:'相对路径要求工作目录为代码/tutorials；加载的是词表模型。'},
17: {0:'输入是 Python 字符串，尚未成为 token Tensor。',1:'* 展开7个普通整数ID，前置BOS得到长度8的 list[int]。'},
20: {0:'os 提供环境变量映射；本次验证跳过整格。',1:'设置本进程HTTP代理；此端口必须已有服务。',2:'HTTPS代理同理；这不是缓存算法的一部分。'},
21: {0:'np用于统计秒数；不是模型后端。',1:'time用于墙钟计时，CUDA时还要另考虑异步执行。',2:'torch提供张量和设备检测。',3:'Auto类按配置选择具体Tokenizer/因果LM。',5:'条件表达式选择当前torch可用的设备。',6:'原版要求GPT-2 tokenizer；验证只将此字符串换为本地路径。',7:'原版加载预训练GPT-2；验证明确使用随机小GPT2并迁到CPU。'},
22: {0:'在Jupyter显示tokenizer repr；普通脚本不会自动展示。'},
23: {0:'依次测试开缓存、关缓存；开关影响generate的past_key_values复用。',1:'每种模式重新创建浮点秒数列表。',2:'10次重复；教学验证仅将这个常量改成2。',3:'计时从分词与输入迁移之前开始。',4:'** 将input_ids等字段解包成generate关键字参数。',5:'True复用旧KV；False每步重新处理完整前缀。',6:'最多新增1000个token，教学验证仅将此常量改成16。',7:'记录一次调用总墙钟时间，原代码未保存生成结果。',8:'均值与总体标准差不是置信区间。',9:'ddof默认0；保留三位小数仅影响打印。',10:'f-string中条件表达式选择with/without标签。'}
}

def excerpt_comment(line):
    s = line.strip()
    if s.startswith('class '): return 'nn.Module 子类的摘录；缺 super 初始化及部分属性，不能单独运行。'
    if s.startswith('def __init__'): return '构造方法：args提供D、head数和最大batch/长度等配置。'
    if 'self.wq =' in s: return 'Q投影：实际本rank输出[B,S,Hq_local*d]。'
    if 'self.wk =' in s: return 'K投影：输出[B,S,Hkv_local*d]。'
    if 'self.wv =' in s: return 'V投影：输出[B,S,Hkv_local*d]。'
    if 'self.wo =' in s: return '合并heads后投影回D通道；完整类在末尾调用它。'
    if 'self.cache_' in s and 'torch.zeros' in s: return '预分配固定容量的普通Tensor属性；不是可训练权重。'
    if 'args.max_batch_size' in s: return '轴0为最大batch容量，实际只使用前B行。'
    if 'args.max_seq_len' in s: return '轴1为最大token容量；切片写入，不动态扩容。'
    if 'self.n_local_kv_heads,' in s: return '轴2为当前rank的KV heads，不一定等于Q heads。'
    if 'self.head_dim,' in s: return '轴3为每head通道d=D/Hq。'
    if ').cuda()' in s: return '原版强制CUDA；CPU教学模块仅显式修改这两处设备调用。'
    if s.startswith('def forward'): return '前向方法在本实例上写cache；摘录缺输入shape解包和最终return。'
    if 'x: torch.Tensor' in s: return '输入浮点隐藏状态[B,S,D]，不是token整数ID。'
    if 'start_pos: int' in s: return 'P为本次首token绝对位置，也是写缓存切片的起点。'
    if 'freqs_cis:' in s: return '本次位置P:P+S的复数频率[S,d/2]，由上层切片。'
    if 'mask: Optional' in s: return 'None或可广播的因果mask，完整得分shape[B,Hq_local,S,P+S]。'
    if 'xq, xk, xv =' in s: return '三次投影只计算本次S个token，旧KV不重新投影。'
    if 'xq = xq.view' in s: return '把投影通道拆为Q heads，得到[B,S,Hq_local,d]。'
    if 'xk = xk.view' in s or 'xv = xv.view' in s: return 'K/V拆KV heads，得到[B,S,Hkv_local,d]；摘录没有定义bsz/seqlen。'
    if 'apply_rotary_emb' in s: return 'Q/K按绝对位置旋转，V不旋转；缓存K是此后的结果。'
    if '.to(xq)' in s: return '匹配Q的设备和dtype，不是保存Q；转换可能重分配。'
    if 'start_pos : start_pos + seqlen] =' in s: return '原地覆写当前[B,P:P+S]区域；此前KV保持不变。'
    if s.startswith(('keys =','values =')): return '读取已用前缀P+S，包括当前新KV，排除未写入的容量尾部。'
    return None

def annotate(source, comments=None, excerpt=False):
    result=[]
    for i,line in enumerate(source.splitlines(keepends=True)):
        note=excerpt_comment(line) if excerpt else (comments or {}).get(i)
        if note:
            indent=line[:len(line)-len(line.lstrip())]
            result.append(indent+'# '+note+'\n')
        result.append(line)
    return ''.join(result)

def reading_text(source):
    # Reading copies can normalize trailing spaces; original notebook sources stay exact.
    return '\n'.join(line.rstrip() for line in source.splitlines()).rstrip()+'\n'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--meta-source',required=True)
    args=parser.parse_args()
    data=Path(args.meta_source).read_bytes()
    assert hashlib.sha256(data).hexdigest()=='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    source=data.decode('utf-8')
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Attention')
    full=ast.get_source_segment(source,cls)+'\n'
    full_comments=[]
    for line in full.splitlines(keepends=True):
        s=line.strip(); note=excerpt_comment(line)
        # Do not insert comment text inside the official multiline docstrings.
        if s.startswith('super().__init__'):note='调用nn.Module父类构造，注册子模块与参数机制。'
        elif s.startswith('self.n_kv_heads ='):note='未指定KV heads时默认与Q heads相同；否则使用GQA配置。'
        elif s.startswith('model_parallel_size ='):note='从FairScale取得模型并行组大小，不是数据并行batch数。'
        elif s.startswith(('self.n_local_heads =','self.n_local_kv_heads =')):note='全局head数按模型并行组切分；本次真实运行world_size=1。'
        elif s.startswith('self.n_rep ='):note='每个KV head服务多少Q heads；完整forward据此repeat_kv。'
        elif s.startswith('self.head_dim ='):note='每head通道d=D/Hq；配置需要可整除。'
        elif s.startswith('bsz, seqlen, _ ='):note='从输入[B,S,D]解包batch与本次token长度，补齐摘录缺失的变量。'
        elif s.startswith('keys = repeat_kv') or s.startswith('values = repeat_kv'):note='将Hkv_local按组复制到Hq_local以对齐Q，缓存自身仍保存较少KV heads。'
        elif '.transpose(1, 2)' in s:note='交换token与head轴，使后续每个head独立矩阵乘法。'
        elif s.startswith('scores = torch.matmul'):note='[B,Hq,S,d]乘[B,Hq,d,P+S]，除√d得attention scores。'
        elif s.startswith('if mask is not None'):note='多token需防止新chunk内部读取未来位置；单token decode无未来新行。'
        elif s.startswith('scores = scores +'):note='mask以0放行、负无穷阻断，需要可广播到[B,Hq,S,P+S]。'
        elif s.startswith('scores = F.softmax'):note='先用FP32沿key轴归一化，再恢复Q dtype。'
        elif s.startswith('output = torch.matmul'):note='注意力权重乘V得[B,Hq,S,d]；V本身不是加权和。'
        elif s.startswith('output = output.transpose'):note='换回[B,S,Hq,d]，contiguous整理内存后合并heads。'
        elif s.startswith('return self.wo'):note='输出投影回D，返回[B,S,D]给残差分支。'
        full_comments.append((line,note))
    in_doc=False; result=[]
    for line,note in full_comments:
        if '"""' in line:
            if line.count('"""')%2:in_doc=not in_doc
            result.append(line);continue
        if note and not in_doc:result.append(line[:len(line)-len(line.lstrip())]+'# '+note+'\n')
        result.append(line)
    (CODE/'Attention官方源码_代码注释.py').write_bytes(('# -*- coding: utf-8 -*-\n# 固定Meta提交7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e的完整Attention类。\n# 只增加中文注释；依赖完整model.py的导入、ModelArgs、RoPE/repeat_kv等，不能单独运行。\n'+''.join(result)).encode('utf-8'))
    original=json.loads((CODE/'llama2_src_cache_kv.ipynb').read_bytes())
    assert len(original['cells'])==25 and set(NOTES)==set(range(25))
    explained=copy.deepcopy(original); annotated=copy.deepcopy(original)
    intro={'cell_type':'markdown','metadata':{},'source':['# 第20讲：KV cache 与增量推理\n\n原单元从0编号；保留原历史输出。新增说明、注释与CPU教学验证由本项目编写，原历史输出不代表本次运行。详见本讲解释.md与复现/README.md。\n']}
    explained['cells']=[copy.deepcopy(intro)];annotated['cells']=[copy.deepcopy(intro)]
    md=['# 第20讲代码与全部单元说明\n\n原单元从0编号，共25格：14代码、11 Markdown；Markdown18的重要源码摘录也有详细注释。\n']
    py=['# -*- coding: utf-8 -*-\n# 第20讲原代码的中文注释阅读版；原加载及benchmark参数保持不变。\n# 本文件含代理设置与远程模型加载，完整原版未作为CPU教学验证运行。\n']
    excerpt=''
    for i,c in enumerate(original['cells']):
        src=''.join(c['source'])
        extra=''
        if i==18:
            excerpt=src.split('```')[1].strip('\n')+'\n'
            extra='\n\n### 摘录的逐语句注释\n\n```python\n'+reading_text(annotate(excerpt,excerpt=True))+'```\n'
        text=f'## 原单元 {i}（{c["cell_type"]}）\n\n'+NOTES[i]+extra+'\n'
        note={'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
        explained['cells'].extend([copy.deepcopy(note),copy.deepcopy(c)])
        new=copy.deepcopy(c)
        if c['cell_type']=='code':
            new['source']=annotate(src,INLINE.get(i)).splitlines(keepends=True)
            py.extend([f'\n# %% 原代码单元 {i}\n',annotate(src,INLINE.get(i))+'\n'])
        annotated['cells'].extend([copy.deepcopy(note),new])
        md.append(text)
    for name,nb in [('原代码_带说明.ipynb',explained),('代码注释.ipynb',annotated)]:
        (CODE/name).write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    (CODE/'llama2_src_cache_kv_代码注释.py').write_bytes(reading_text(''.join(py)).encode('utf-8'))
    (ROOT/'代码单元解释.md').write_bytes('\n'.join(md).encode('utf-8'))
    (CODE/'Attention摘录_代码注释.py').write_bytes(reading_text('# -*- coding: utf-8 -*-\n# 原Markdown18的不完整摘录，供阅读，不可独立实例化运行。\n'+annotate(excerpt,excerpt=True)).encode('utf-8'))
    print('Built two 51-cell notebooks, whole-cell notes, reading Python and annotated Markdown18 excerpt.')

if __name__=='__main__': main()
