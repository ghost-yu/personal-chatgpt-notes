"""固定原件，生成只增加注释的阅读版本；不访问网络或下载权重。"""
import ast
import copy
import hashlib
import inspect
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import tokenize
import io

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path('G:/zzz/Llama 入门教程/代码/personal_chatgpt')
COMMIT = '7fbc29824171fd07dc2c30fe922e61ebc018ad1d'
REL = 'tutorials/llama/llama2_7b_llama3_8b.ipynb'
FOLDER = ROOT/'代码/tutorials/llama'
CORE_NAMES = ['rotate_half', 'apply_rotary_pos_emb', 'repeat_kv', 'LlamaRMSNorm',
              'LlamaMLP', 'LlamaAttention', 'LlamaSdpaAttention']
NOTES = [
 '导入Transformers、PyTorch、自动因果语言模型工厂与Image。os.environ两行会修改当前进程代理，127.0.0.1:7890只适用于讲师环境；不是加载模型的必需设置。本次在独立验证进程执行并恢复原代理。',
 '区分Meta原生权重与Hugging Face转换格式；同一个模型的不同键名/分片格式不意味着模型结构不同。',
 'Markdown三条加载语句：torch.load返回checkpoint字典，Transformer(model_args)创建架构，load_state_dict复制权重并返回missing/unexpected keys；strict=False不等于忽略同名尺寸不匹配。索引JSON把张量名映射到文件，不是device_map。原文存在乱码，原件保留；新增解释按源码语义校正。',
 '比较对象是早期Llama2-7b-hf和Meta-Llama-3-8B基础模型，不是Chat/Instruct，也不是后来的3.1/3.2。',
 'llama2_id和llama3_id都是str类型Hub标识符，不是已经加载的模型；后续from_pretrained才解析配置与checkpoint。',
 'AutoModelForCausalLM.from_pretrained类工厂按config选择LlamaForCausalLM；torch_dtype要求BF16权重（每元素2 bytes），device_map=auto让Accelerate分配设备。14GB只是7×10^9×2的粗估，不含KV/激活/临时峰值。本次不执行原远程加载；meta构造和本地随机CPU权重加载分别记录为明确变体。',
 '原格只有注释，不会打印任何参数。named_parameters遍历(name,Parameter)，para.device是所在设备；不要把原注释误称实际多卡输出。',
 '第二次加载Llama3；8B的16GB也只是近似权重体积。两次赋值让两模型同时驻留，不能把16GB当同时装入两者的内存。真实小模型只在原4格改成本地模型路径，5/7格保留BF16与device_map=auto参数；补充训练才转FP32。',
 '自动设备映射标题；它与数据并行/张量并行/训练optimizer分片是不同机制。',
 'get_balanced_memory摘录计算各设备可用预算，infer_auto_device_map另负责分配；不是这个if块直接搬运权重。low_zero只在balanced_low_0为True，auto并不承诺每卡严格一半。上下文变量未定义，摘录不是独立可运行脚本。',
 '可训练参数统计标题：requires_grad为True表示autograd资格，不表示此刻已经训练或optimizer已包含该参数。',
 '定义普通Python函数，不是模型方法。model是nn.Module；named_parameters默认去重，numel返回int元素数，requires_grad是bool。两个int累计器原地更新，函数只print、隐式返回None，不会改变模型权重或开关；空模型分母为0，另记录反例。',
 '对Llama2实际模型对象调用统计函数。历史6738415616是精确元素数；本次完整meta架构能复核这个数，但meta没有权重数值，不能用于真实forward。',
 '同一函数用于Llama3；历史8030261248与8B近似命名区分。100%只表示所有参数requires_grad=True，不证明训练发生。',
 '末尾表达式在Jupyter触发模型repr；展示Embedding、32个DecoderLayer、norm和lm_head。文本repr不会执行forward。普通.py直接执行此表达式不会自动显示，需要print。',
 'Llama3结构：Embedding[128256,4096]、Q/O4096、K/V1024、MLP14336；K/V总投影宽不同，但每个head仍是128，所以QK点积合法。',
 '属性链定位输入embedding的weight，numel按V×D计数；f-string的:,是千位分隔，返回两个str组成的tuple，不是Tensor。131072000与525336576只算输入表，未算独立lm_head。',
 'Python / 返回float；11008/4096=2.6875，14336/4096=3.5。比较的是MLP中间维与D的比值，不是模型大小或head数。',
 '完整attention公式：Q/K每head宽必须同为dh，V宽可不同（本实现也为dh）；softmax沿key位置j归一化。原式省略batch/head/causal mask；GQA先按g(h)=floor(h/R)选择共享K/V，再逐head做点积。',
 'Image只构造远程HTML图对象，本次未验证外站图片内容。quality/efficiency图不能直接证明本机速度或所有MHA都质量最高；本讲用分组张量与实际梯度说明机制。',
 '4096/32=128.0是每query head的宽度；GQA的KV head减少不改变它。公式中的dk对应单head维，不是K投影拼接总宽1024。',
 'Markdown完整摘录需要逐句读：nn.Linear(in,out)实际weight[out,in]；Q为H×dh，K/V为G×dh；H//G为R，32/8=4个query heads共享一份KV。MLP gate/up先升维，SiLU(gate)*up是逐元素乘法，再down投影；Meta w1/w3/w2分别对应gate/up/down。self表示实例，config字段来自LlamaConfig，这里不是独立类定义。',
 '参数表heads列是query head数，不是KV head数；Llama2家族不能一概称MHA，7B/13B与70B不同。两代本例head_dim都是128。Llama3 K/V单个矩阵参数剩1/4即减少75%，但整个attention剩5/8，不能说整个模型减少75%。',
 '原最后一格为空，保留空格和历史元数据，不编造其执行结果。',
]

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')

def line_note(line):
    s = line.strip()
    rules = [
        ('os.environ', '修改进程环境变量；讲师本地代理地址不通用。'),
        ('from_pretrained', '真正读取配置与checkpoint；原语句要求远程模型访问和足够内存。'),
        ('named_parameters', '遍历去重的参数对象，名字与Parameter配成一对。'),
        ('requires_grad', '只检查是否允许求梯度，不会执行训练。'),
        ('numel()', '统计元素个数，不是字节数；字节还需乘dtype大小。'),
        ('all_params +=', '累加当前参数的元素总数。'),
        ('trainable_params +=', '累加允许求梯度的元素个数。'),
        ('print(f', '格式化打印统计结果；函数仍隐式返回None。'),
        ('print_trainable_parameters(', '定义或调用参数统计函数，输入是模型对象。'),
        ('llama2_id', 'Llama2 Hub标识字符串，不是模型对象。'),
        ('llama3_id', 'Llama3 Hub标识字符串，不是模型对象。'),
        ('import ', '导入实际使用的模块或对象；不在此行下载模型。'),
        ('Image(', '构造远程图展示对象；没有验证图片内容。'),
        ('11008/4096', '中间层宽度除以模型宽度，得到两个float比值。'),
        ('4096/32', '模型宽度除以query头数，得到单头128维。'),
        ('trainable_params =', '初始化可训练元素数的int累计器。'),
        ('all_params =', '初始化全部元素数的int累计器。'),
        ('self.q_proj', 'Q投影：D→H×dh；Linear权重存储为[out,in]。'),
        ('self.k_proj', 'K投影：D→G×dh；GQA只保留G个KV头。'),
        ('self.v_proj', 'V投影：D→G×dh；与K共用头数，不共享参数矩阵。'),
        ('self.o_proj', '拼接H个query头输出后映射回D。'),
        ('num_key_value_groups =', 'R=H//G，每份KV服务R个连续query头。'),
        ('self.gate_proj', 'gate分支D→F；前向中施加SiLU激活。'),
        ('self.up_proj', 'up分支D→F，随后与激活后的gate逐元素相乘。'),
        ('self.down_proj', 'down分支F→D，恢复残差流宽度。'),
        ('down_proj = self.down_proj', 'SiLU(gate(x))*up(x)逐元素门控后投回D。'),
        ('self.w2(', 'Meta w1/w3/w2对应HF gate/up/down。'),
        ('get_balanced_memory(', '计算设备内存预算，不直接生成最终device_map。'),
        ('if device_map', 'sequential分配策略不走此处平衡预算。'),
        ('low_zero=', '只有balanced_low_0策略才特意给GPU0少放权重。'),
        ('torch.load', '读取checkpoint字典；map_location控制加载设备。'),
        ('Transformer(model_args)', '按配置创建Meta Transformer对象。'),
        ('load_state_dict', '将checkpoint复制进同名参数；strict=False仍不能忽略同名shape冲突。'),
    ]
    for pattern, note in rules:
        if pattern in s:
            return note
    if s in ('llama2', 'llama3'):
        return 'Jupyter展示对象结构；不是前向或文本生成。'
    return None

def annotated(source, intro=None):
    # 不向字符串内部插注释，保留docstring的值与全部非注释token。
    protected=set()
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type == tokenize.STRING and t.start[0] != t.end[0]:
            protected.update(range(t.start[0], t.end[0]+1))
    out = ['# '+intro] if intro else []
    for i,line in enumerate(source.splitlines(),1):
        note=line_note(line) if i not in protected and not line.lstrip().startswith('#') else None
        if note:
            out.append(line[:len(line)-len(line.lstrip())]+'# '+note)
        out.append(line.rstrip() if line.lstrip().startswith('#') else line)
    return '\n'.join(out)+'\n' if out else ''

def annotate_markdown(source):
    # 只处理围栏内的Python摘录，公式/链接/原乱码全部保留。
    return re.sub(r'```[^\n]*\n(.*?)```',lambda m:'```\n'+annotated(m.group(1))+'```',source,flags=re.S)

def core_note(line):
    s=line.strip()
    for pattern,note in [
        ('query_states.view','[B,T,H*dh]拆头并换轴为[B,H,T,dh]。'),
        ('key_states.view','[B,T,G*dh]拆为[B,G,T,dh]；cache保留G头。'),
        ('value_states.view','V同样拆成G个KV头，V不施加RoPE。'),
        ('repeat_kv(', '沿head轴把每份KV连续重复R次，匹配H个query头。'),
        ('apply_rotary_pos_emb(', '对Q/K应用同位置RoPE；不旋转V。'),
        ('past_key_value.update','真实Cache按layer_idx追加KV，发生状态修改。'),
        ('torch.matmul(query_states','逐query头计算QK转置，除sqrt(dh)，得[B,H,T,S]。'),
        ('attn_weights +','加性mask：有效处0、不可见处负无穷或极小值。'),
        ('softmax(attn_weights','沿key位置归一化，内部升FP32再转回query dtype。'),
        ('dropout(attn_weights','训练时按概率丢弃注意力权重；eval时关闭。'),
        ('torch.matmul(attn_weights','[B,H,T,S]乘V[B,H,S,dh]得到每头输出。'),
        ('scaled_dot_product_attention(', '调用PyTorch真实SDPA内核，默认含1/sqrt(dh)缩放。'),
        ('is_causal =','无显式mask且T>1时才让SDPA自行施加因果约束。'),
        ('attn_output.transpose','从[B,H,T,dh]换回[B,T,H,dh]，contiguous使后续view合法。'),
        ('attn_output.view','拼接head，恢复[B,T,D]。'),
        ('attn_output.reshape','拼接head，恢复[B,T,D]。'),
        ('self.o_proj(attn_output)','输出投影；不会减少query头数或序列长度。'),
        ('if output_attentions:', 'SDPA不直接返回权重，要求输出时回退eager分支。'),
        ('pretraining_tp > 1','旧张量切片计算分支，不等同device_map=auto。'),
        ('torch.cat','沿指定维连接分片；本验证pretraining_tp=1不走此分支。'),
        ('variance =','RMSNorm按最后一维算平方均值，不减去均值。'),
        ('torch.rsqrt','乘1/sqrt(均方+eps)，维持[B,T,D]。'),
        ('hidden_states.to(torch.float32)','RMSNorm内部FP32统计，随后恢复输入dtype。'),
        ('cos.unsqueeze','为head轴插入可广播维，cos/sin原为[B,T,dh]。'),
        ('q_embed =','cos*q加sin*半区90度旋转，shape与Q相同。'),
        ('k_embed =','对K做相同位置旋转，KV头数可与Q不同。'),
        ('hidden_states[:, :, None','插入重复轴，用expand共享原KV数据。'),
        ('reshape(batch','把KV头和重复轴合并；不据此声称整个算子零拷贝。'),
    ]:
        if pattern in s:return note
    return line_note(line)

def build():
    from transformers.models.llama import modeling_llama as hf
    FOLDER.mkdir(parents=True,exist_ok=True)
    raw=subprocess.check_output(['git','-C',str(UPSTREAM),'show',COMMIT+':'+REL])
    (FOLDER/'llama2_7b_llama3_8b.ipynb').write_bytes(raw)
    nb=json.loads(raw);assert len(nb['cells'])==len(NOTES)==24
    docs=['# 第29讲：全部原单元说明\n\n单元从0编号，原乱码与历史输出保留。\n']
    reading=['# 第29讲 Python 阅读版：原代码详注，原Markdown作为注释保留。',
             '# 原5/7格会加载远程7B/8B权重；日常直接阅读。验证请使用复现/scripts/verify_delivery.py。']
    for mode,name in [('annotated','代码注释.ipynb'),('explained','原代码_带说明.ipynb')]:
        derived=copy.deepcopy(nb);cells=[{'cell_type':'markdown','id':'lecture29-header','metadata':{},'source':['# 第29讲：Llama2 7B vs Llama3 8B\n原单元、历史输出保留；新增注释不是新运行输出。\n']}]
        for i,c in enumerate(nb['cells']):
            cells.append({'cell_type':'markdown','id':f'lecture29-note-{i}','metadata':{},'source':[f'### 原单元 {i} · {c["cell_type"]}\n\n{NOTES[i]}\n']})
            d=copy.deepcopy(c);src=''.join(c['source'])
            if mode=='annotated':
                d['source']=(annotated(src,NOTES[i]) if c['cell_type']=='code' else annotate_markdown(src)).splitlines(keepends=True)
            cells.append(d)
        derived['cells']=cells;write(FOLDER/name,json.dumps(derived,ensure_ascii=False,indent=1)+'\n')
    for i,c in enumerate(nb['cells']):
        src=''.join(c['source']);docs.append(f'\n## 原单元 {i} · {c["cell_type"]}\n\n{NOTES[i]}\n\n~~~'+('python' if c['cell_type']=='code' else 'markdown')+'\n'+src+'\n~~~\n')
        reading.extend(['',f'# %% 原单元 {i} · {c["cell_type"]}','# '+NOTES[i]])
        if c['cell_type']=='code':reading.append(annotated(src))
        else:reading.extend(('# '+line).rstrip() for line in annotate_markdown(src).splitlines())
    # 阅读文档只移除行首为#的标题/原注释行末空格；原notebook字节不变。
    doc_text='\n'.join(line.rstrip() if line.lstrip().startswith('#') else line for line in ''.join(docs).splitlines())+'\n'
    write(ROOT/'代码单元解释.md',doc_text);write(FOLDER/'llama2_7b_llama3_8b_代码注释.py','\n'.join(reading).rstrip()+'\n')
    core=['# Transformers 4.44.2 官方实现摘录，只增中文注释。',
          '# 版权所有 HuggingFace Inc.；Apache-2.0：https://github.com/huggingface/transformers/blob/v4.44.2/LICENSE',
          '# 导入原模块依赖，再定义完整原函数/类；该文件可导入，但不是完整LlamaForCausalLM。',
          'from transformers.models.llama import modeling_llama as _official',
          'globals().update({name: value for name, value in vars(_official).items() if not name.startswith("__")})','']
    hashes={}
    for name in CORE_NAMES:
        source=inspect.getsource(getattr(hf,name));hashes[name]=hashlib.sha256(source.encode()).hexdigest()
        protected=set()
        for t in tokenize.generate_tokens(io.StringIO(source).readline):
            if t.type==tokenize.STRING and t.start[0]!=t.end[0]:protected.update(range(t.start[0],t.end[0]+1))
        core.append('# ===== '+name+'：完整对象 =====')
        for n,line in enumerate(source.splitlines(),1):
            note=core_note(line) if n not in protected and not line.lstrip().startswith('#') else None
            if note:core.append(line[:len(line)-len(line.lstrip())]+'# '+note)
            core.append(line.rstrip())
        core.append('')
    write(FOLDER/'Llama核心实现_代码注释.py','\n'.join(core).rstrip()+'\n')
    dist=metadata.distribution('transformers')
    license_file=next(p for p in dist.files if str(p).endswith('.dist-info/LICENSE'))
    (FOLDER/'LICENSE.transformers').write_bytes(Path(dist.locate_file(license_file)).read_bytes())
    manifest={'upstream_commit':COMMIT,'upstream_path':REL,'original_sha256':hashlib.sha256(raw).hexdigest(),
              'transformers_version':'4.44.2','official_object_source_sha256':hashes,
              'subtitle':'E:/ff65f590-f38b-43dc-9c18-aa1220456074.txt，仅保留本地；新增解释校正ASR错误',
              'full_config':'按原notebook历史结构重建，不冒称已下载官方config.json；HF请求TLS EOF失败',
              'remote_image':'原cell19 URL保留，未验证远程图内容',
              'papers':['https://arxiv.org/abs/2307.09288v2','https://arxiv.org/abs/2305.13245v3'],
              'meta_llama3_reference_commit':'cf3ea595be1edfb019d21b14d8b215a39161c860'}
    write(ROOT/'来源清单.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Built 24 cells,',sum(c['cell_type']=='code' for c in nb['cells']),'code cells; 49-cell derivatives and 7 complete HF objects')

if __name__=='__main__':build()
