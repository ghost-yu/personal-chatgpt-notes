"""从固定notebook生成只加中文注释的派生版，包含Markdown中的完整CoPE。"""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'代码/tutorials/position_encoding'
NOTES=[
 '导入Image、torch、math、NumPy；不下载模型、不建立网络。Image(url=...)构造展示对象，不等于浏览器已获取图片。',
 '问题标题：为什么Transformer需要位置信息？要区分不带mask的attention置换等变与自回归mask本身带来的顺序约束。',
 'X[L,D]分别乘Wq/Wk/Wv得Q/K[L,Dk]和V[L,Dv]；只有Q/K特征维必须相同。QK转置得[L,L]，softmax沿key轴归一化，再乘V。无位置无mask时同步置换Q/K/V会同步置换输出，不是输出完全不变；RoPE角度独立于内容，但Q/K及最终注意力依然依赖内容。',
 'BERT/GPT示例标题；不能据此概括所有名为GPT的后续模型。',
 '远程示意图URL保留，实际只构造HTML图片对象。没有验证该远程图片内容；本讲用公式与实际小模型embedding解释，不把外链可构造当图已下载。',
 '两个Markdown源码片段：BERT加token/type/position embedding；GPT风格把token与position相加，可选token_type项。它们是摘录而非完整模型。代码注释版在块内也增加中文注释；第27讲已实际演示GPT2左padding的位置编号问题。',
 '固定正弦位置编码标题。绝对位置与是否可学习是两个不同分类维度。',
 '偶数通道sin、奇数通道cos，位置t与通道i不能混淆；相邻两通道共享一个频率。d_model=4时频率1与0.01，pos0是[0,1,0,1]。没有可训练位置参数，但仍能通过相位关系表达相对距离。',
 'LLaMA RoPE标题；这里频率dim应为head_dim，而非把整个hidden_size当一个head。',
 'RoPE将Q/K旋转，以Rm转置Rn=R(n-m)得到带符号相对位移；不是只用绝对值距离。freqs_cis实际是复数相位表，不是显式D×D矩阵。Markdown中的wq/wk/wv/apply/复数变量来自不同层次摘录，不能单独当完整可运行Attention。',
 '展示原RoPE论文图，原图已按固定Git字节保留。theta是角频率，m*theta是旋转角，两者要区分。',
 '定义完整频率预计算函数。arange隔2取通道，end个位置与D/2频率outer成[end,D/2]角度，polar把单位幅值和角度转为complex64。函数本身没检查dim为偶数，也不返回旋转后的Q/K。',
 '调用D=4/end=3，得到complex64[3,2]；两条频率1和0.01。只生成相位表，不涉及模型权重或训练。',
 '对上一格每步手算。符号j为虚数单位，此处不是key下标；e^(j angle)=cos(angle)+j sin(angle)。',
 'polar([1],[2])把幅值1和弧度2组成单位复数，约-0.4161+0.9093j；不是e的实指数。',
 '手写[3,2]角度矩阵、ones_like幅值和polar，验证与预计算相位表相同。ones_like继承shape/dtype，不是创建训练Parameter。',
 '苏剑林RoPE推导链接，属于作者解释材料；本讲保留来源，不把外链当命令。',
 'R12/R34若是完整D×D嵌入块可相乘；若指两个2×2小块，应写block_diag而不是小矩阵连乘。本例20格使用完整4×4矩阵，所以相乘合法且作用在不相交维度。',
 '展示四维空间六种坐标平面的旋转；后面20格只选择互不相交的(1,2)与(3,4)两个平面，不是把图中六个旋转全部相乘。',
 '展示相邻维配对图，向量1/2、3/4分别旋转。不要与23格的前后半区配对混用。',
 '构造两个4×4嵌入旋转，角pi/6与pi/3，torch.mm相乘得R。torch.tensor([...tensor表达式...])会拷贝数值并切断这些角度的原计算图；适合展示，不是可微构造范例。',
 '第一行仅检验RR转置与R转置R相同，正规矩阵也满足，单独不足证明正交；第二行与单位阵比较才给出当前例子的正交数值检查。',
 '图中仍是相邻配对的逐元素cos/sin实现；接下来23格函数改用前后半区配对，必须额外说明维度置换，不能当作同一坐标顺序的直接延续。',
 '完整get_rope_matrix：cos/sin各[D/2]，填入四个半区对角块，配对坐标(r,r+D/2)。torch.zeros默认CPU float32，因此即使theta是double，也会写入float32矩阵。维度奇数/长度不匹配缺少显式校验。',
 'd=4；列表推导逐一把角度换成弧度。theta_m/n各[2]，只是展示两组角，不是训练得到的位置向量。若要推广到m/n任意位置，应保持同一组基础频率的线性关系。',
 'Jupyter展示两组角度tensor的tuple，不做矩阵乘法。',
 '逐元素角差，各维约一度；相对旋转保留差的正负。',
 '构造同值两元素tensor核对角差；列表推导中的i未使用，重复常数并非两个不同频率。',
 '用同一半区配对函数建立Rm/Rn，shape各[4,4]。',
 'Rm.T是转置view；用角差建立相对矩阵R(n-m)。这里矩阵T不是论文最大位置数T。',
 '实际计算Rm转置乘Rn；应与上一格R(n-m)数值接近，不能只看打印的矩阵就认为所有维度已经证明。',
 '用q/k两个[4]向量计算旋转结果；原result_1那行被注释，所以原notebook没有真正执行第一条dot比较。独立验证中另行计算三种表达式。',
 '计算q转置Rm转置Rn k，返回零维浮点tensor。',
 '计算q转置R(n-m) k；与上格比较可核对该例相对旋转。新增验证还检查反向梯度与公共位置平移。',
 '切换到CoPE论文标题；它是2024 Contextual Position Encoding，不是同名Clipped RoPE/Complex PE。',
 '论文和第三方链接保留。CoPE根据query/key门控来计数，可能学到句子/词等单位；这不是保证未经训练就能识别句子。普通相对PE并非在所有任务都无法定位句子，论文的简化假设和实验需分开看。',
 '论文Figure1原图：门控句号形成同句共享的理想位置；图中0/1门是说明用，有限logit经sigmoid通常是0到1之间小数，不是硬分类。',
 '原相对位置矩阵示意图；其i-j与论文CoPE含当前项的求和i-j+1有一个偏移约定。',
 '公式链：sigmoid gate[B,Q,K]、沿key从右往左累加为context position、在整数embedding相邻槽间插值，再加到attention logit。先算query乘位置表，避免物化[B,Q,K,D]。因果mask必须在算gate前进入logit，防止未来key参与计数。',
 '这格是Markdown，但含完整CoPE和SelfAttn两个nn.Module类，必须阅读和实际核对。提取到独立Python并导入真实torch.nn即可运行。CoPE返回位置logit偏置[B,Q,K]，SelfAttn返回对V加权后的[B,Q,Dv]；这里没有QKV投影、head拆分、残差或完整语言模型。Parameter位置表零初始化，第一步位置bias为0，需训练后才产生非零位置作用。',
]
HINTS={
 'from IPython': '导入Jupyter展示对象；输出url对象不代表网络图片已验证。',
 'import torch':'真实PyTorch提供张量和autograd，后续不是NumPy模拟模型。',
 'def precompute':'参数dim=head维度、end=位置个数、theta=基础常数；输出complex64[end,dim/2]。',
 'torch.arange(0, dim, 2)':'每对通道一个频率，指数是2r/dim；不是位置m本身。',
 't = torch.arange':'产生绝对位置0到end-1，右端不含end。',
 'torch.outer':'位置[end]外积频率[D/2]，广播关系得到所有旋转角。',
 'torch.polar':'幅值和弧度角转复数；幅值为1时得到cos+i sin。',
 'torch.ones_like':'保持角度矩阵的shape/dtype，单位圆幅值均为1。',
 'def get_rope_matrix':'参数d为偶数维，theta需有d/2个角；返回半区配对矩阵，不是相邻配对布局。',
 'mat = torch.zeros':'默认CPU float32；不会自动跟随theta的dtype/device。',
 'mat[0:d//2, 0:d//2]':'左上对角块cos，各半区对应同一频率。',
 'mat[d//2:, d//2:]':'右下对角块cos；与左上角共享角度。',
 'mat[0:d//2, d//2:]':'右上角-sin，对应二维旋转的负交叉项。',
 'mat[d//2:, 0:d//2]':'左下角+sin；与右上共同将两半区配成旋转平面。',
 'torch.mm':'二维矩阵乘法；不要用逐元素乘法代替旋转复合。',
 'torch.mv':'二维矩阵乘一维向量，输出仍为一维向量。',
 'torch.dot':'两个一维向量内积得到标量；与二维matmul的批次语义不同。',
 'torch.allclose':'容差内比较浮点结果；这里只检查例子，不是通用数学证明。',
 'Image(':'保留原相对图片路径；公开目录带有对应本地原图，远程URL另标未验证。',
 'class CoPE':'nn.Module子类，负责位置偏置；类定义本身还没有建立实例。',
 'class SelfAttn':'单head注意力示例，Q/K/V由调用者传入，不在类中训练投影。',
 'super().__init__':'先初始化Module，之后赋给属性的Parameter/子Module才能被正常登记。',
 'self.npos_max =':'保存整数位置表槽数P；合法索引0到P-1，不等于最大token长度。',
 'self.pos_emb =':'注册可训练表[1,D,P]，初始全0；不是nn.Embedding对象但同样由optimizer发现。',
 'def forward(self, query, attn_logits)':'query[B,Q,D]和已缩放且已mask的attn_logits[B,Q,K]，返回位置bias同[B,Q,K]。',
 'gates =':'sigmoid把每个内容分数映到(0,1)；屏蔽位置logit=-inf时门值为0。',
 'pos = gates.flip':'右到左累积：pos[...,j]=sum_{k=j}^{K-1}gate[...,k]；因果mask使未来贡献0。',
 'pos = pos.clamp':'上界P-1防止gather越界；超过上界的位置都落到末槽，距离区分会饱和。',
 'pos_ceil =':'向上取整再转int64索引；floor/ceil索引本身不可微。',
 'pos_floor =':'向下取整槽；小数权重仍从浮点pos保留梯度。',
 'logits_int =':'[B,Q,D]乘[1,D,P]广播成[B,Q,P]；先点积再插值避免四维位置向量。',
 'logits_ceil =':'按[B,Q,K]索引取右槽分数，输出[B,Q,K]。',
 'logits_floor =':'取左槽分数；两个gather共享query和位置表的梯度。',
 'w = pos -':'小数部分在[0,1)，pos恰为整数时两个索引相等，此代码选择的门梯度可能为0。',
 'return logits_ceil':'线性插值，前向等价于先插值embedding再与query点积。',
 'self.cope =':'注册CoPE子模块，SelfAttn.parameters会包含其位置表。',
 'self.head_dim =':'保存Q/K的最后维度D，用sqrt(D)缩放内容分数。',
 'def forward(self, query, key, val, mask)':'query/key/val为三维batch张量；mask须是允许=1/屏蔽=0的可广播张量，不是加性0/-inf mask。',
 'attn_logits = torch.bmm':'[B,Q,D]乘[B,D,K]得[B,Q,K]，bmm不自动广播batch。',
 'attn_logits = attn_logits /':'缩放内容分数；CoPE门直接使用这份已缩放分数。',
 'attn_logits += mask.log':'二值mask的log变为0/-inf，先屏蔽再计数；全屏蔽行softmax会NaN。',
 'attn_logits += self.cope':'将位置偏置加回同一logit；输出不是把position加到hidden states。',
 'attn = torch.softmax':'沿最后key维归一化；每条非空有效行权重和为1。',
 'out = torch.bmm':'[B,Q,K]乘V[B,K,Dv]返回[B,Q,Dv]，没有额外output投影。',
 'self.position_embeddings':'绝对位置表查找后与token/type逐元素相加，位置参数是可训练的。',
 'self.wte':'按token ID查token embedding，而不是按位置ID查。',
 'self.wpe':'按绝对position ID查可学习位置表。',
 'hidden_states =':'同shape向量相加；这里只是摘录，需要其余变量已在真实模型中定义。',
 'apply_rotary_emb':'对Q/K施加RoPE；V不在这个函数调用里旋转。',
 'torch.view_as_real':'复数实虚部变回二维实通道，再flatten恢复head维；需有完整前置变量。',
}

def annotate(s):
    protected=set()
    for t in tokenize.generate_tokens(io.StringIO(s).readline):
        if t.type==tokenize.STRING and t.end[0]>t.start[0]:protected.update(range(t.start[0],t.end[0]+1))
    out=[]
    for i,line in enumerate(s.splitlines(True),1):
        if i not in protected and line.strip() and not line.lstrip().startswith('#'):
            for k,hint in HINTS.items():
                if k in line:out.append(line[:len(line)-len(line.lstrip())]+'# 中文说明：'+hint+'\n');break
        out.append(line)
    return ''.join(out)

def write(p,s):
    if p.suffix!='.ipynb':
        protected=set()
        if p.suffix=='.py':
            for t in tokenize.generate_tokens(io.StringIO(s).readline):
                if t.type==tokenize.STRING and t.end[0]>t.start[0]:protected.update(range(t.start[0],t.end[0]+1))
        s=''.join(line if i in protected else line.rstrip()+'\n' for i,line in enumerate(s.splitlines(True),1)).rstrip()+'\n'
    p.write_text(s,encoding='utf-8',newline='\n')

def main():
    raw=json.loads((FOLDER/'rope_cope.ipynb').read_bytes());assert len(raw['cells'])==len(NOTES)==40
    variants=[copy.deepcopy(raw),copy.deepcopy(raw)]
    for n in variants:n['cells']=[{'cell_type':'markdown','metadata':{},'source':['# 第28讲原单元与中文说明\n','编号从0开始，保留上游历史输出；实际验证另见复现目录。\n']}]
    reading=['# 第28讲原notebook Python阅读版，新增内容均为注释。\n# Markdown中的CoPE完整实现另见CoPE核心源码_代码注释.py。\n']
    units=['# 第28讲全部40个原单元\n\n代码、Markdown公式/源码与仅注释行均按0起编号说明。\n']
    for i,c in enumerate(raw['cells']):
        s=''.join(c['source']);note=f'## 单元 {i}（{c["cell_type"]}）\n\n{NOTES[i]}\n'
        units.append(note+'\n~~~'+('python' if c['cell_type']=='code' else 'markdown')+'\n'+s+'\n~~~\n\n')
        for n,changed in zip(variants,[False,True]):
            n['cells'].append({'cell_type':'markdown','metadata':{},'source':note.splitlines(True)});new=copy.deepcopy(c)
            if changed:
                if c['cell_type']=='code':new['source']=('# 本格说明：'+NOTES[i]+'\n'+annotate(s)).splitlines(True)
                elif i in [5,9,39]:
                    new['source']=re.sub(r'```[^\n]*\n(.*?)```',lambda m:'```python\n'+annotate(m.group(1))+'```',s,flags=re.S).splitlines(True)
            n['cells'].append(new)
        reading.append('\n# ===== 原单元 '+str(i)+' =====\n# '+NOTES[i]+'\n')
        reading.append(annotate(s)+'\n' if c['cell_type']=='code' else ''.join('# '+line+'\n' for line in s.splitlines()))
    for name,n in zip(['原代码_带说明.ipynb','代码注释.ipynb'],variants):write(FOLDER/name,json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    write(FOLDER/'rope_cope_代码注释.py',''.join(reading));write(ROOT/'代码单元解释.md',''.join(units))
    core=re.search(r'```[^\n]*\n(.*?)```',''.join(raw['cells'][39]['source']),re.S).group(1)
    write(FOLDER/'CoPE核心源码_代码注释.py','# CoPE/SelfAttn完整实现，来自原单元39，对照论文2405.18719v2附录B Listing1。\n# 作者Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar，FAIR at Meta。\n# 本交付新增中文注释与必要imports；未找到并锁定独立官方工程，不冒称其Git源码。\nimport math\nimport torch\nfrom torch import nn\n\n'+annotate(core))
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [FOLDER/'rope_cope.ipynb',*(ROOT/'代码/imgs').glob('*.png')]}
    manifest={'upstream_commit':'7fbc29824171fd07dc2c30fe922e61ebc018ad1d','files':files,'paper':'https://arxiv.org/abs/2405.18719v2','paper_code':'Appendix B Listing1，已通过PDF文本逐行核对；不是Git仓库原文件','official_repository':'未找到可核实并锁定的独立官方工程；论文附录为本讲实现依据','subtitle':'E:/05717ac9-bf18-46c2-b4c4-3ea2f2c1a2ca.txt，仅本地保存','remote_image_cell4':'URL原样保留；未验证图片内容','pdf_download':'系统curl TLS握手失败；通过web读取固定v2 PDF，不声明本地原论文.pdf存在'}
    write(ROOT/'来源清单.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print('Built40cells/25code, two81-cell derivatives, two Python readings; Markdown5/9/39 code annotated.')

if __name__=='__main__':main()
