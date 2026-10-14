# -*- coding: utf-8 -*-
"""All main/overview cells, preserving original statements and historical outputs."""
import copy
import json
from pathlib import Path

MAIN={
0:('导入框架','【必须会】torch提供实/复数Tensor与运算；nn为神经网络模块；np是NumPy别名。导入不加载模型。'),
1:('复平面章节','【必须会】相邻两个实通道可视作一个复数，尚未加入位置。'),
2:('Euler公式与指数','【必须会】i²=-1，exp(iθ)=cosθ+i sinθ。a+ib的指数半径exp(a)、角度b；Python用1j。exp(iπ)+1约0受浮点误差影响，不是额外模型步骤。'),
3:('单位圆章节','【建议会】半径为1的旋转因子与被旋转向量的长度不同。'),
4:('单位模长','【必须会】exp(iθ)的实部/虚部平方和为1；乘它只旋转二维向量，不改变长度。'),
5:('画单位圆','【必须会】linspace含两端，500个float64角度；np.exp(1j*thetas)为complex128[500]，real/imag为float64[500]。plot修改当前图，axis equal令坐标单位等长。不是loss或模型评测曲线。'),
6:('传统位置表章节','【必须会】下面先生成sin/cos特征表，既没有加到embedding，也没有旋转Q/K。'),
7:('奇偶位置公式','【必须会】频率base^(-2j/d)，相邻通道sin/cos使用同一频率。位置t、通道对j与维度d区别明确；d要为正偶数。'),
8:('配置小表','【必须会】max_sequence_length=10为位置数量，d_model=6为教学表宽度；实际Llama precompute的dim是head_dim。'),
9:('生成通道索引','【必须会】arange右端不含，even_i=[0,2,4]、odd_i=[1,3,5]，float32[3]；dtype检查被注释，没有执行。print显示索引。'),
10:('位置向量','【必须会】position为float32[10]，值0到9，不含10；裸表达式只在Notebook自动展示。'),
11:('sin表','【必须会】1/pow(10000,even_i/d_model)是[3]频率；outer([10],[3])→[10,3]角度，再sin得到偶数通道。outer不对通道求和。'),
12:('cos表','【必须会】odd_i-1=[0,2,4]使频率与sin一致，得到cos float32[10,3]；不能直接用奇数频率。'),
13:('交错排列位置表','【必须会】stack dim2→[10,3,2]，末轴顺序[sin,cos]；flatten轴1到2→[10,6]，保持位置轴，不做归约。'),
14:('实数位置表模块','【必须会】class继承nn.Module；__init__、self、super初始化实例和基类。配置为普通字段，无Parameter。forward不接受x，每次重新构造[10,6]，不是缓存buffer，默认CPU；d为奇数时两支列数不同而stack报错。'),
15:('调用实数类','【必须会】spe是对象，spe()调用forward，返回[10,6]；创建对象不等于已经计算表。保存此类引用可与后面同名新类比较。'),
16:('复数位置表章节','【必须会】用于重建传统表的复数表示和真正RoPE旋转仍需区分。'),
17:('Llama来源小节','【建议会】下面函数来自Llama，精确提交见来源清单。'),
18:('官方源码入口','【建议会】原main链接会变化；本篇固定Meta提交7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e。'),
19:('预计算单位复数','【必须会】普通函数，无self；dim偶数通道、end位置数量、theta默认10000为base。arange步长2、截取dim//2、float()/dim、幂倒数→频率[dim/2]；t[end]和outer→相位[end,dim/2]；ones_like半径1、polar→complex64同shape。freqs变量由频率变相位，两次print仅显示shape。无可学习参数或状态更新。'),
20:('调用参数说明纠正','【必须会】独立表例dim=d_model=6；真实模型dim=head_dim=D/H，不是整个模型宽度。end不包含端点，官方调用预留max_seq_len*2；theta是基数，不是直接角度。'),
21:('生成复数表','【必须会】dim6/end10→complex64[10,3]，0位置全1+0j；两次shape打印后返回表，不旋转任何Q/K。'),
22:('复数恢复传统表','【必须会】view_as_real→float32[10,3,2]末轴[cos,sin]；flip轴2变[sin,cos]，flatten→[10,6]。单元素tuple(2,)尾逗号不能丢。flip适用于恢复传统表，不应额外用于实际Q/K旋转。'),
23:('cis命名','【建议会】cis意为cosine+i sine，描述单位复数因子。'),
24:('实虚部排列','【必须会】exp(ix)实部cos、虚部sin；传统表顺序sin/cos相反，故重建时flip。complex64含两个float32，元素数减半不代表存储减半。'),
25:('复数模块章节','【建议会】下面同名类将遮盖原类名，但不会改变已经创建的旧spe实例。'),
26:('同名类与全局变量缺陷','【必须会】新forward只构造even_i和outer/polar，再view/flip/flatten。原分母写的是全局d_model，而非self.d_model！在当前全局6/实例6时正常；改实例宽度会依赖外部状态，无全局变量则NameError。注释版保留此语句，独立补充只修正这一Name再验证。奇数宽度5实际会输出6列而非报错，不能说正确支持奇数维。'),
27:('调用新类','【必须会】重新创建spe后调用新forward；输出[10,6]仅在当前全局d_model=6满足预期。与实数类按float32容差比较，不能强求bit相等。'),
28:('空代码格','【建议会】原空格保留，仅加说明，不补造算法。'),
}
OVERVIEW={
0:('导入展示与Torch','【建议会】Image读取配图，不执行注意力；torch/nn提供位置表计算。'),
1:('基础章节','【建议会】整理位置机制背景。'),
2:('RoFormer来源','【必须会】固定v4论文的二维/多维旋转；博客仅辅助，模型依赖不能与bert4keras路线混用。'),
3:('相对点积公式纠正','【必须会】分数仍依赖内容q/k；复数点积必须Re(zq*conj(zk))，原简写省略内容共轭与取实部。m-n和n-m要随约定一致。0<epsilon<2π/N不是Meta多频率函数的限制，不能据此断言可无限外推。'),
4:('sin位置表章节','【必须会】传统表作为比较，不是已应用RoPE。'),
5:('共享频率公式','【必须会】偶数通道i用sin、奇数通道i用cos且频率索引i-1；分母指数必须除d_model。'),
6:('配置','【必须会】位置10、通道6为小例子，不是7B的配置。'),
7:('奇偶通道','【必须会】float32[3]偶数/奇数索引，打印值；后续频率配对。'),
8:('二维位置列','【必须会】arange→[10]，reshape(-1,1)→float32[10,1]，-1推断第一轴；用于和[3]广播。主notebook改为一维outer，二者不是同一shape。'),
9:('广播计算sin/cos','【必须会】位置[10,1]/频率分母[3]→[10,3]角度，再逐元素三角函数；不需要outer但公式一致。'),
10:('查看sin表','【必须会】even_pe float32[10,3]，不改变对象。'),
11:('查看cos表','【必须会】odd_pe float32[10,3]；位置0全1。'),
12:('堆叠shape','【必须会】stack dim2→[10,3,2]；shape是torch.Size对象，不是数值表。'),
13:('展平通道','【必须会】flatten轴1到2→[10,6]按sin/cos交错，不合并位置轴。'),
14:('广播位置类','【必须会】无可学习参数，每次forward创建位置[10,1]、broadcast sin/cos、stack/flatten；与主实数类公式一致但实现路径不同。'),
15:('调用位置类','【必须会】创建spe并返回[10,6]，没有传入embedding。'),
16:('RoPE章节','【必须会】下面定位旋转结构和预计算函数。'),
17:('二维旋转矩阵','【必须会】R(mθ)是[2,2]正交矩阵，负sin在右上，正sin在左下，列向量约定正角逆时针。'),
18:('先投影再按通道对旋转','【必须会】x→Wq/Wk投影再R，矩阵乘次序不可任意交换；多维R为d/2个2×2块，[d,d]。每对独立θj。实现使用成对乘加/复数乘法，不必分配稀疏大矩阵。'),
19:('原配图中的实数实现','【必须会】Image从../imgs/hf_llama_rope.png读取600像素宽截图。图中rotate_half把末维前后两半配对，cat(-x2,x1)；apply_rotary_pos_emb将cos/sin按position_ids选取后unsqueeze轴1，广播到Q/K。此半区布局与Meta相邻通道复数配对不同，需要一致的通道排列；不能把图当成箭头示意或混用频率布局。转录详注见代码/配图_RoPE源码阅读.py，来源是此固定截图，未找到截图对应的HF版本。'),
20:('频率函数章节','【必须会】先生成固定因子，实际旋转调用归第19讲。'),
21:('Markdown源码与公式不一致','【必须会】六行函数含device=freqs.device与/div dim的正确频率。下方公式省略/dim且把端点写成包含2*max_seq_len，均需纠正；正确索引0..end-1。complex64不减少两实通道的字节数。4096/32=128是head_dim，预留2048×64不是两倍上下文质量保证。源码块在Markdown不执行，仍加逐行注释。'),
22:('重复导入','【建议会】再次import torch使用已导入模块，不重新建立模型。'),
23:('CPU示例频率函数','【必须会】与正确公式一致，t未显式device，依赖默认设备；本次CPU可运行但不能声称GPU通用。返回[end,dim/2] complex64，无shape打印或状态更新。'),
24:('生成5行正确表','【必须会】dim6/end5→[5,3]，与主例end10区别；end是数量，最后位置4。'),
25:('遗漏维度分母反例','【必须会】原手写频率1/(10000**[0,2,4])为[1,1e-8,1e-16]，遗漏/dim，不等价于正确函数。theta为int，整数幂后倒数按Torch类型转换；polar产生[5,3]。保留错误原代码用于比较，不偷偷修正。'),
26:('查看错误表shape','【必须会】torch.Size([5,3])与正确表相同不能证明值正确。'),
27:('查看实虚视图','【必须会】错误complex_tensor的view_as_real为float32[5,3,2]，末轴实/虚；shape正确不能修复频率。'),
28:('空格','【建议会】不包含计算，保留原空单元和编号。'),
}

COMMENTS={
'import torch':'导入实/复数Tensor与自动求导模块，不加载模型。',
'from torch import nn':'导入Module基类，供位置表类继承。',
'import numpy as np':'NumPy别名np用于复数指数与几何绘图。',
'import matplotlib.pyplot as plt':'绘图接口别名plt；验证选择Agg后端。',
'from IPython.display import Image':'导入配图展示类，不执行attention。',
'thetas = np.linspace(0, 2*np.pi, 500)':'500个float64弧度，包含0与2π；一个完整单位圆。',
'xs = np.real(np.exp(1j*thetas))':'复数指数complex128[500]取实部cos→float64[500]。',
'ys = np.imag(np.exp(1j*thetas))':'取虚部sin→float64[500]；Python虚数单位是1j。',
'plt.plot(xs, ys)':'连线绘单位圆，不是loss或模型性能曲线。',
"plt.axis('equal')":'两坐标轴单位等长，避免圆被显示成椭圆。',
'max_sequence_length = 10':'位置数量10，对应0..9，端点不包含10。',
'd_model = 6':'教学通道数6；真实模型频率dim应使用head_dim。',
'even_i = torch.arange(0, d_model, 2).float()':'偶数通道0/2/4，float32[3]；右端不含。',
'print(even_i)':'展示偶数通道索引，不修改数值。',
'odd_i = torch.arange(1, d_model, 2).float()':'奇数通道1/3/5，float32[3]。',
'print(odd_i)':'展示奇数通道索引。',
'position = torch.arange(max_sequence_length, dtype=torch.float32)':'一维位置float32[10]；供outer使用。',
'position = torch.arange(max_sequence_length, dtype=torch.float).reshape(-1, 1)':'位置列[10,1]，-1推断10，用广播计算[10,3]。',
'position':'Notebook展示当前位置tensor；普通Python不自动显示。',
'even_pe = torch.sin(torch.outer(position, 1./torch.pow(10000, even_i/d_model)))':'[10]外积[3]频率→[10,3]角度再sin，指数除d_model。',
'odd_pe = torch.cos(torch.outer(position, 1./torch.pow(10000, (odd_i-1)/d_model)))':'odd_i-1使sin/cos频率配对；输出[10,3]。',
'even_pe = torch.sin(position / torch.pow(10000, even_i/d_model))':'[10,1]和[3]广播→[10,3]，先按频率产生角度再sin。',
'odd_pe = torch.cos(position / torch.pow(10000, (odd_i - 1)/d_model))':'奇数列使用odd_i-1的配对频率，输出[10,3]。',
'even_pe':'展示sin表，不更新对象。','odd_pe':'展示cos表，不更新对象。',
'stacked = torch.stack([even_pe, odd_pe], dim=2)':'新轴2堆叠→[10,3,2]，末轴[sin,cos]。',
'stacked.shape':'查看torch.Size([10,3,2])，不是数值归约。',
'pe = torch.flatten(stacked, start_dim=1, end_dim=2)':'合并频率轴和二分量轴→[10,6]，保留位置轴。',
'torch.flatten(stacked, start_dim=1, end_dim=2)':'交错展平→[10,6]，没有求和或丢数据。',
'pe':'显示传统位置表，不是已经旋转的Q/K。',
'class SinPositionEncoding(nn.Module):':'定义位置表模块；同名重定义只改变名称指向，不改变旧对象。',
'def __init__(self, max_sequence_length, d_model, base=10000):':'构造配置：位置数、偶数通道数、默认频率基数10000；没有Parameter。',
'super().__init__()':'初始化Module基类；self表示当前实例。',
'self.max_sequence_length = max_sequence_length':'保存位置数量，forward每次重新生成表。',
'self.d_model = d_model':'保存实例的通道配置；与全局d_model是不同变量。',
'self.base = base':'保存频率基数，既非逐位置角度也非训练参数。',
'def forward(self):':'无输入张量，返回位置表；不缓存buffer或更新参数。',
'even_i = torch.arange(0, self.d_model, 2).float()':'按实例宽度生成偶数通道索引；奇数宽度会多一个通道对。',
'odd_i = torch.arange(1, self.d_model, 2).float()':'按实例宽度生成奇数索引；奇数宽度与even列数不等。',
'position = torch.arange(self.max_sequence_length, dtype=torch.float)':'float32一维位置[T]，当前默认CPU。',
'position = torch.arange(self.max_sequence_length, dtype=torch.float).reshape(-1, 1)':'float32位置列[T,1]，用广播与频率[通道对]组合。',
'even_pe = torch.sin(torch.outer(position, 1./torch.pow(self.base, even_i/self.d_model)))':'按实例通道宽度计算频率，outer→[T,d/2]再sin。',
'odd_pe = torch.cos(torch.outer(position,  1./torch.pow(self.base, (odd_i-1)/self.d_model)))':'配对频率计算cos表，与sin共享各对的频率。',
'even_pe = torch.sin(position / torch.pow(self.base, even_i/self.d_model))':'广播位置列与实例频率分母得到sin[T,d/2]。',
'odd_pe = torch.cos(position / torch.pow(self.base, (odd_i-1)/self.d_model))':'广播cos表；odd_i-1与偶数列频率一致。',
'return torch.flatten(stacked, start_dim=1, end_dim=2)':'输出传统交错表[T,d]；奇数维在stack处已失败。',
'spe = SinPositionEncoding(max_sequence_length=10, d_model=6)':'创建实例，不自动forward；后面同名类重新创建新对象。',
'spe()':'通过Module调用forward，返回传统位置表[10,6]。',
'def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):':'普通函数：dim为偶数通道，end位置数量，theta频率base默认10000；不旋转Q/K。',
'freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))':'偶数索引除dim作指数→幂倒数频率[dim/2]；//整数除法，不能漏/div dim。',
't = torch.arange(end, device=freqs.device)':'int64[end]位置，0..end-1；与频率同设备，type ignore不影响运行。',
't = torch.arange(end)':'CPU例未显式device，不能视为任意默认设备配置下都安全。',
'freqs = torch.outer(t, freqs).float()':'外积把频率变为相位[end,dim/2]，变量freqs被复用。',
'print(freqs.shape)':'展示实数相位shape，不表示复数dtype。',
'freqs_cis = torch.polar(torch.ones_like(freqs), freqs)':'半径1、角度相位→complex64[end,dim/2]，实部cos/虚部sin。',
'print(freqs_cis.shape)':'shape同相位表，但dtype为complex64。',
'return freqs_cis':'返回旋转因子，未应用到任何Q/K。',
'precompute_freqs_cis(6, 10)':'生成complex64[10,3]，第0行全1+0j。',
'torch.flip(torch.view_as_real(precompute_freqs_cis(6, 10)), dims=(2, )).flatten(start_dim=1, end_dim=2)':'[10,3]→[10,3,2]实虚→flip成sin/cos→[10,6]；恢复传统表才flip。',
'freqs = torch.outer(position, 1./torch.pow(self.base, even_i/d_model))':'原缺陷：分母是外部全局d_model，非self.d_model！当前都为6掩盖问题，原语句保留。',
'return torch.flip(torch.view_as_real(freqs_cis), dims=(2, )).flatten(start_dim=1, end_dim=2)':'按sin/cos恢复传统表；odd d会输出d+1列，非正确奇数维支持。',
"Image('../imgs/hf_llama_rope.png', width=600)":'读取保留的原配图，宽度600只控制显示；须在tutorials目录运行。',
'dim, end = 6, 5':'小配置6通道、5位置，与主notebook的end10不同。',
'freqs_cis = precompute_freqs_cis(dim, end)':'正确函数返回[5,3]，不旋转输入内容。',
'freqs_cis':'显示复数因子表，不是学习得到的embedding。',
'theta = 10000':'手算例频率基数，整数。',
'my_freqs_cis = torch.outer(torch.arange(end), 1./(theta ** (torch.arange(0, 6, 2))))':'原错误省略/dim，频率变成1/1e8/1e16，不等于正确表；这里变量其实是角度。',
'complex_tensor = torch.polar(torch.ones_like(my_freqs_cis), my_freqs_cis)':'按错误频率生成[5,3]单位复数；模长1或shape对不代表频率正确。',
'complex_tensor':'显示原错误频率例，注释版不偷偷修正。',
'complex_tensor.shape':'查看[5,3]，形状与正确表一致但数值不同。',
'torch.view_as_real(complex_tensor)':'共享存储的float32[5,3,2]实/虚视图，没有修复错误频率。',
}

def annotate(s):
    out=[]
    if not s.strip():out.append('# 原空代码格保留，不添加计算。\n')
    for line in s.splitlines(True):
        key=line.partition('#')[0].strip()
        if key:
            assert key in COMMENTS,key
            out.append(line[:len(line)-len(line.lstrip())]+'# '+COMMENTS[key]+'\n')
        out.append(line)
    return ''.join(out)

def markdown(s,role,i):
    if role!='overview' or i!=21:return s
    prefix,block,suffix=s.split('```')
    return prefix+'```\n'+annotate(block.lstrip('\n'))+'```'+suffix

def main():
    root=Path(__file__).resolve().parents[2];folder=root/'代码/tutorials'
    doc='# 第18讲代码单元解释\n\n[主解释](解释.md) · [阅读入口](README.md)\n\n两份原notebook共58格、35代码格；两套编号分别从0开始。全部空格、公式、原图和Markdown源码覆盖；历史输出保留，错误原语句不改。\n\n'
    for role,base,notes,suffix in [('main','llama2_src_03_RoPE_01_complex_position_encoding',MAIN,''),('overview','llama2_src_03_RoPE',OVERVIEW,'_RoPE总览')]:
        raw=json.loads((folder/(base+'.ipynb')).read_bytes())
        assert len(raw['cells'])==29 and set(notes)==set(range(29))
        for changed,name in [(False,'原代码_带说明'+suffix+'.ipynb'),(True,'代码注释'+suffix+'.ipynb')]:
            n=copy.deepcopy(raw);n['cells']=[dict(cell_type='markdown',metadata={},id='lesson18-'+role,source=['# 第18讲 '+role+'\n','保留原语句；频率/全局变量错误的修正另验证，不改原例。\n'])]
            for i,c in enumerate(raw['cells']):
                title,detail=notes[i];text='## 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
                n['cells'].append(dict(cell_type='markdown',metadata={},id='lesson18-'+role+'-'+str(i),source=text.splitlines(True)))
                saved=copy.deepcopy(c);s=''.join(c['source'])
                if changed:s=annotate(s) if c['cell_type']=='code' else markdown(s,role,i)
                saved['source']=s.splitlines(True);n['cells'].append(saved)
            (folder/name).write_bytes((json.dumps(n,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
        doc+='## '+('主notebook' if role=='main' else 'RoPE总览补充')+'\n\n'
        py=['# -*- coding: utf-8 -*-','# 第18讲 '+role+' 原代码与中文注释；错误原语句保留，修正另验证。','']
        for i,c in enumerate(raw['cells']):
            title,detail=notes[i];note='### 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
            s=''.join(c['source']);doc+=note+'\n';py+=['# %% '+str(i)]+['# '+l for l in note.splitlines()]
            if c['cell_type']=='code':doc+='```python\n'+annotate(s)+'\n```\n\n';py+=annotate(s).splitlines()
            else:
                # This document lives at lesson root, so fix only its image display path.
                doc+=markdown(s,role,i)+'\n\n';py+=['# '+l for l in markdown(s,role,i).splitlines()]
            py+=['']
        (folder/(base+'_代码注释.py')).write_bytes(('\n'.join(l.rstrip() for l in py).rstrip()+'\n').encode('utf-8'))
    (root/'代码单元解释.md').write_bytes(('\n'.join(l.rstrip() for l in doc.splitlines()).rstrip()+'\n').encode('utf-8'))
    print('58 notes/35 code cells, Markdown function, four59-cell derivatives and two reading Python files.')

if __name__=='__main__':main()
