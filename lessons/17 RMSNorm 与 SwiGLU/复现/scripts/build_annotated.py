# -*- coding: utf-8 -*-
"""Explain all 28 cells, including the duplicate Markdown RMSNorm class."""
import copy
import ast
import hashlib
import json
from pathlib import Path

NOTES = {
0: ('基础资料章节', '【建议会】标题用于组织材料，不执行模型。'),
1: ('来源与术语', '【建议会】原链接可能指向变化的main/stable；固定课程与Meta版本见来源清单。Swish在β=1时等于SiLU；SwiGLU还需要门控双分支和输出投影。Medium是补充材料，核心公式以原论文和官方源码为准。'),
2: ('归一化章节', '【必须会】关注最后通道维度的统计，不能将batch或序列混合求均值。'),
3: ('pre-norm、公式与类预览', '【必须会】a为一个token的n维向量，g为可学习通道增益。RMS不减均值，norm位于attention/FFN之前。原文√n结论只在非零输入、epsilon=0且g全1时成立，不能无条件套用。这里的Markdown类和单元11重复，下面注释同样覆盖它；Markdown展示不执行。原文关于省时间是背景结论，本机没有测速证据。'),
4: ('导入数组与张量框架', '【必须会】np是NumPy模块别名；torch提供Tensor/随机数/自动求导；nn提供Module、Parameter与LayerNorm。导入不产生训练数据。'),
5: ('构造隐藏状态', '【必须会】多变量赋值给bs=20、seq_len=5、embedding_dim=10；torch.randn生成标准正态浮点Tensor[20,5,10]，三轴为batch、位置、通道。原格未设随机种子，所以新结果不能要求等于历史数值。'),
6: ('LayerNorm小节', '【必须会】先减每个token通道均值，再按总体方差缩放。'),
7: ('创建LayerNorm实例', '【必须会】nn.LayerNorm(10)只处理最后10维，默认eps=1e-5；weight[10]全1、bias[10]全0均可训练。ln是模块对象，不是已经归一化的数据。'),
8: ('调用LayerNorm', '【必须会】ln(x)通过nn.Module调用forward，返回x_ln[20,5,10]，不覆盖x，不更新参数。'),
9: ('检查两个token的统计', '【必须会】[1,3,:]与[1,2,:]是两个不同token的[10]向量。mean()返回0维均值；std(unbiased=False)使用分母10而不是9。epsilon使标准差略小于1；学习后的增益和偏置不保证均值0/标准差1。print只展示数值。'),
10: ('RMSNorm小节', '【必须会】使用均方根而不是中心化标准差。'),
11: ('实现RMSNorm类', '【必须会】class继承torch.nn.Module，__init__构造实例；self代表当前实例，super初始化基类。dim:int、eps:float=1e-6是类型提示/默认值，不是运行时shape检查。Parameter将ones[dim]注册成可训练weight；eps是常量。_norm中平方[ B,T,d ]→最后轴mean[ B,T,1 ]→加eps→rsqrt→广播乘x。forward先float32统计，再type_as回输入dtype，最后与weight相乘；最终dtype还受weight类型提升影响。返回同shape，不在forward更新参数；梯度和optimizer更新是后续步骤。'),
12: ('创建RMSNorm实例', '【必须会】dim=10，eps默认1e-6，只有10个weight参数，无bias或running统计。模块train/eval采用同一公式。'),
13: ('调用RMSNorm', '【必须会】x_rms[20,5,10]与x同shape，每token有独立均方[20,5,1]，所有token共享通道增益[10]。'),
14: ('查看一个原始token', '【必须会】[0,0,:]索引第一个样本/位置，返回[10]视图；Notebook末尾表达式会显示，普通Python脚本不会自动显示。'),
15: ('查看对应归一化token', '【必须会】同一索引取x_rms[10]，幅度统一调整但不减均值；不能由显示精度判断公式严格相等。'),
16: ('不带epsilon的手算', '【必须会】pow(2)逐元素平方；torch.sum全部10项→0维，除embedding_dim=10求均方，再sqrt开方，广播除回[10]。此格省略eps，非零普通输入近似匹配RMSNorm，零输入会0/0，近零时与带eps版本明显不同。'),
17: ('查看输出L2长度', '【必须会】torch.norm对该[10]向量给0维欧氏长度。eps>0、weight初值1时通常略小于√10；学习weight后不保证此长度。'),
18: ('比较理论√d', '【必须会】np.sqrt(10)返回NumPy浮点标量，约3.162277660；这是理想条件比较值，不是模型参数或第二个归一化操作。'),
19: ('SwiGLU章节', '【必须会】标题虽为SwiGLU，后面课程代码只演示激活SiLU，没有完整三矩阵FFN；官方完整模块另在复现补充验证。'),
20: ('SiLU公式与命名纠正', '【必须会】σ(z)=1/(1+exp(-z))，SiLU(z)=zσ(z)，任意shape逐元素运算。原文“也叫SiLU”不能作为SwiGLU同义解释：SwiGLU=SiLU(XW1ᵀ)⊙XW3ᵀ再乘W2ᵀ。β=1的Swish与SiLU相同，门控结果并非概率。'),
21: ('NumPy sigmoid与swish', '【必须会】def只定义普通函数，return返回结果；np.exp逐元素指数，两个函数保留输入数组shape。swish调用sigmoid再逐元素乘x。这里没有训练参数或β，极大负输入的朴素exp可能溢出。'),
22: ('准备绘图', '【建议会】plt是pyplot别名；rcParams是全局绘图配置，figure.dpi=120设置显示/保存密度，不改变数据。验证采用Agg后端以便无GUI保存。'),
23: ('绘制SiLU曲线并改变x类型', '【必须会】np.arange(-5,5,.01)生成约1000个float64值，右端5不包含。x从先前Torch三维Tensor被重新赋成NumPy[1000]；plt.plot返回Line2D列表并修改当前图。横轴输入、纵轴激活值，负侧下凹不是loss曲线或性能实验。'),
24: ('再次改变x类型', '【必须会】torch.randn(5)重新赋值，x变成float32 Tensor[5]，不再是绘图数组。独立抽取5个随机数，Notebook末表达式显示。'),
25: ('Torch手算SiLU', '【必须会】exp(-x)、加1、逐元素除法都保留[5]；不修改x或产生可学习参数。此处没有三矩阵门控。'),
26: ('导入functional接口', '【建议会】F是torch.nn.functional模块别名；F.silu是函数而不是需实例化的nn.Module。'),
27: ('使用官方SiLU函数', '【必须会】F.silu(x,inplace=False默认)返回[5]，不原地修改x；数值/梯度与β=1手算对应。训练和推理无不同分支，没有持久参数。'),
}

COMMENTS = {
'import numpy as np': 'NumPy模块用于sqrt、数组和手写激活；别名np。',
'import torch': 'Torch模块提供Tensor、随机数、rsqrt和自动求导。',
'from torch import nn': '导入神经网络子模块，后面使用LayerNorm和Parameter。',
'bs, seq_len, embedding_dim = 20, 5, 10': '同一语句解包三个整数：batch、序列位置数、隐藏通道数。',
'x = torch.randn(bs, seq_len, embedding_dim)': '标准正态Tensor[20,5,10]，原格未固定seed。',
'ln = nn.LayerNorm(embedding_dim)': '构造只归一化最后10维的模块；eps默认1e-5，带weight/bias。',
'x_ln = ln(x)': 'Module调用forward：[20,5,10]→[20,5,10]，不覆盖x。',
'print(x_ln[1, 3, :].mean())': '取样本1/位置3的10通道均值；结果0维，初始化时约0。',
'print(x_ln[1, 2, :].std(unbiased=False))': '位置2是另一个token；总体标准差分母10，加eps后不严格为1。',
'class RMSNorm(torch.nn.Module):': '继承Module以注册参数、使用forward调用和autograd。',
'def __init__(self, dim: int, eps: float = 1e-6):': '构造方法：dim为通道数，eps默认1e-6；类型提示不强制验证输入shape。',
'super().__init__()': '初始化父类Module，之后Parameter赋值才会正确注册。',
'self.eps = eps': '保存稳定项配置，不是优化器参数。',
'self.weight = nn.Parameter(torch.ones(dim))': '注册可训练通道增益[dim]，初始全1；没有beta偏置。',
'def _norm(self, x):': '内部方法：沿最后通道轴计算均方根，不减均值。',
'return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)': '平方→通道均值[B,T,1]→加eps→倒数平方根→广播乘x[B,T,d]。',
'def forward(self, x):': '模块调用的前向方法，返回同shape，参数更新不在这里发生。',
'output = self._norm(x.float()).type_as(x)': '内部用FP32统计；归一化中间结果再转回输入dtype。',
'return output * self.weight': '逐通道乘共享增益；最终dtype还取决于weight与类型提升。',
'rms_norm = RMSNorm(embedding_dim)': '创建dim10/eps1e-6实例，只有10个weight参数。',
'x_rms = rms_norm(x)': '真实RMSNorm forward：[20,5,10]，每个token独立统计。',
'x[0, 0, :]': '查看第一个token的10维原始向量，Notebook会显示末表达式。',
'x_rms[0, 0, :]': '查看同一token归一化后的10维向量。',
'x[0, 0, :] / torch.sqrt(torch.sum(x[0, 0, :].pow(2)) / embedding_dim)': '手算x/RMS，省略eps；零输入会0/0，不能声称与类严格相等。',
'torch.norm(x_rms[0, 0, :])': '欧氏L2长度标量；eps>0或学习增益时不保证√10。',
'np.sqrt(embedding_dim)': '理想g1/eps0/非零输入的理论长度√10。',
'def sigmoid(x):': '定义普通函数：标量或NumPy数组到同shape sigmoid。',
'return  1/(1 + np.exp(-x))': '逐元素1/(1+exp(-x))；极大负数时朴素exp可溢出。',
'def swish(x):': '定义β1 Swish，也就是这里的SiLU激活，而非完整SwiGLU。',
'return x*sigmoid(x)': '逐元素x乘sigmoid(x)，无矩阵或可学习参数。',
'import matplotlib.pyplot as plt': '导入绘图接口别名plt，不执行模型。',
"plt.rcParams['figure.dpi'] = 120": '修改全局图像dpi配置，不改变曲线数值。',
'x = np.arange(-5, 5, .01)': 'x重新赋成float64 NumPy[1000]；右端5不包含。',
'plt.plot(x, swish(x))': '横轴输入，纵轴SiLU；返回Line2D列表，更新当前图。',
'x = torch.randn(5)': 'x再次赋成float32 Tensor[5]，不再是绘图数组。',
'x': 'Notebook显示x；普通.py的裸表达式不会自动打印。',
'x/(1+torch.exp(-x))': '手写Torch SiLU，逐元素运算保持[5]且不覆盖x。',
'import torch.nn.functional as F': 'functional模块别名F；silu是函数。',
'F.silu(x)': '默认inplace=False，返回[5]激活；完整SwiGLU还需要三投影。',
}

def annotate(s):
    out = []
    for line in s.splitlines(True):
        if line.strip():
            out.append(line[:len(line)-len(line.lstrip())] + '# ' + COMMENTS[line.strip()] + '\n')
        out.append(line)
    return ''.join(out)

def markdown(s, i):
    if i != 3:
        return s
    prefix, block, suffix = s.split('```')
    return prefix + '```\n' + annotate(block.lstrip('\n')) + '```' + suffix

def official_reading(root):
    path=Path('G:/zzz/Personal ChatGPT 系列/17 RMSNorm 与 SwiGLU/资料/llama-model.py')
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()=='56a45f1833eedb70dd94d31dfe9d9be378a9f33458f2f5137e681873a7603754'
    s=data.decode('utf-8')
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='FeedForward')
    comments={
        'class FeedForward(nn.Module):':'完整SwiGLU前馈模块；不是单独SiLU。继承Module注册三投影参数。',
        'def __init__(':'构造签名四配置都需传入；Optional只表示允许None，签名没有默认值。',
        'dim: int,':'输入/输出通道d；Llama2 7B为4096，小验证为4。',
        'hidden_dim: int,':'传入候选宽度4d，下面还会缩小与向上对齐。',
        'multiple_of: int,':'对齐倍数，如256；小验证设4。',
        'ffn_dim_multiplier: Optional[float],':'None不再缩放；ModelArgs默认None，但本构造调用必须显式提供。',
        'super().__init__()':'初始化基类以注册后续模块。',
        'hidden_dim = int(2 * hidden_dim / 3)':'从候选4d变为约8d/3，使三矩阵参数接近传统两矩阵FFN。',
        'if ffn_dim_multiplier is not None:':'is not None检查是否显式指定额外宽度比例。',
        'hidden_dim = int(ffn_dim_multiplier * hidden_dim)':'应用额外乘数并取整数，配置须与checkpoint匹配。',
        'hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)':'整数向上对齐；4096→10922→11008，不是直接四舍五入。',
        'self.w1 = ColumnParallelLinear(':'门控分支投影，逻辑权重[m,d]；world1小验证[12,4]。',
        'dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x':'无bias/不收集分支输出；lambda原样返回权重，必须加载checkpoint或另行明确初始化。',
        'self.w2 = RowParallelLinear(':'最终输出投影，权重[d,m]；小验证[4,12]。',
        'hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x':'输入已按中间通道划分；world1无需跨rank归约，不能据此验证多卡。',
        'self.w3 = ColumnParallelLinear(':'内容分支独立参数[m,d]，与w1没有共享weight。',
        'def forward(self, x):':'输入[B,T,d]，同位置独立计算，不更新参数，返回[B,T,d]。',
        'return self.w2(F.silu(self.w1(x)) * self.w3(x))':'双投影[B,T,m]→SiLU门控逐元素乘→w2恢复d；不是w1→w2→w3顺序。',
    }
    lines=['# -*- coding: utf-8 -*-', '# Copyright (c) Meta Platforms, Inc. and affiliates.',
        '# Llama 2 Community License Agreement applies to the original source excerpt.',
        '# 固定来源：https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/model.py',
        '# 官方类的原语句和docstring保留，只添加中文注释；这是源码阅读补充，不是原notebook单元。',
        '# 实际验证加载完整未修改model.py，使用已初始化的单进程Gloo/FairScale与显式随机权重。',
        'from typing import Optional', 'from torch import nn', 'import torch.nn.functional as F',
        'from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear', '']
    for line in ast.get_source_segment(s,node).splitlines():
        if line.strip() in comments:
            lines.append(line[:len(line)-len(line.lstrip())]+'# '+comments[line.strip()])
        lines.append(line)
    (root/'代码/SwiGLU_源码阅读.py').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))

def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / '代码/tutorials'
    raw = json.loads((folder/'llama2_src_02_RMSNorm_SwiGLU.ipynb').read_bytes())
    assert len(raw['cells']) == 28 and set(NOTES) == set(range(28))
    doc = '# 第17讲代码单元解释\n\n[主解释](解释.md) · [阅读入口](README.md)\n\n覆盖全部28格、20代码格及Markdown3类。编号从0开始；历史输出保留，新执行结果另存。\n\n'
    py = ['# -*- coding: utf-8 -*-', '# 第17讲原代码阅读版；未改变语句，裸表达式在脚本中不会自动显示。', '']
    for changed, name in [(False, '原代码_带说明.ipynb'), (True, '代码注释.ipynb')]:
        n = copy.deepcopy(raw)
        n['cells'] = [dict(cell_type='markdown', metadata={}, id='lesson17-intro', source=['# 第17讲说明与注释\n', 'RMSNorm与SiLU原格；完整SwiGLU真实验证属于独立补充。\n'])]
        for i, c in enumerate(raw['cells']):
            title, detail = NOTES[i]
            note = '## 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
            n['cells'].append(dict(cell_type='markdown', metadata={}, id='lesson17-note-'+str(i), source=note.splitlines(True)))
            saved = copy.deepcopy(c)
            s = ''.join(c['source'])
            if changed:
                s = annotate(s) if c['cell_type']=='code' else markdown(s,i)
            saved['source'] = s.splitlines(True)
            n['cells'].append(saved)
        (folder/name).write_bytes((json.dumps(n,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    for i,c in enumerate(raw['cells']):
        title,detail = NOTES[i]
        note = '## 单元 {}（{}）：{}\n\n{}\n'.format(i,c['cell_type'],title,detail)
        s = ''.join(c['source'])
        doc += note+'\n'
        py += ['# %% '+str(i)]+['# '+x for x in note.splitlines()]
        if c['cell_type']=='code':
            doc += '```python\n'+annotate(s)+'\n```\n\n'
            py += annotate(s).splitlines()
        else:
            doc += markdown(s,i)+'\n\n'
            py += ['# '+x for x in markdown(s,i).splitlines()]
        py += ['']
    (root/'代码单元解释.md').write_bytes(('\n'.join(line.rstrip() for line in doc.splitlines()).rstrip()+'\n').encode('utf-8'))
    (folder/'llama2_src_02_RMSNorm_SwiGLU_代码注释.py').write_bytes(('\n'.join(x.rstrip() for x in py).rstrip()+'\n').encode('utf-8'))
    official_reading(root)
    print('28 notes, 20 annotated code cells, Markdown class, two57-cell notebooks and reading Python.')

if __name__ == '__main__':
    main()
