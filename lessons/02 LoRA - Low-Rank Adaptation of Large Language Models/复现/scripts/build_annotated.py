"""为第02讲所有原单元插入说明，另生成只加中文注释的版本。"""
import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

CODE_CELLS = [0, 4, 7, 8, 10, 14, 15]
COMMENTS = {
    0: '导入notebook图片展示类；不创建模型，也不读取权重。',
    4: '读取LoRA论文公式截图；width=600仅指定展示宽度，公式没有在此执行。',
    7: '展示完整微调的三个阶段示意；只读取PNG，不执行反向传播。',
    8: '展示基权重与完整增量两个分支；width=500不是模型特征维度。',
    10: '展示低秩增量的双分支结构；没有在此构造WA/WB或进行训练。',
    14: '展示历史BLOOM结构截图；不下载模型；省略width使用默认展示行为。',
    15: '展示LoRA插入融合QKV后的历史截图；图中rank=16，不是伪代码的8。',
}
PSEUDO_COMMENTS = {
    'input_dim =': '【必须会】input_dim 是整数配置，记作输入特征宽度 k=768，不是 batch_size。\n'
                   '本格是 Markdown 文本，不会随 notebook 的 Run Cell 执行。\n'
                   '原文未导入 torch、torch.nn as nn 和 math；此注释版不补写执行语句。',
    'output_dim =': 'output_dim 是输出特征宽度 d=768，不是词表大小，也不是生成 token 数。\n'
                    '若输入 X 为 [N,k]，线性层输出 H 为 [N,d]；N 可理解为本例的样本数。',
    'rank =': 'rank 是低秩瓶颈 r=8，不是输入矩阵 X 的实际 rank。\n'
              'Delta W = W_A @ W_B 的秩至多为 r；两个因子共有 r(k+d)=12288 个参数。\n'
              '原大矩阵有 kd=589824 个参数；这里比较可训练因子数，不等于整体显存比例。',
    'W =': '... 是 Python 的 Ellipsis 对象，不是自动读取预训练权重的指令。\n'
           '真实行向量约定需要 W 为浮点 Tensor [k,d]=[768,768]，与 X 设备及 dtype 相容。\n'
           '完整 LoRA 实现还应冻结基权重；本赋值既没有真实 Tensor，也没有 requires_grad_(False)。\n'
           'nn.Linear 存储权重为 [out,in]=[d,k]；不能直接把它的 weight 当成本例 W 而忽略转置。',
    'W_A =': 'torch.empty(input_dim, rank) 分配 [k,r]=[768,8] Tensor；empty 不保证填零。\n'
             'nn.Parameter(...) 包装为默认 requires_grad=True 的可学习叶子参数。\n'
             '原文没有指定 dtype/device，遵从当前默认；该行的 nn 和 torch 仍需导入。\n'
             '这里只是独立变量；赋给 nn.Module 的属性才会被模块注册为参数。',
    'W_B =': '第二个可学习因子形状为 [r,d]=[8,768]，与 W_A 的内侧维度 8 匹配。\n'
             '两个因子乘积是 [k,d]；W_B 的命名不表示它是 batch 维度 B。\n'
             'empty 分配的初值未定义，后面 zeros_ 初始化完成后才适合前向使用。',
    'nn.init.kaiming_uniform_': '原地随机初始化 W_A；末尾下划线表示修改传入 Tensor 的内容。\n'
                              'a=math.sqrt(5) 是初始化分布参数，不是 LoRA 的 alpha 或 alpha/r。\n'
                              '按本例 [k,r] 布局，二维初始化的 fan_in 是第二维 r。\n'
                              'PEFT Linear 的 A 权重通常存为 [r,k]，其 fan_in 是 k；不能宣称初始化分布相同。',
    'nn.init.zeros_': '原地把 W_B 全部置零，使初始 W_A @ W_B=0，初始输出等于基模型输出。\n'
                     '若之后有非零损失梯度，B 通常先获得梯度；因 B=0，A 的第一次梯度为零。\n'
                     '这不表示 A 永远不学习；更新 B 后，A 可以有非零梯度。两因子均为零会让两边起始梯度都为零。',
    'def regular_forward_matmul': 'def 定义独立函数，不是类方法；没有 self，也没有默认参数。\n'
                                  '预期输入 x:[N,k] 与 W:[k,d]，计算基线性层输出 [N,d]。\n'
                                  '函数体由缩进划定；下面原 return 未缩进，因此原文件不能直接编译执行。',
    'def lora_forward_matmul': '独立函数接收 x、W、W_A、W_B，没有默认值；alpha 将从外部作用域查找。\n'
                               '预期数据流：基分支 x@W 和低秩增量分支相加，输出 [N,d]。\n'
                               '原文 alpha 未定义、W 为占位、return 缩进不合法；不要把本格当成可运行训练器。',
    'h =': '@ 是矩阵乘法，不是逐元素乘法；[N,k] @ [k,d] -> h:[N,d]。\n'
           '该表达式创建基输出，不修改 W；是否冻结 W 由 requires_grad/优化器设置决定。',
    'h +=': '先 W_A@W_B：[k,r]@[r,d] -> [k,d]，再 x@该增量 -> [N,d]。\n'
            '乘 alpha 后与同 shape 的 h 相加；h += 修改输出 h，不是修改基权重 W。\n'
            '这里显式构造大增量 [k,d]；教学实现改用 (x@W_A)@W_B，经过 [N,r] 中间特征。\n'
            '论文常用缩放 s=alpha/r；原文只乘 alpha，无法判断作者是否已把 r 吸收到 alpha 中。\n'
            '实数下结合律给出等价表达式；浮点下只能按误差容差比较，不能保证逐位一致。',
    'return h': '【原文缺陷】此 return 没有缩进到函数体，compile 会报 return outside function。\n'
                '只有放进函数体的 return 才会向调用者返回 h；注释版保留原行，不静默修复。\n'
                '原格也没有数据、loss、optimizer 或训练循环；可运行补充见复现/scripts/verify_lora.py。',
}


def source(cell):
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


def pseudocode(cell):
    match = re.search(r'```\s*\n(.*?)\n```', source(cell), re.S)
    assert match, 'Markdown13围栏未找到'
    return match.group(1)


def annotate_pseudo(text):
    lines = text.splitlines(keepends=True)
    result = []
    count = 0
    for line in lines:
        stripped = line.strip()
        for prefix, comment in PSEUDO_COMMENTS.items():
            if stripped.startswith(prefix):
                indent = line[:len(line) - len(line.lstrip())]
                result.extend(indent + '# ' + row + '\n' for row in comment.splitlines())
                count += 1
                break
        result.append(line)
    return ''.join(result), count


def annotate_code(index, cell):
    text = source(cell)
    if index == 0:
        rows = [COMMENTS[index],
                'from 模块 import 名称：把 IPython.display 中的 Image 类绑定到当前内核名字。',
                '类是创建对象的模板；这里只导入类，尚未调用 Image(...) 创建展示对象。',
                '后续六格需要先运行本格；否则会因为 Image 未定义触发 NameError。',
                '本格没有参数张量、dtype、shape、loss 或参数更新；它是展示辅助代码。']
    else:
        path = re.search(r"Image\('([^']+)'", text).group(1)
        width = {4:600, 7:600, 8:500, 10:600, 14:None, 15:None}[index]
        rows = [COMMENTS[index],
                'Image(...) 调用展示类构造实例；第一个位置实参是 str 本地图片路径。',
                '原路径 %s 由内核当前工作目录解析，不自动以 notebook 文件位置为基准。' % path,
                '从本讲代码/tutorials 启动内核，.. 才会指向代码，再到配套 imgs 目录。',
                '若图片路径不可用会报 FileNotFoundError；这不是模型加载失败。',
                ('width=%s 是 int 关键字参数，控制前端展示宽度，不修改原图像素。' % width) if width else
                '原代码未传 width，使用 Image 默认展示宽度；它不表示模型 hidden_size 未定义。',
                '返回 IPython.display.Image 对象，data 属性包含嵌入的 PNG 字节。',
                '数据流为 str 路径 -> PNG bytes -> Image 实例 -> notebook 展示，无 torch.Tensor。',
                'notebook 显示单元末尾表达式；普通 Python 脚本不会自动渲染这个对象。',
                '保留的图片输出是作者历史记录；本次实际读取与哈希比较另存复现记录。']
        if index == 4:
            rows += ['原 arxiv URL 是 Python 注释，不会发起请求或下载论文。',
                     '图对应论文低秩公式：列向量 W0 x + BAx；本教程后文采用行向量 xW + x(W_AW_B)。']
        elif index == 7:
            rows += ['图示输入 -> 原权重 -> 输出，再经完整微调更新权重；图中的箭头不是 autograd 调用。']
        elif index == 8:
            rows += ['数学上 x(W+Delta W)=xW+xDelta W；两路输出均 [N,d]，相加不是 cat 拼接。',
                     '完整 Delta W 仍为 [k,d]；增加一条画出来的分支本身不节省参数。']
        elif index == 10:
            rows += ['右支路 X:[N,k] -> XW_A:[N,r] -> XW_AW_B:[N,d]，再与 XW 相加。',
                     '基 W 冻结，两个小因子可训练；本单元只展示关系，不创建这些参数。']
        elif index == 14:
            rows += ['截图配置：Embedding 后 [batch,time,4096]；融合 QKV 投影得到 [batch,time,12288]。',
                     '12288=3*4096 是融合输出宽度，不是 attention head 数；MLP 中间宽度16384。',
                     '截图中的 BloomForCausalLM 和 Linear8bitLt 都没有被本行实例化。']
        elif index == 15:
            rows += ['图中 adapter r=16，A 权重 [16,4096]、B 权重 [12288,16]（nn.Linear 的 [out,in] 方向）。',
                     '输入 [batch,time,4096] -> 瓶颈 [batch,time,16] -> 增量 [batch,time,12288]。',
                     '单个融合 QKV 的因子数 16*(4096+12288)=262144；不代表整网总参数比例。',
                     '图中 rank 与 Markdown13 的 r=8 不同；图片未给 alpha，不能据此推断缩放值。']
    return ''.join('# ' + row + '\n' for row in rows) + text


def export_readable(notebook, target):
    parts = ['# 第02讲 LoRA 代码注释阅读版\n'
             '# 来源：chunhuizhang/personal_chatgpt，提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。\n'
             '# 不含 JSON 或历史图片编码；# %% 分格，建议使用配套 notebook 执行展示。\n'
             '# 原 Markdown13 伪代码保留为注释，含原缩进缺陷；可运行教学例另见 verify_lora.py。\n\n']
    for cell in notebook['cells']:
        text = source(cell)
        if cell['cell_type'] == 'code':
            parts.append('# %%\n' + text.rstrip() + '\n\n')
        else:
            parts.append('# %% [markdown]\n' + '\n'.join(('# ' + row).rstrip() for row in text.splitlines()) + '\n\n')
    Path(target).write_text(''.join(parts).rstrip()+'\n', encoding='utf-8')


def notebook_note(text):
    # 说明文档在本讲根目录，生成版在代码/tutorials；为插入说明中的相对链接增加两级返回。
    return re.sub(r'\]\(([^)]+)\)', lambda m: '](' + (m.group(1) if '://' in m.group(1)
                  or m.group(1).startswith('#') else '../../'+m.group(1)) + ')', text)


def generate(notebook, explanation, output_dir):
    raw = Path(notebook).read_bytes()
    original = json.loads(raw.decode('utf-8-sig'))
    cells = original['cells']
    assert len(cells) == 16
    assert [i for i,c in enumerate(cells) if c['cell_type'] == 'code'] == CODE_CELLS
    document = Path(explanation).read_text(encoding='utf-8')
    headings = list(re.finditer(r'^## 单元 (\d+)：.*$', document, re.M))
    notes = {}
    for pos, match in enumerate(headings):
        end = headings[pos+1].start() if pos+1 < len(headings) else document.index('## 覆盖、学习优先级')
        notes[int(match.group(1))] = document[match.start():end].strip()
    assert set(notes) == set(range(16))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'original_cells': 16, 'explained_cells': sorted(notes), 'code_cells': CODE_CELLS,
              'commented_code_cells': sorted(COMMENTS), 'commented_markdown_code_cells': [13],
              'source_sha256': hashlib.sha256(raw).hexdigest(), 'artifacts': {}}
    for filename, commented in [('原代码_带说明.ipynb',False), ('代码注释.ipynb',True)]:
        result = copy.deepcopy(original)
        result['cells'] = [{'cell_type':'markdown','metadata':{},'id':'reading-header',
                            'source':'# 第02讲中文代码说明\n\n新增说明与注释为教学内容；原输出为历史记录。\n'
                                     '原版另存。工作目录为代码/tutorials。Markdown13保留原伪代码缺陷；可运行示例另见复现脚本。'}]
        for i, cell in enumerate(cells):
            result['cells'].append({'cell_type':'markdown','metadata':{},'id':'explain-%02d'%i,'source':notebook_note(notes[i])})
            current = copy.deepcopy(cell)
            if commented and i in COMMENTS:
                current['source'] = annotate_code(i, cell)
                assert ast.dump(ast.parse(source(cell))) == ast.dump(ast.parse(source(current)))
            elif commented and i == 13:
                current['source'], count = annotate_pseudo(source(cell))
                assert count == 15, count
                assert ast.dump(ast.parse(pseudocode(cell))) == ast.dump(ast.parse(pseudocode(current)))
                report['pseudocode_commented_statements'] = count
            else:
                assert current == cell
            result['cells'].append(current)
        target = output_dir/filename
        target.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        report['artifacts'][filename] = {'cells':len(result['cells']), 'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        if commented:
            export_readable(result, output_dir/'02_lora_baics_代码注释.py')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook',required=True)
    parser.add_argument('--explanation',default=str(Path(__file__).resolve().parents[2]/'代码单元解释.md'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--record',required=True)
    args = parser.parse_args()
    report = generate(args.notebook,args.explanation,args.output_dir)
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
