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
    'input_dim =': '输入特征宽度k；这里是整数配置，不是batch大小。',
    'output_dim =': '输出特征宽度d；后续输出最后一维为768。',
    'rank =': '低秩瓶颈r=8；两个小因子的共同维度。',
    'W =': '原始占位符Ellipsis，不能参与矩阵乘法；需要另提供真实冻结权重。',
    'W_A =': '分配[k,r]未初始化Tensor并包装为可训练Parameter；此行未注册到模型属性。',
    'W_B =': '分配[r,d]可训练Parameter，随后还需要初始化。',
    'nn.init.kaiming_uniform_': '原地随机初始化WA；math.sqrt(5)是初始化参数，不是LoRA缩放。',
    'nn.init.zeros_': '原地将WB置零，使初始LoRA输出增量为零。',
    'def regular_forward_matmul': '定义普通独立函数；没有self，不是类方法。',
    'def lora_forward_matmul': '定义两分支函数；alpha缺少定义，W仍需真实Tensor。',
    'h =': '矩阵乘法得到[N,d]基输出；@不是逐元素乘法。',
    'h +=': '原地累加增量；先WA@WB产生[k,d]；alpha未定义且未显式除r。',
    'return h': '原文return位于函数外，编译会报SyntaxError；注释版保留此缺陷。',
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
                result.append(indent + '# ' + comment + '\n')
                count += 1
                break
        result.append(line)
    return ''.join(result), count


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
            result['cells'].append({'cell_type':'markdown','metadata':{},'id':'explain-%02d'%i,'source':notes[i]})
            current = copy.deepcopy(cell)
            if commented and i in COMMENTS:
                current['source'] = '# '+COMMENTS[i]+'\n# 相对图片路径基于内核工作目录，配套图片位于上一级imgs目录。\n'+source(cell)
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
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        report['artifacts'][filename] = {'cells':len(result['cells']), 'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
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
