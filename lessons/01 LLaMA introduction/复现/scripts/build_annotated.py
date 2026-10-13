"""生成保留原单元的说明版与中文代码注释版；不执行大模型命令。"""
import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

COMMENTS = {
    0: '# 【必须会】这行准备 notebook 的图片展示工具，后面的单元 4 会用到。\n'
       '# from 包.模块 import 名称：从 IPython.display 模块导入 Image 类。\n'
       '# Image 是类；本行只把这个类绑定到当前内核的名字 Image，还没有创建实例。\n'
       '# import 会加载展示模块，但不会读取 llama_weights.png、下载模型或分配模型张量。\n'
       '# 本单元没有输入参数和返回表达式；状态变化是命名空间中新增 Image。\n'
       '# 先运行本格，再运行单元 4；否则 Image 未定义会触发 NameError。\n'
       '# 它是展示辅助代码，未实现论文中的 Attention、RMSNorm 或训练损失。\n',
    4: '# 【必须会】本行展示权重目录的 PNG 截图，便于认识 checkpoint 文件布局。\n'
       '# Image(...) 调用上一格导入的类，构造 IPython.display.Image 实例。\n'
       '# 第一个位置实参是 str 路径；在本例中它被当作本地文件读取。\n'
       '# 相对路径基于内核当前工作目录：先由 .. 返回上一级，再进入 imgs。\n'
       '# 请从本讲的代码/tutorials 目录启动内核；配套图片位于代码/imgs。\n'
       '# 若从仓库根目录执行，此路径可能指向错误位置，并触发 FileNotFoundError。\n'
       '# width=300 是关键字实参，类型为 int，表示前端展示宽度为 300 像素。\n'
       '# 该参数不重新采样 PNG，也不是 embedding_dim、batch_size 或模型参数量。\n'
       '# 数据流：路径字符串 -> PNG 文件字节 -> Image 对象 -> notebook 展示输出。\n'
       '# 默认 embed=None；本地文件情形会嵌入图片数据，文件内容保存在对象的 data 属性。\n'
       '# 这里没有 torch.Tensor，也没有 [B,C,H,W] 或 [B,T,D] 的模型 shape。\n'
       '# 返回对象没有赋给变量；IPython 会显示本格最后一个表达式的展示表示。\n'
       '# 普通 Python 脚本运行这行只创建对象；要显示需额外使用 display 或 notebook。\n'
       '# 截图中的 .pth、params.json、tokenizer.model 只是图片内容，并未被本行加载。\n'
       '# 原格保留的图片输出是作者历史记录；本次重新读取的结果另记在复现文档。\n',
}


def comment_commands(text):
    """只在完整命令前添加 Bash 注释，不打断反斜杠续行。"""
    result = []
    for line in text.splitlines(keepends=True):
        if line.startswith('torchrun '):
            count = re.search(r'--nproc_per_node[ =](\d+)', line).group(1)
            scale = {'1': '7B', '2': '13B', '4': '33B'}[count]
            result.extend([
                '# 【必须会】下面展示 %s 的历史推理启动命令；本次没有执行。\n' % scale,
                '# torchrun 是分布式进程启动器；不是模型类，也不会自动提供权重。\n',
                '# --nproc_per_node 的值 %s 表示每节点启动 %s 个进程，不是 batch_size。\n' % (count, count),
                '# example.py 是外部入口；启动器之后的参数由该脚本解析。\n',
                '# --ckpt_dir 指向模型 checkpoint 目录；--tokenizer_path 指向分词模型文件。\n',
                '# ./weights 相对于启动时的工作目录；主教程未包含这些权重和该入口脚本。\n',
                '# 第一代模型并行需核对分片数与进程数，不能仅为省显存随意改进程数。\n',
                '# 行末反斜杠是 Bash 续行符；续行内部不要插入独立注释。\n',
                '# PowerShell 不使用此续行符；这里保留原 Bash 写法供阅读。\n',
                '# 参数可写为 --名称 值 或 --名称=值；两种格式不表示不同生成算法。\n',
            ])
        result.append(line)
    return ''.join(result)


def export_readable(notebook, target):
    """用 # %% 分格导出可直接阅读的 Python，不携带 JSON 和历史图片字节。"""
    parts = ['# 第 01 讲代码注释阅读版\n'
             '# 来源：chunhuizhang/personal_chatgpt；固定提交 7fbc29824171fd07dc2c30fe922e61ebc018ad1d。\n'
             '# 建议在配套 notebook 中运行；本文件仅导入 Image 并创建展示对象。\n'
             '# Markdown 和六条 shell 命令均以 Python 注释保留，不会执行模型启动。\n\n']
    for cell in notebook['cells']:
        text = source(cell)
        if cell['cell_type'] == 'code':
            parts.append('# %%\n' + text.rstrip() + '\n\n')
        else:
            parts.append('# %% [markdown]\n' + '\n'.join(('# ' + line).rstrip() for line in text.splitlines()) + '\n\n')
    Path(target).write_text(''.join(parts).rstrip() + '\n', encoding='utf-8')


def source(cell):
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


def generate(notebook, explanation, output_dir):
    raw = Path(notebook).read_bytes()
    original = json.loads(raw.decode('utf-8-sig'))
    text = Path(explanation).read_text(encoding='utf-8')
    matches = list(re.finditer(r'^## 单元 (\d+)：.*$', text, re.M))
    notes = {}
    for pos, match in enumerate(matches):
        end = matches[pos + 1].start() if pos + 1 < len(matches) else text.index('## 覆盖、学习优先级')
        notes[int(match.group(1))] = text[match.start():end].strip()
    cells = original['cells']
    assert set(notes) == set(range(len(cells))) == set(range(9))
    assert [i for i, c in enumerate(cells) if c['cell_type'] == 'code'] == [0, 4]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'original_cells': 9,
              'explained_cells': sorted(notes), 'code_cells': [0, 4],
              'commented_code_cells': sorted(COMMENTS), 'shell_command_cells': [6, 7],
              'shell_commands_count': 6, 'shell_commands_executed': False,
              'commented_markdown_cells': [6, 7],
              'original_cells_preserved': True, 'commented_ast_equal': True,
              'historical_outputs_preserved': True, 'artifacts': {}}
    for filename, commented in [('原代码_带说明.ipynb', False), ('代码注释.ipynb', True)]:
        result = copy.deepcopy(original)
        result['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'id': 'reading-header',
                            'source': '# 第 01 讲中文说明\n\n插入的说明为新编内容；原输出是历史记录，不表示本次执行。\n'
                                      '工作目录应为本讲代码/tutorials；原版、图片和 Python 阅读版均随仓库提供。'}]
        for i, cell in enumerate(cells):
            result['cells'].append({'cell_type': 'markdown', 'metadata': {},
                                    'id': 'explain-%02d' % i, 'source': notes[i]})
            original_cell = copy.deepcopy(cell)
            if commented and i in COMMENTS:
                original_cell['source'] = COMMENTS[i] + source(cell)
                assert ast.dump(ast.parse(source(cell))) == ast.dump(ast.parse(source(original_cell)))
            elif commented and i in (6, 7):
                original_cell['source'] = comment_commands(source(cell))
                without_comments = ''.join(line for line in source(original_cell).splitlines(keepends=True)
                                           if not line.startswith('# '))
                assert without_comments == source(cell)
            else:
                assert original_cell == cell
            result['cells'].append(original_cell)
        target = output_dir / filename
        target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        record['artifacts'][filename] = {'cells': len(result['cells']),
                                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        if commented:
            export_readable(result, output_dir / '01_llama_intro_代码注释.py')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', required=True)
    parser.add_argument('--explanation', default=str(Path(__file__).resolve().parents[2] / '代码单元解释.md'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    record = generate(args.notebook, args.explanation, args.output_dir)
    Path(args.record).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))
