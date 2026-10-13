"""生成保留原单元的说明版与中文代码注释版；不执行大模型命令。"""
import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

COMMENTS = {
    0: '# 导入 IPython 的图片展示类；这里只建立名字，不加载图片或 LLaMA 模型。\n',
    4: '# 从内核当前工作目录的上一级 imgs 目录读取 PNG 截图。\n'
       '# width=300 是展示宽度，不改变模型维度或原图像素。\n'
       '# 返回 Image 对象；notebook 会展示本单元最后一个表达式。\n',
}


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
              'original_cells_preserved': True, 'commented_ast_equal': True,
              'historical_outputs_preserved': True, 'artifacts': {}}
    for filename, commented in [('原代码_带说明.ipynb', False), ('代码注释.ipynb', True)]:
        result = copy.deepcopy(original)
        result['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'id': 'reading-header',
                            'source': '# 第 01 讲中文说明\n\n插入的说明为新编内容；原输出是历史记录，不表示本次执行。\n'
                                      '工作目录应为配套代码/tutorials；原始 notebook 另存。'}]
        for i, cell in enumerate(cells):
            result['cells'].append({'cell_type': 'markdown', 'metadata': {},
                                    'id': 'explain-%02d' % i, 'source': notes[i]})
            original_cell = copy.deepcopy(cell)
            if commented and i in COMMENTS:
                original_cell['source'] = COMMENTS[i] + source(cell)
                assert ast.dump(ast.parse(source(cell))) == ast.dump(ast.parse(source(original_cell)))
            else:
                assert original_cell == cell
            result['cells'].append(original_cell)
        target = output_dir / filename
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        record['artifacts'][filename] = {'cells': len(result['cells']),
                                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
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
