"""核对 schema、原单元/历史输出、注释 AST，并实际执行两个展示单元。"""
import argparse
import ast
import json
import os
import platform
from pathlib import Path
import IPython
import nbformat
from build_annotated import source


def execute(cells, cwd):
    namespace = {}
    results = []
    old_cwd = os.getcwd()
    try:
        os.chdir(str(cwd))
        for i, cell in enumerate(cells):
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(source(cell))
            value = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                expr = tree.body.pop()
                exec(compile(tree, '<cell>', 'exec'), namespace)
                value = eval(compile(ast.Expression(expr.value), '<cell>', 'eval'), namespace)
            else:
                exec(compile(tree, '<cell>', 'exec'), namespace)
            row = {'cell': i, 'executed': True}
            if value is not None:
                assert isinstance(value, IPython.display.Image)
                assert value.width == 300 and value.data.startswith(b'\x89PNG\r\n\x1a\n')
                row.update({'type': type(value).__name__, 'width': value.width,
                            'png_bytes': len(value.data)})
            results.append(row)
    finally:
        os.chdir(old_cwd)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, help='原版及两种注释版所在的代码/tutorials目录')
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    original = nbformat.read(str(directory / '01_llama_intro.ipynb'), as_version=4)
    nbformat.validate(original)
    records = {}
    for filename in ['原代码_带说明.ipynb', '代码注释.ipynb']:
        annotated = nbformat.read(str(directory / filename), as_version=4)
        nbformat.validate(annotated)
        assert len(annotated.cells) == 19
        preserved = annotated.cells[2::2]
        assert len(preserved) == len(original.cells)
        for i, (before, after) in enumerate(zip(original.cells, preserved)):
            a, b = dict(before), dict(after)
            if filename == '代码注释.ipynb' and before.cell_type == 'code':
                assert ast.dump(ast.parse(source(before))) == ast.dump(ast.parse(source(after)))
                assert '# ' in source(after)
                a.pop('source'); b.pop('source')
            elif filename == '代码注释.ipynb' and i in (6, 7):
                stripped = ''.join(line for line in source(after).splitlines(keepends=True)
                                   if not line.startswith('# '))
                assert stripped == source(before), (filename, i, 'shell command changed')
                a.pop('source'); b.pop('source')
            assert a == b, (filename, i)
        records[filename] = execute(preserved, directory)
    records['original'] = execute(original.cells, directory)
    assert records['original'] == records['原代码_带说明.ipynb'] == records['代码注释.ipynb']
    readable = (directory / '01_llama_intro_代码注释.py').read_text(encoding='utf-8')
    expected = '\n'.join(source(c) for c in original.cells if c.cell_type == 'code')
    assert ast.dump(ast.parse(readable)) == ast.dump(ast.parse(expected))
    report = {'status': '部分成功', 'python': platform.python_version(), 'IPython': IPython.__version__,
              'nbformat': nbformat.__version__, 'schema_valid': True, 'original_cells_equal': True,
              'historical_outputs_equal': True, 'commented_ast_equal': True,
              'markdown_commands_unchanged_after_removing_comments': True,
              'readable_python_ast_equal': True, 'runs': records,
              'not_executed': ['六条模型启动命令', 'LLaMA权重加载/推理', '论文实验']}
    Path(args.record).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
