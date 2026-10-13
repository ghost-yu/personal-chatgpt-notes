"""真实执行7个图片展示单元，核验两份注释版并记录伪代码的真实失败。"""
import argparse
import ast
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import IPython
import nbformat
from build_annotated import source, pseudocode, CODE_CELLS


def execute(cells, directory):
    scope = {}
    records = []
    previous = os.getcwd()
    try:
        os.chdir(str(directory))
        for i, cell in enumerate(cells):
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(source(cell))
            result = None
            if tree.body and isinstance(tree.body[-1],ast.Expr):
                expression = tree.body.pop()
                exec(compile(tree,'<cell>','exec'),scope)
                result = eval(compile(ast.Expression(expression.value),'<cell>','eval'),scope)
            else:
                exec(compile(tree,'<cell>','exec'),scope)
            record = {'cell':i,'executed':True}
            if result is not None:
                assert isinstance(result,IPython.display.Image)
                assert result.data.startswith(b'\x89PNG\r\n\x1a\n')
                expected = {4:600,7:600,8:500,10:600,14:None,15:None}[i]
                assert result.width == expected
                record.update({'type':'Image','width':result.width,'bytes':len(result.data),
                               'png_sha256':hashlib.sha256(result.data).hexdigest()})
            records.append(record)
    finally:
        os.chdir(previous)
    assert [r['cell'] for r in records] == CODE_CELLS
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',required=True)
    parser.add_argument('--record',required=True)
    args = parser.parse_args()
    directory = Path(args.directory)
    original = nbformat.read(str(directory/'02_lora_baics.ipynb'),as_version=4)
    nbformat.validate(original)
    runs = {'original':execute(original.cells,directory)}
    failures = {}
    for filename in ['原代码_带说明.ipynb','代码注释.ipynb']:
        result = nbformat.read(str(directory/filename),as_version=4)
        nbformat.validate(result)
        assert len(result.cells) == 33
        for note in result.cells[1::2]:
            for link in re.findall(r'\]\(([^)]+)\)', source(note)):
                if '://' not in link and not link.startswith('#'):
                    assert (directory/link).exists(), ('broken inserted-note link', link)
        preserved = result.cells[2::2]
        assert len(preserved) == 16
        for i,(before,after) in enumerate(zip(original.cells,preserved)):
            a,b = dict(before),dict(after)
            if filename == '代码注释.ipynb' and i in CODE_CELLS+[13]:
                first,second = (pseudocode(before),pseudocode(after)) if i==13 else (source(before),source(after))
                assert ast.dump(ast.parse(first)) == ast.dump(ast.parse(second))
                assert len(second) > len(first)
                # 删除新插入的行即可逐字找回原代码，不能只用AST掩盖其他改动。
                original_lines=first.splitlines(); new_lines=second.splitlines()
                at=0
                for line in new_lines:
                    if at<len(original_lines) and line==original_lines[at]:
                        at+=1
                    else:
                        assert line.lstrip().startswith('# '), (i,line)
                assert at==len(original_lines)
                a.pop('source'); b.pop('source')
            assert a==b, (filename,i)
        runs[filename]=execute(preserved,directory)
        assert runs[filename]==runs['original']
        try:
            compile(pseudocode(preserved[13]),'<markdown13>','exec')
        except SyntaxError as error:
            assert 'return' in str(error)
            failures[filename]={'type':type(error).__name__,'message':error.msg}
        else:
            raise AssertionError('原伪代码缺陷不应被静默修复')
    readable = (directory/'02_lora_baics_代码注释.py').read_text(encoding='utf-8')
    expected = '\n'.join(source(cell) for cell in original.cells if cell.cell_type=='code')
    assert ast.dump(ast.parse(readable)) == ast.dump(ast.parse(expected))
    report={'status':'部分成功','python':platform.python_version(),'IPython':IPython.__version__,
            'nbformat':nbformat.__version__,'schema_valid':True,'original_cells_and_outputs_preserved':True,
            'comment_only_and_ast_equal':True,'readable_python_ast_equal':True,
            'inserted_note_relative_links_valid':True, 'runs':runs,'markdown13_compile_failures':failures,
            'not_executed':['原伪代码训练','BLOOM加载/微调','论文数据集评估']}
    Path(args.record).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
