# -*- coding: utf-8 -*-
"""72格/52代码格核对：51可解析AST、原22错误、2条shell转换、Markdown2。"""
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager

FENCE = re.compile(r'(```[^\n]*\n)(.*?)(```)', re.S)

def tree(source):
    return ast.dump(ast.parse(TransformerManager().transform_cell(source)), include_attributes=False)

def added_comments(old, new):
    a, b = old.splitlines(True), new.splitlines(True)
    for op, i, j, u, v in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != 'equal':
            assert op == 'insert' and all(line.lstrip().startswith('# ') for line in b[u:v])

def syntax(source):
    try:
        tree(source)
    except SyntaxError as error:
        return type(error).__name__, error.msg
    raise AssertionError('原22必须保留SyntaxError')

def main():
    lesson = Path(__file__).resolve().parents[2]
    folder = lesson / '代码/tutorials'
    manifest = json.loads((lesson / '来源清单.json').read_text(encoding='utf-8'))
    for item in manifest['published_originals']:
        assert hashlib.sha256((lesson / item['published_path']).read_bytes()).hexdigest() == item['sha256']
    reports = []
    count = 0
    for filename in ('tokenizer_basics.ipynb', 'gpt2_train_tokenizer.ipynb', 'gpt2.ipynb'):
        original = json.loads((folder / filename).read_text(encoding='utf-8'))
        nbformat.validate(nbformat.from_dict(original))
        indices = [i for i, c in enumerate(original['cells']) if c['cell_type'] == 'code']
        count += len(indices)
        prefix = '' if filename == 'tokenizer_basics.ipynb' else filename.removesuffix('.ipynb') + '_'
        for suffix in ('原代码_带说明.ipynb', '代码注释.ipynb'):
            path = folder / (prefix + suffix)
            nb = json.loads(path.read_text(encoding='utf-8'))
            nbformat.validate(nbformat.from_dict(nb))
            assert len(nb['cells']) == 1 + 2 * len(original['cells'])
            assert {k: v for k, v in nb.items() if k != 'cells'} == {k: v for k, v in original.items() if k != 'cells'}
            for i, cell in enumerate(original['cells']):
                note, keep = nb['cells'][1 + 2*i:3 + 2*i]
                assert '## 单元 {}（'.format(i) in ''.join(note['source'])
                assert '【' in ''.join(note['source']) and '】' in ''.join(note['source'])
                assert {k: v for k, v in cell.items() if k != 'source'} == {k: v for k, v in keep.items() if k != 'source'}
                old, new = ''.join(cell['source']), ''.join(keep['source'])
                if suffix == '代码注释.ipynb' and cell['cell_type'] == 'code':
                    added_comments(old, new)
                    if filename == 'tokenizer_basics.ipynb' and i == 22:
                        assert syntax(old) == syntax(new)
                    else:
                        assert tree(old) == tree(new)
                elif suffix == '代码注释.ipynb' and filename == 'tokenizer_basics.ipynb' and i == 2:
                    a, b = list(FENCE.finditer(old)), list(FENCE.finditer(new))
                    assert len(a) == len(b) == 1
                    added_comments(a[0][2], b[0][2]); assert tree(a[0][2]) == tree(b[0][2])
                    assert old[:a[0].start()] == new[:b[0].start()] and old[a[0].end():] == new[b[0].end():]
                else:
                    assert keep['source'] == cell['source']
            reports.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'original_cells': len(original['cells']), 'derived_cells': len(nb['cells']),
                            'code_indices': indices, 'schema_valid': True, 'original_metadata_history_preserved': True,
                            'code_only_comment_insertions': True, 'all_parseable_AST_equal': True,
                            'original_SyntaxError22_preserved': filename == 'tokenizer_basics.ipynb'})
        raw = '\n'.join(TransformerManager().transform_cell(''.join(c['source'])) for i, c in enumerate(original['cells'])
                        if c['cell_type'] == 'code' and not (filename == 'tokenizer_basics.ipynb' and i == 22))
        path = folder / (filename.removesuffix('.ipynb') + '_代码注释.py')
        py = path.read_text(encoding='utf-8')
        assert tree(py) == tree(raw)
        if filename == 'tokenizer_basics.ipynb':
            assert all('# ' + line.rstrip() in py for line in ''.join(original['cells'][22]['source']).splitlines())
        reports.append({'reading_file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'AST_equals_all_parseable_code': True, 'reading_exception': 'basics22整格注释；basics9同值字面量排版；shell转IPython调用'})
    assert count == 52
    doc = (lesson / '代码单元解释.md').read_text(encoding='utf-8')
    assert [int(s) for s in re.findall(r'^## 单元 (\d+)（', doc, re.M)] == list(range(39)) + list(range(27)) + list(range(6))
    record = {'all72_notes_present': True, 'code_cells_total': 52, 'parseable_cells_AST_equal': 51,
              'included_IPython_shell_cells': 2, 'basics22_preserved_SyntaxError': True,
              'Markdown2_code_only_comments_and_AST_equal': True,
              'originals_and_PNG_hashes_match': True, 'notebook_schemas_valid': 9, 'variants': reports}
    (lesson / '复现/outputs/annotation_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('72 notes, 51 parseable ASTs + preserved SyntaxError22, 9 schemas, Markdown2 and 3 reading ASTs passed.')

if __name__ == '__main__':
    main()
